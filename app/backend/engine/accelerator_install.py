"""GPU 加速运行时安装（onnxruntime-gpu / onnxruntime-directml）。

为什么不直接 ``pip install onnxruntime-gpu``
--------------------------------------------
1. ``onnxruntime`` / ``onnxruntime-gpu`` / ``onnxruntime-directml`` 发布的是**同一个**
   ``onnxruntime`` 包（同名目录、同名 ``.pyd``），装在同一个环境里会互相覆盖，
   行为不可预期（谁最后写入谁生效，且残留文件可能对不上版本）。
2. Windows 上 ``onnxruntime`` 的 ``.pyd`` 一旦被 ``import`` 就被进程锁定，
   ``pip uninstall / install`` 改动 ``site-packages`` 会直接失败（WinError 5）。
   而打开硬件面板时 onnxruntime 早已被 import 过。
3. 因此这里用 ``pip --target`` 把 GPU 版装进 ``<data_dir>/runtime/<variant>`` 独立目录，
   再在**进程启动时**把该目录插到 ``sys.path`` 头部：不污染当前环境、
   删目录即可回滚、CPU 版始终作为兜底保留。

切换运行时必须重启进程：``sys.modules`` 里的 onnxruntime 与已加载的 DLL
不会因为 ``sys.path`` 变化而换掉，所以安装完成后统一返回 ``restartRequired``。
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from app.backend.core.errors import (
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationFailedError,
)
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings, get_settings
from app.backend.engine.hardware import SystemInfo, collect_system_info

logger = get_logger("engine.accelerator_install")

Variant = Literal["cuda", "directml"]
InstallState = Literal["queued", "running", "done", "error", "cancelled"]
InstallPhase = Literal["preparing", "downloading", "installing", "verifying", "done"]

# 变体 → pip 目标包。CUDA 版必须自带 cuDNN/cuBLAS 运行库，否则 EP 能枚举但建会话必失败。
VARIANT_PACKAGES: dict[str, tuple[str, ...]] = {
    "cuda": (
        "onnxruntime-gpu>=1.22",
        "nvidia-cudnn-cu12>=9",
        "nvidia-cublas-cu12",
    ),
    "directml": ("onnxruntime-directml>=1.20",),
}

VARIANT_META: dict[str, tuple[str, str, int]] = {
    "cuda": (
        "CUDA",
        "NVIDIA 显卡走 CUDA EP，吞吐最高；自动附带 cuDNN / cuBLAS 运行库",
        1_300_000_000,
    ),
    "directml": (
        "DirectML",
        "走 Windows DX12，NVIDIA / AMD / Intel 通用，速度低于 CUDA 但无需 CUDA 环境",
        350_000_000,
    ),
}

GPU_PROVIDERS = {
    "cudaexecutionprovider",
    "tensorrtexecutionprovider",
    "coremlexecutionprovider",
    "dmlexecutionprovider",
}

# CUDA 12 是 onnxruntime-gpu 1.20+ 的硬性要求，低于此版本装上也加载不了 DLL
MIN_CUDA_VERSION = (12, 0)

# 单次 pip 尝试的硬超时。大包下载期间 pip 不输出任何行，只能靠独立计时终止；
# 超时会重试，已完整下载的 wheel 留在 pip 缓存里复用，不会从头下所有包。
_PIP_ATTEMPT_TIMEOUT_SECONDS = 10 * 60
_PIP_MAX_ATTEMPTS = 3
_PIP_RETRY_DELAYS = (3.0, 8.0)
_HEARTBEAT_SECONDS = 10
_LOG_TAIL_LINES = 8
_COLLECT_RE = re.compile(r"^Collecting\s+([A-Za-z0-9_.\-]+)")
_DOWNLOAD_RE = re.compile(r"^Downloading\s+(\S+)\s+\(([\d.]+\s+\w+)\)")
_ATTEMPT_RE = re.compile(r"^\[第 (\d+)/(\d+) 次尝试\]")
_ACTIVE_FILE = "active.json"


def _creationflags(default: int = 0) -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | default


_DETACHED_FLAGS = (
    getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    if os.name == "nt"
    else 0
)


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
class AcceleratorOption(BaseModel):
    """可安装的加速后端；``blockedReason`` 非空表示当前机器装不了。"""

    variant: Variant
    label: str
    description: str
    packages: list[str]
    estimatedBytes: int
    recommended: bool = False
    blockedReason: str | None = None
    installed: bool = False


class InstallTask(BaseModel):
    id: str
    variant: Variant
    state: InstallState = "queued"
    phase: InstallPhase = "preparing"
    percent: float = 0.0
    message: str = ""
    logTail: list[str] = Field(default_factory=list)
    error: str | None = None
    restartRequired: bool = False
    startedAt: float = Field(default_factory=time.time)
    finishedAt: float | None = None


class InstalledRuntime(BaseModel):
    variant: Variant
    path: str
    sizeBytes: int = 0
    active: bool = False


class AcceleratorPayload(BaseModel):
    options: list[AcceleratorOption] = Field(default_factory=list)
    installed: list[InstalledRuntime] = Field(default_factory=list)
    active: Variant | None = None
    currentProviders: list[str] = Field(default_factory=list)
    gpuAccelerated: bool = False
    restartRequired: bool = False
    canInstall: bool = True
    blockedReason: str | None = None
    restartSupported: bool = True
    restartBlockedReason: str | None = None
    frozen: bool = False
    runtimeDir: str
    task: InstallTask | None = None


# --------------------------------------------------------------------------- #
# 目录与激活（进程启动期调用）
# --------------------------------------------------------------------------- #
def runtime_root(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / "runtime"


def variant_dir(variant: str, settings: Settings | None = None) -> Path:
    return runtime_root(settings) / variant


def _active_file(settings: Settings | None = None) -> Path:
    return runtime_root(settings) / _ACTIVE_FILE


def _read_active(settings: Settings | None = None) -> dict[str, object]:
    path = _active_file(settings)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("加速运行时配置文件损坏，已忽略：%s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _write_active(settings: Settings, payload: dict[str, object]) -> None:
    root = runtime_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    path = _active_file(settings)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_active(settings: Settings) -> None:
    path = _active_file(settings)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # noqa: BLE001
        logger.warning("清理加速运行时配置失败：%s", exc)


def active_variant(settings: Settings | None = None) -> Variant | None:
    """当前 *配置* 的运行时（不代表本进程已经加载它）。"""
    data = _read_active(settings)
    variant = str(data.get("variant") or "")
    if variant not in VARIANT_PACKAGES:
        return None
    if not _looks_installed(variant_dir(variant, settings)):
        return None
    return variant  # type: ignore[return-value]


def _looks_installed(path: Path) -> bool:
    return (path / "onnxruntime" / "__init__.py").is_file()


def _loaded_runtime_path() -> Path | None:
    """本进程实际加载的 onnxruntime/__init__.py；未 import 返回 None。"""
    module = sys.modules.get("onnxruntime")
    if module is None:
        return None
    file = getattr(module, "__file__", None)
    if not file:
        return None
    try:
        return Path(file).resolve()
    except OSError:
        return None


def _loaded_from(path: Path) -> bool:
    loaded = _loaded_runtime_path()
    if loaded is None:
        return False
    try:
        return path.resolve() in loaded.parents
    except OSError:
        return False


_dll_handles: list[object] = []
_activated_path: Path | None = None
_activation_done = False


def _dll_directories(path: Path) -> list[Path]:
    """nvidia-* wheel 把 DLL 放在 ``<pkg>/bin``（Windows）或 ``<pkg>/lib``。"""
    candidates: list[Path] = []
    nvidia = path / "nvidia"
    if nvidia.is_dir():
        for child in sorted(nvidia.iterdir()):
            if not child.is_dir():
                continue
            for sub in ("bin", "lib"):
                candidate = child / sub
                if candidate.is_dir():
                    candidates.append(candidate)
    return candidates


def _inject_dll_dirs(path: Path) -> None:
    directories = _dll_directories(path)
    if not directories:
        return
    prefix = os.pathsep.join(str(item) for item in directories)
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for item in directories:
        try:
            # 句柄必须保留引用，否则被 GC 回收后目录即失效
            _dll_handles.append(os.add_dll_directory(str(item)))
        except OSError:
            logger.debug("无法加入 DLL 搜索目录：%s", item, exc_info=True)


def activate(settings: Settings | None = None) -> str | None:
    """把已安装的加速运行时插到 ``sys.path`` 头部；幂等，可在启动路径上多次调用。

    必须在 ``import onnxruntime`` 之前执行，否则本进程只会用到 CPU 版。
    """
    global _activation_done, _activated_path
    if _activation_done:
        return _activated_path.name if _activated_path else None
    _activation_done = True
    settings = settings or get_settings()
    variant = active_variant(settings)
    if variant is None:
        return None
    path = variant_dir(variant, settings)
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    _inject_dll_dirs(path)
    _activated_path = path
    logger.info("已启用独立 onnxruntime 运行时：%s", path)
    return variant


def reset_activation_state() -> None:
    """测试用：忘记激活结果（不会卸载已 import 的模块）。"""
    global _activation_done, _activated_path
    _activation_done = False
    _activated_path = None


# --------------------------------------------------------------------------- #
# 能力判定（纯函数，便于单测）
# --------------------------------------------------------------------------- #
def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for chunk in re.split(r"[._-]", value):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts)


def _primary_gpu(system: SystemInfo):
    if not system.gpus:
        return None
    for gpu in system.gpus:
        if gpu.vendor != "unknown":
            return gpu
    return system.gpus[0]


def install_supported() -> tuple[bool, str | None]:
    """打包版没有 pip，装包能力受限；这里统一给出原因。"""
    if getattr(sys, "frozen", False):
        return False, "当前是打包版本，未内置 pip，无法在应用内安装 GPU 运行时"
    try:
        import pip  # noqa: F401, PLC0415 - 仅探测可用性
    except Exception:  # noqa: BLE001
        return False, "当前 Python 环境没有 pip，无法在应用内安装 GPU 运行时"
    return True, None


def restart_supported() -> tuple[bool, str | None]:
    if getattr(sys, "frozen", False):
        return True, None
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    name = getattr(spec, "name", "") or ""
    if name.startswith("app.desktop"):
        return True, None
    return False, "当前以开发服务器方式运行，请手动重启（Ctrl+C 后重跑 start 脚本）"


def windows_video_adapters() -> list[tuple[str, str]]:
    """Windows 下枚举显卡（vendor, name）。

    ``hardware.py`` 只认得 NVIDIA（pynvml / nvidia-smi）与 Apple Silicon，
    AMD / Intel 机器需要这里补一次 WMI 查询，否则 DirectML 选项永远不会出现。
    """
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            creationflags=_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("WMI 显卡探测失败", exc_info=True)
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    rows = raw if isinstance(raw, list) else [raw]
    adapters: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        maker = str(row.get("AdapterCompatibility") or "").lower()
        if not name:
            continue
        if "nvidia" in maker:
            vendor = "nvidia"
        elif "advanced micro devices" in maker or maker.strip() == "amd":
            vendor = "amd"
        elif "intel" in maker:
            vendor = "intel"
        else:
            vendor = "unknown"
        adapters.append((vendor, name))
    return adapters


def build_options(
    system: SystemInfo,
    *,
    frozen: bool = False,
    adapters: Sequence[tuple[str, str]] = (),
) -> list[AcceleratorOption]:
    """按 GPU 厂商给出可用的加速后端；不适用的一律不出现在列表里。"""
    gpu = _primary_gpu(system)
    options: list[AcceleratorOption] = []

    is_windows = system.platform == "Windows"
    vendors = {item.vendor for item in system.gpus if item.vendor != "unknown"}
    vendors |= {vendor for vendor, _ in adapters if vendor != "unknown"}

    if gpu is not None and gpu.vendor == "nvidia":
        label, description, size = VARIANT_META["cuda"]
        cuda_version = gpu.cuda_version
        blocked: str | None = None
        if frozen:
            blocked = "打包版本未内置 pip，无法在应用内安装"
        elif cuda_version and _version_tuple(cuda_version) < MIN_CUDA_VERSION:
            blocked = (
                f"驱动支持的 CUDA 版本为 {cuda_version}，"
                f"需要 {'.'.join(str(part) for part in MIN_CUDA_VERSION)}+，请先升级 NVIDIA 驱动"
            )
        options.append(
            AcceleratorOption(
                variant="cuda",
                label=label,
                description=description,
                packages=list(VARIANT_PACKAGES["cuda"]),
                estimatedBytes=size,
                recommended=blocked is None,
                blockedReason=blocked,
            )
        )

    if is_windows and vendors & {"nvidia", "amd", "intel"}:
        label, description, size = VARIANT_META["directml"]
        blocked = "打包版本未内置 pip，无法在应用内安装" if frozen else None
        options.append(
            AcceleratorOption(
                variant="directml",
                label=label,
                description=description,
                packages=list(VARIANT_PACKAGES["directml"]),
                estimatedBytes=size,
                recommended=not any(option.recommended for option in options) and blocked is None,
                blockedReason=blocked,
            )
        )

    # Apple Silicon 的 CoreML EP 由官方 onnxruntime 自带，不需要额外安装
    return options


# --------------------------------------------------------------------------- #
# 目录替换（Windows 文件占用）
# --------------------------------------------------------------------------- #
# 刚写完的 wheel 可能被杀软 / 索引器短暂占用，rename 会报 WinError 5 / 32；
# 重试仍失败时退化为「复制 + 删除」，结果一致但不要求句柄立刻释放。
_PROMOTE_DELAYS = (0.0, 0.5, 1.5, 3.0)


def _retryable_file_error(exc: OSError) -> bool:
    if getattr(exc, "winerror", None) in {5, 32, 183}:
        return True
    return exc.errno in {errno.EACCES, errno.EBUSY, errno.ENOTEMPTY}


def _force_rmtree(path: Path) -> None:
    def _clear_readonly(func, item, _exc_info) -> None:
        os.chmod(item, stat.S_IWRITE)
        func(item)

    shutil.rmtree(path, onerror=_clear_readonly)


def _promote_staging(staging: Path, target: Path) -> None:
    """把 ``staging`` 替换为 ``target``；Windows 句柄占用时重试并回退复制。"""
    if target.exists():
        for index, delay in enumerate(_PROMOTE_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                _force_rmtree(target)
                break
            except OSError:
                if index == len(_PROMOTE_DELAYS) - 1:
                    raise
    for index, delay in enumerate(_PROMOTE_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            staging.rename(target)
            return
        except OSError as exc:
            if not _retryable_file_error(exc):
                raise
            if index == len(_PROMOTE_DELAYS) - 1:
                break
    shutil.copytree(staging, target, dirs_exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 安装管理器
# --------------------------------------------------------------------------- #
class AcceleratorInstallManager:
    """进程内安装任务表；同一时间只跑一个安装任务。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tasks: dict[str, InstallTask] = {}
        self._current: str | None = None
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._cancel = threading.Event()
        self._system: SystemInfo | None = None
        self._system_at = 0.0
        self._adapters: list[tuple[str, str]] = []
        self._adapters_at = 0.0
        self._size_cache: dict[str, tuple[float, int]] = {}
        self._cleanup_staging()

    def _cleanup_staging(self) -> None:
        """清掉上次中断安装留下的 ``.{variant}.tmp``，避免白占磁盘。"""
        root = runtime_root(self.settings)
        if not root.is_dir():
            return
        for item in root.glob(".*.tmp"):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                logger.info("已清理上次安装残留：%s", item)

    # ---- 探测 ----
    def system(self, max_age: float = 20.0) -> SystemInfo:
        now = time.monotonic()
        if self._system is None or now - self._system_at > max_age:
            self._system = collect_system_info()
            self._system_at = now
        return self._system

    def adapters(self, max_age: float = 300.0) -> list[tuple[str, str]]:
        """Windows 下补一次 WMI 显卡枚举（AMD / Intel 机器只能靠它识别）。"""
        now = time.monotonic()
        if not self._adapters and now - self._adapters_at > max_age:
            self._adapters = windows_video_adapters()
            self._adapters_at = now
        return self._adapters

    def invalidate(self) -> None:
        self._system = None
        self._system_at = 0.0
        self._adapters = []
        self._adapters_at = 0.0
        self._size_cache.clear()

    def _dir_size(self, path: Path) -> int:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return 0
        cached = self._size_cache.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        self._size_cache[str(path)] = (stamp, total)
        return total

    def status(self) -> AcceleratorPayload:
        system = self.system()
        frozen = bool(getattr(sys, "frozen", False))
        can_install, blocked = install_supported()
        restart_ok, restart_reason = restart_supported()
        active = active_variant(self.settings)

        installed: list[InstalledRuntime] = []
        for variant in VARIANT_PACKAGES:
            path = variant_dir(variant, self.settings)
            if not _looks_installed(path):
                continue
            installed.append(
                InstalledRuntime(
                    variant=variant,  # type: ignore[arg-type]
                    path=str(path),
                    sizeBytes=self._dir_size(path),
                    active=variant == active,
                )
            )

        options = build_options(system, frozen=frozen, adapters=self.adapters())
        for option in options:
            option.installed = any(item.variant == option.variant for item in installed)
            if option.blockedReason is None and not can_install:
                option.blockedReason = blocked

        providers = list(system.onnx_providers)
        gpu_accelerated = any(name.lower() in GPU_PROVIDERS for name in providers)

        with self._lock:
            task = self._tasks.get(self._current) if self._current else None
            task_copy = task.model_copy() if task else None

        return AcceleratorPayload(
            options=options,
            installed=installed,
            active=active,
            currentProviders=providers,
            gpuAccelerated=gpu_accelerated,
            restartRequired=self.restart_required(active),
            canInstall=can_install,
            blockedReason=blocked,
            restartSupported=restart_ok,
            restartBlockedReason=restart_reason,
            frozen=frozen,
            runtimeDir=str(runtime_root(self.settings)),
            task=task_copy,
        )

    def restart_required(self, active: Variant | None = None) -> bool:
        """配置的运行时与本进程实际加载的不一致 → 需要重启才生效。"""
        active = active if active is not None else active_variant(self.settings)
        loaded = _loaded_runtime_path()
        if loaded is None:
            return False  # 还没 import，下次 import 自然会走新路径
        root = runtime_root(self.settings)
        try:
            inside = root.resolve() in loaded.parents
        except OSError:
            inside = False
        if active is None:
            return inside
        return not _loaded_from(variant_dir(active, self.settings))

    # ---- 安装 ----
    def start(self, variant: str) -> InstallTask:
        if variant not in VARIANT_PACKAGES:
            raise ValidationFailedError(
                "未知的加速运行时", details={"variant": variant, "allowed": sorted(VARIANT_PACKAGES)}
            )
        can_install, reason = install_supported()
        if not can_install:
            raise ServiceUnavailableError(reason or "当前环境不支持安装 GPU 运行时")

        option = next((item for item in build_options(self.system()) if item.variant == variant), None)
        if option is None:
            raise ValidationFailedError(
                "当前硬件不适用该加速后端", details={"variant": variant}
            )
        if option.blockedReason:
            raise ValidationFailedError(option.blockedReason, details={"variant": variant})

        target = variant_dir(variant, self.settings)
        if _loaded_from(target):
            raise ConflictError(
                "该运行时正在被当前进程使用，需先切回 CPU 推理并重启后再安装",
                details={"variant": variant},
            )

        with self._lock:
            current = self._tasks.get(self._current) if self._current else None
            if current is not None and current.state in {"queued", "running"}:
                if current.variant == variant:
                    return current.model_copy()
                raise ConflictError(
                    "已有安装任务在进行中，请等待完成或取消",
                    details={"running": current.variant},
                )
            task = InstallTask(id=uuid.uuid4().hex[:10], variant=variant)  # type: ignore[arg-type]
            self._tasks[task.id] = task
            self._current = task.id
            self._cancel.clear()
            self._prune()

        thread = threading.Thread(
            target=self._run, args=(task, list(VARIANT_PACKAGES[variant])), name=f"install-{variant}", daemon=True
        )
        thread.start()
        return task.model_copy()

    def cancel(self) -> InstallTask | None:
        with self._lock:
            task = self._tasks.get(self._current) if self._current else None
            if task is None or task.state not in {"queued", "running"}:
                return task.model_copy() if task else None
            self._cancel.set()
            proc = self._proc
            task.message = "正在取消安装"
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                logger.debug("终止 pip 进程失败", exc_info=True)
        return task.model_copy()

    def _publish(self, task: InstallTask, **fields: object) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(task, key, value)

    def _append_log(self, task: InstallTask | None, line: str) -> None:
        if task is None:
            return
        text = line.rstrip()
        if not text:
            return
        with self._lock:
            tail = list(task.logTail)
            tail.append(text[:400])
            task.logTail = tail[-_LOG_TAIL_LINES:]

    def _run(self, task: InstallTask, packages: list[str]) -> None:
        root = runtime_root(self.settings)
        target = variant_dir(task.variant, self.settings)
        staging = root / f".{task.variant}.tmp"
        try:
            self._publish(task, state="running", phase="preparing", percent=2.0, message="准备安装目录")
            root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
            if staging.exists():
                raise RuntimeError(f"无法清理上次残留的临时目录：{staging}")
            staging.mkdir(parents=True, exist_ok=True)

            self._publish(
                task,
                phase="downloading",
                percent=5.0,
                message="正在下载 " + "、".join(packages),
            )
            collected = 0
            expected = max(1, len(packages))
            current_download: dict[str, str] = {}
            attempt_state: dict[str, int] = {"n": 1, "total": _PIP_MAX_ATTEMPTS}
            stop_heartbeat = threading.Event()
            heartbeat_started = time.monotonic()

            def heartbeat() -> None:
                while not stop_heartbeat.wait(_HEARTBEAT_SECONDS):
                    name = current_download.get("name")
                    if not name:
                        continue
                    elapsed = int(time.monotonic() - heartbeat_started)
                    size = current_download.get("size", "")
                    n, total = attempt_state["n"], attempt_state["total"]
                    self._publish(
                        task,
                        message=f"第 {n}/{total} 次 · 正在下载（{collected}/{expected}）{name}"
                        f"（{size}，已用 {elapsed} 秒）",
                    )

            heartbeat_thread = threading.Thread(target=heartbeat, name="pip-progress", daemon=True)
            heartbeat_thread.start()

            def on_line(line: str) -> None:
                nonlocal collected
                self._append_log(task, line)
                stripped = line.strip()
                attempt = _ATTEMPT_RE.match(stripped)
                if attempt:
                    attempt_state["n"] = int(attempt.group(1))
                    attempt_state["total"] = int(attempt.group(2))
                    current_download.clear()
                    collected = 0
                    self._publish(
                        task,
                        phase="downloading",
                        percent=5.0,
                        message=f"第 {attempt_state['n']}/{attempt_state['total']} 次尝试，"
                        f"单次超时 {int(_PIP_ATTEMPT_TIMEOUT_SECONDS // 60)} 分钟",
                    )
                    return
                if "次尝试失败" in stripped:
                    self._publish(task, message=stripped)
                    return
                match = _COLLECT_RE.match(stripped)
                if match:
                    current_download.clear()
                    collected = min(expected, collected + 1)
                    percent = 5.0 + 65.0 * collected / expected
                    self._publish(
                        task,
                        phase="downloading",
                        percent=round(percent, 1),
                        message=f"第 {attempt_state['n']}/{attempt_state['total']} 次 · "
                        f"正在下载（{collected}/{expected}）{match.group(1)}",
                    )
                    return
                download = _DOWNLOAD_RE.match(stripped)
                if download:
                    current_download["name"] = download.group(1)
                    current_download["size"] = download.group(2)
                    self._publish(
                        task,
                        phase="downloading",
                        message=f"第 {attempt_state['n']}/{attempt_state['total']} 次 · "
                        f"正在下载（{collected}/{expected}）{download.group(1)}"
                        f"（{download.group(2)}）",
                    )
                    return
                if "installing collected packages" in stripped.lower():
                    current_download.clear()
                    self._publish(task, phase="installing", percent=75.0, message="正在解压并写入文件")

            def on_proc(proc: subprocess.Popen[str] | None) -> None:
                self._proc = proc

            try:
                code = _pip_install(staging, packages, on_line, self._cancel, on_proc)
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join(timeout=2)

            if self._cancel.is_set():
                shutil.rmtree(staging, ignore_errors=True)
                self._publish(
                    task,
                    state="cancelled",
                    phase="done",
                    message="已取消安装",
                    finishedAt=time.time(),
                )
                return
            if code != 0:
                raise RuntimeError(f"pip 退出码 {code}，详见日志")
            if not _looks_installed(staging):
                raise RuntimeError("安装完成但未找到 onnxruntime 包，pip 输出见日志")

            self._publish(task, phase="verifying", percent=88.0, message="校验执行提供器")
            providers = self._verify(task.variant, staging)
            wanted = _expected_provider(task.variant)
            if providers and wanted not in providers:
                raise RuntimeError(
                    f"安装完成但未检测到 {wanted}，已放弃启用（详见日志）"
                )

            try:
                _promote_staging(staging, target)
            except OSError as exc:
                raise RuntimeError(
                    f"无法把安装结果写入 {target.name}：{exc}。"
                    "请关闭占用该目录的程序（或重启应用）后重试"
                ) from exc

            _write_active(
                self.settings,
                {
                    "variant": task.variant,
                    "python": sys.executable,
                    "providers": providers,
                    "installedAt": time.time(),
                },
            )
            self.invalidate()
            self._invalidate_hardware_cache()
            self._publish(
                task,
                state="done",
                phase="done",
                percent=100.0,
                message="安装完成，重启应用后生效",
                restartRequired=True,
                finishedAt=time.time(),
            )
            logger.info("GPU 运行时安装完成：%s（providers=%s）", task.variant, providers)
        except Exception as exc:  # noqa: BLE001 - 后台线程把错误交给前端展示
            logger.warning("GPU 运行时安装失败：%s", exc)
            shutil.rmtree(staging, ignore_errors=True)
            timed_out_error = isinstance(exc, RuntimeError) and str(exc).startswith("安装超时")
            self._publish(
                task,
                state="error",
                phase="done",
                message="安装超时" if timed_out_error else "安装失败",
                error=str(exc),
                finishedAt=time.time(),
            )

    def _verify(self, variant: str, path: Path) -> list[str]:
        """子进程里 import 新装的 onnxruntime，确认 EP 真的被编译进去。"""
        script = (
            "import json, sys;"
            f"sys.path.insert(0, {str(path)!r});"
            "import onnxruntime as ort;"
            "print('PROBE:' + json.dumps({'version': ort.__version__,"
            " 'providers': ort.get_available_providers()}))"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        directories = _dll_directories(path)
        if directories:
            prefix = os.pathsep.join(str(item) for item in directories)
            env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                env=env,
                creationflags=_creationflags(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._append_log(self._task_by_variant(variant), f"校验进程失败：{exc}")
            return []
        output = f"{completed.stdout}\n{completed.stderr}"
        for line in output.splitlines():
            if line.startswith("PROBE:"):
                try:
                    payload = json.loads(line[len("PROBE:") :])
                except json.JSONDecodeError:
                    break
                task = self._task_by_variant(variant)
                if task is not None:
                    self._append_log(
                        task, f"onnxruntime {payload.get('version')} → {', '.join(payload.get('providers', []))}"
                    )
                return list(payload.get("providers") or [])
        task = self._task_by_variant(variant)
        if task is not None:
            for line in output.strip().splitlines()[-_LOG_TAIL_LINES:]:
                self._append_log(task, line)
        return []

    def _task_by_variant(self, variant: str) -> InstallTask | None:
        with self._lock:
            task = self._tasks.get(self._current) if self._current else None
            if task is not None and task.variant == variant:
                return task
        return None

    @staticmethod
    def _invalidate_hardware_cache() -> None:
        try:
            from app.backend.api.hardware import reset_cache  # noqa: PLC0415 - 避免循环导入
        except Exception:  # noqa: BLE001
            return
        reset_cache()

    # ---- 切换 / 回滚 ----
    def activate_variant(self, variant: str) -> None:
        """把已安装的另一个运行时设为下次启动生效项。"""
        if variant not in VARIANT_PACKAGES:
            raise ValidationFailedError("未知的加速运行时", details={"variant": variant})
        path = variant_dir(variant, self.settings)
        if not _looks_installed(path):
            raise NotFoundError("该运行时没有安装", details={"variant": variant})
        _write_active(
            self.settings,
            {
                "variant": variant,
                "python": sys.executable,
                "installedAt": time.time(),
            },
        )
        self.invalidate()
        self._invalidate_hardware_cache()
        logger.info("已切换加速运行时为 %s（重启后生效）", variant)

    def remove(self, variant: str) -> None:
        if variant not in VARIANT_PACKAGES:
            raise ValidationFailedError("未知的加速运行时", details={"variant": variant})
        path = variant_dir(variant, self.settings)
        if not path.exists():
            raise NotFoundError("该运行时没有安装", details={"variant": variant})
        if _loaded_from(path):
            raise ConflictError(
                "该运行时正在被当前进程使用，需重启后再移除",
                details={"variant": variant},
            )
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            raise ConflictError("删除运行时目录失败，可能仍有文件被占用", details={"variant": variant})
        if active_variant(self.settings) == variant:
            _clear_active(self.settings)
        self.invalidate()
        self._invalidate_hardware_cache()
        logger.info("已移除 GPU 运行时：%s", variant)

    def _prune(self) -> None:
        finished = [task for task in self._tasks.values() if task.state in {"done", "error", "cancelled"}]
        finished.sort(key=lambda item: item.finishedAt or 0.0, reverse=True)
        for task in finished[4:]:
            self._tasks.pop(task.id, None)


# --------------------------------------------------------------------------- #
# 进程重启
# --------------------------------------------------------------------------- #
def _pip_command(target: Path, packages: Iterable[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-input",
        "--disable-pip-version-check",
        "--progress-bar",
        "off",
        "--timeout",
        "60",
        "--retries",
        "10",
        "--no-warn-script-location",
        "--target",
        str(target),
        *packages,
    ]


def _pip_install(
    target: Path,
    packages: Sequence[str],
    on_line: Callable[[str], None],
    cancel: threading.Event,
    on_proc: Callable[[subprocess.Popen[str] | None], None] | None = None,
    *,
    attempts: int = _PIP_MAX_ATTEMPTS,
    timeout: float = _PIP_ATTEMPT_TIMEOUT_SECONDS,
) -> int:
    """运行 pip；失败或单次超时自动重试，返回最后一次退出码。

    大包下载期间 pip 不输出任何行，所以每轮都由独立看门狗计时；已完整下载的
    wheel 会留在 pip 缓存中，重试不会重复下载所有包。
    """
    command = _pip_command(target, packages)
    total = max(1, attempts)
    last_reason = "未知错误"

    for attempt in range(1, total + 1):
        if cancel.is_set():
            return -1
        if attempt > 1:
            delay = _PIP_RETRY_DELAYS[min(attempt - 2, len(_PIP_RETRY_DELAYS) - 1)]
            on_line(f"第 {attempt - 1}/{total} 次尝试失败：{last_reason}，{delay:.0f} 秒后重试")
            if cancel.wait(delay):
                return -1
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

        on_line(f"[第 {attempt}/{total} 次尝试] 单次超时 {int(timeout // 60)} 分钟")
        on_line("$ " + " ".join(command))
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=_creationflags(),
        )
        if on_proc is not None:
            on_proc(proc)
        started = time.monotonic()
        timed_out = threading.Event()

        def watchdog(
            proc: subprocess.Popen[str] = proc,
            started: float = started,
            timed_out: threading.Event = timed_out,
            attempt: int = attempt,
            total: int = total,
            timeout: float = timeout,
        ) -> None:
            while not timed_out.wait(2.0):
                if time.monotonic() - started <= timeout:
                    continue
                timed_out.set()
                logger.warning("pip 第 %d/%d 次尝试超过 %d 秒，终止进程", attempt, total, timeout)
                try:
                    proc.terminate()
                except OSError:
                    pass

        monitor = threading.Thread(target=watchdog, name=f"pip-timeout-{attempt}", daemon=True)
        monitor.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                on_line(line)
                if cancel.is_set():
                    proc.terminate()
                    break
            try:
                code = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                code = 1
            if timed_out.is_set():
                last_reason = f"单次尝试超过 {int(timeout // 60)} 分钟未响应"
                continue
            if code == 0:
                return 0
            last_reason = f"pip 退出码 {code}"
        finally:
            timed_out.set()
            if on_proc is not None:
                on_proc(None)
            monitor.join(timeout=3)

    raise RuntimeError(
        f"安装超时：{last_reason}，已尝试 {total} 次仍未完成；请检查网络或更换 pip 镜像"
    )


def _expected_provider(variant: str) -> str:
    return "CUDAExecutionProvider" if variant == "cuda" else "DmlExecutionProvider"


def _restart_command() -> list[str] | None:
    """重建启动命令；无法重建时返回 None（调用方提示用户手动重启）。"""
    if getattr(sys, "frozen", False):
        return [sys.executable, *sys.argv[1:]]
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    name = getattr(spec, "name", "") or ""
    if name.startswith("app.desktop"):
        return [sys.executable, "-m", name, *sys.argv[1:]]
    return None


def schedule_restart(delay: float = 0.6) -> None:
    """先拉起新进程再退出当前进程：onnxruntime 的 DLL 无法在进程内热替换。"""
    command = _restart_command()
    if command is None:
        raise ServiceUnavailableError("当前运行方式不支持自动重启，请手动重启应用")

    def _worker() -> None:
        time.sleep(delay)
        try:
            subprocess.Popen(
                command,
                cwd=str(Path.cwd()),
                close_fds=True,
                creationflags=_DETACHED_FLAGS | _creationflags(),
            )
        except Exception as exc:  # noqa: BLE001 - 起不来就不要退出，避免用户彻底失去程序
            logger.exception("重建进程失败：%s", exc)
            return
        logger.info("已启动新进程 %s，当前进程退出", command)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)

    threading.Thread(target=_worker, name="runtime-restart", daemon=True).start()


__all__ = [
    "AcceleratorInstallManager",
    "AcceleratorOption",
    "AcceleratorPayload",
    "InstallTask",
    "InstalledRuntime",
    "VARIANT_PACKAGES",
    "activate",
    "active_variant",
    "build_options",
    "install_supported",
    "reset_activation_state",
    "restart_supported",
    "runtime_root",
    "schedule_restart",
    "variant_dir",
]
