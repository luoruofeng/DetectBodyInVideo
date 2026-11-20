"""系统信息与加速器探测（任务 2.1 / 2.2）。

设计原则：任何探测失败都不能抛异常打断主流程，只降级并写入 ``notes``。
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger

logger = get_logger("hardware")

MB = 1024 * 1024
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
class CpuInfo(BaseModel):
    model: str = "unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    avx2: bool | None = None
    avx512: bool | None = None
    features_known: bool = False


class GpuInfo(BaseModel):
    vendor: str  # nvidia / apple / amd / intel / unknown
    name: str
    memory_mb: int | None = None
    driver: str | None = None
    cuda_version: str | None = None
    api: str = "unknown"  # cuda / metal / directml / vulkan / unknown
    unified_memory: bool = False


class MemoryInfo(BaseModel):
    total_mb: int | None = None
    available_mb: int | None = None


class SystemInfo(BaseModel):
    platform: str
    platform_release: str = ""
    platform_version: str = ""
    architecture: str = ""
    machine: str = ""
    glibc: str | None = None
    is_apple_silicon: bool = False
    is_wsl: bool = False
    cpu: CpuInfo = Field(default_factory=CpuInfo)
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    gpus: list[GpuInfo] = Field(default_factory=list)
    onnx_providers: list[str] = Field(default_factory=list)
    directml_available: bool = False
    notes: list[str] = Field(default_factory=list)

    def max_gpu_memory_mb(self) -> int:
        return max((gpu.memory_mb or 0) for gpu in self.gpus) if self.gpus else 0


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _run_command(cmd: list[str], timeout: float = 5.0) -> str | None:
    """执行外部命令取 stdout；失败/超时一律返回 None。"""
    try:
        completed = subprocess.run(  # noqa: S603 - 固定命令列表，无 shell
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Darwin":
            return _run_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
    except Exception:  # noqa: BLE001 - 探测失败不影响主流程
        logger.debug("CPU 型号探测失败", exc_info=True)
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores() -> int | None:
    try:
        import psutil

        value = psutil.cpu_count(logical=False)
        if value:
            return int(value)
    except Exception:  # noqa: BLE001
        pass

    system = platform.system()
    try:
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            pairs: set[tuple[str, str]] = set()
            physical_id = core_id = None
            for line in text.splitlines():
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if physical_id is not None and core_id is not None:
                        pairs.add((physical_id, core_id))
            if pairs:
                return len(pairs)
        if system == "Darwin":
            out = _run_command(["sysctl", "-n", "hw.physicalcpu"])
            if out and out.isdigit():
                return int(out)
    except Exception:  # noqa: BLE001
        logger.debug("物理核数探测失败", exc_info=True)
    return None


def _numpy_cpu_features() -> set[str]:
    """Windows / 兜底路径：借 numpy 的 CPU 分发表读取指令集。"""
    for module_name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        table = getattr(module, "__cpu_features__", None)
        if isinstance(table, dict):
            return {str(key).lower() for key, enabled in table.items() if enabled}
    return set()


def _cpu_features() -> set[str]:
    system = platform.system()
    features: set[str] = set()
    try:
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.startswith("flags"):
                    features.update(part.lower() for part in line.split(":", 1)[1].split())
                    break
        elif system == "Darwin":
            for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
                out = _run_command(["sysctl", "-n", key])
                if out:
                    features.update(part.lower() for part in out.split())
    except Exception:  # noqa: BLE001
        logger.debug("CPU 指令集探测失败", exc_info=True)
    if not features:
        features = _numpy_cpu_features()
    return features


def _memory_info() -> MemoryInfo:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return MemoryInfo(total_mb=int(vm.total // MB), available_mb=int(vm.available // MB))
    except Exception:  # noqa: BLE001
        pass

    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return MemoryInfo(
                    total_mb=int(status.ullTotalPhys // MB),
                    available_mb=int(status.ullAvailPhys // MB),
                )
        except Exception:  # noqa: BLE001
            logger.debug("Windows 内存探测失败", exc_info=True)
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            return MemoryInfo(total_mb=int(page_size * total_pages // MB))
        except (ValueError, OSError, AttributeError):
            logger.debug("POSIX 内存探测失败", exc_info=True)

    return MemoryInfo()


# --------------------------------------------------------------------------- #
# GPU / 执行提供器
# --------------------------------------------------------------------------- #
def _cuda_version_from_smi() -> str | None:
    out = _run_command(["nvidia-smi"])
    if not out:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return match.group(1) if match else None


def _nvidia_smi_gpus() -> list[GpuInfo]:
    out = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    cuda = _cuda_version_from_smi()
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            memory_mb = int(float(parts[1]))
        except (TypeError, ValueError):
            memory_mb = None
        gpus.append(
            GpuInfo(
                vendor="nvidia",
                name=parts[0],
                memory_mb=memory_mb,
                driver=parts[2] if len(parts) > 2 else None,
                cuda_version=cuda,
                api="cuda",
            )
        )
    return gpus


def _nvidia_gpus() -> list[GpuInfo]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            driver = _decode(pynvml.nvmlSystemGetDriverVersion())
            cuda = _cuda_version_from_smi()
            gpus: list[GpuInfo] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append(
                    GpuInfo(
                        vendor="nvidia",
                        name=_decode(pynvml.nvmlDeviceGetName(handle)),
                        memory_mb=int(memory.total // MB),
                        driver=driver,
                        cuda_version=cuda,
                        api="cuda",
                    )
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - 无 pynvml / 无驱动时走 CLI
        logger.debug("pynvml 不可用，改用 nvidia-smi", exc_info=True)
        return _nvidia_smi_gpus()


def _apple_gpu(total_memory_mb: int | None) -> GpuInfo:
    chip = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    return GpuInfo(
        vendor="apple",
        name=chip,
        memory_mb=total_memory_mb,
        api="metal",
        unified_memory=True,
    )


def available_providers() -> list[str]:
    """onnxruntime 可用的执行提供器；未安装则返回空列表。"""
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:  # noqa: BLE001
        logger.debug("onnxruntime 不可用", exc_info=True)
        return []


def collect_system_info() -> SystemInfo:
    """采集完整系统信息（含 GPU 与 EP 枚举）。"""
    system = platform.system()
    machine = platform.machine()
    total_mem = _memory_info()
    is_apple_silicon = system == "Darwin" and machine.lower() in {"arm64", "aarch64"}
    is_wsl = "microsoft" in platform.release().lower()

    features = _cpu_features()
    physical = _physical_cores()
    logical = os.cpu_count() or 1
    cpu = CpuInfo(
        model=_cpu_model(),
        physical_cores=physical or logical,
        logical_cores=logical,
        avx2=("avx2" in features) if features else None,
        avx512=("avx512f" in features) if features else None,
        features_known=bool(features),
    )

    glibc: str | None = None
    if system == "Linux":
        try:
            lib, version = platform.libc_ver()
            glibc = f"{lib} {version}".strip() or None
        except Exception:  # noqa: BLE001
            glibc = None

    providers = available_providers()
    provider_names = {name.lower() for name in providers}

    gpus: list[GpuInfo] = []
    if system == "Windows" or system == "Linux":
        gpus.extend(_nvidia_gpus())
    if is_apple_silicon:
        gpus.append(_apple_gpu(total_mem.total_mb))

    directml_available = "dmlexecutionprovider" in provider_names

    notes: list[str] = []
    if not gpus and not directml_available:
        notes.append("未检测到可加速的 GPU，推理将回退到 CPU")
    if is_wsl:
        notes.append("检测到 WSL 环境：CUDA 依赖 Windows 侧驱动与 WSL GPU 直通")
    if cpu.avx2 is False:
        notes.append("CPU 不支持 AVX2，建议使用 lite 档位模型")
    if physical is None and system == "Windows":
        notes.append("无法确定物理核数，已按逻辑核数近似")
    if not providers:
        notes.append("onnxruntime 未安装或不可用，无法枚举加速后端")

    return SystemInfo(
        platform=system,
        platform_release=platform.release(),
        platform_version=platform.version(),
        architecture=platform.machine(),
        machine=machine,
        glibc=glibc,
        is_apple_silicon=is_apple_silicon,
        is_wsl=is_wsl,
        cpu=cpu,
        memory=total_mem,
        gpus=gpus,
        onnx_providers=providers,
        directml_available=directml_available,
        notes=notes,
    )
