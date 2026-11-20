"""GPU 加速运行时：决策、安装任务与接口（界面内一键安装）。"""

from __future__ import annotations

import errno
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backend.core.settings import Settings
from app.backend.engine import accelerator_install
from app.backend.engine.accelerator_install import (
    AcceleratorInstallManager,
    build_options,
    reset_activation_state,
)
from app.backend.engine.hardware import CpuInfo, GpuInfo, MemoryInfo, SystemInfo
from app.backend.main import create_app

NVIDIA_GPU = GpuInfo(
    vendor="nvidia",
    name="NVIDIA GeForce RTX 5080 Laptop GPU",
    memory_mb=16384,
    driver="580.65",
    cuda_version="12.8",
    api="cuda",
    unified_memory=False,
)
OLD_CUDA_GPU = NVIDIA_GPU.model_copy(update={"cuda_version": "11.8"})
APPLE_GPU = GpuInfo(
    vendor="apple",
    name="Apple M3 Pro",
    memory_mb=18432,
    driver=None,
    cuda_version=None,
    api="metal",
    unified_memory=True,
)


def make_system(*, platform: str = "Windows", gpus=(), providers=()) -> SystemInfo:
    return SystemInfo(
        platform=platform,
        platform_release="10",
        platform_version="10.0.22631",
        architecture="AMD64",
        machine="AMD64",
        glibc=None,
        is_apple_silicon=platform == "Darwin",
        is_wsl=False,
        cpu=CpuInfo(
            model="Test CPU",
            physical_cores=8,
            logical_cores=16,
            avx2=True,
            avx512=False,
            features_known=True,
        ),
        memory=MemoryInfo(total_mb=32768, available_mb=16384),
        gpus=list(gpus),
        onnx_providers=list(providers),
        directml_available=False,
        notes=[],
    )


def variants(options) -> set[str]:
    return {option.variant for option in options}


# --------------------------------------------------------------------------- #
# 选项决策
# --------------------------------------------------------------------------- #
def test_nvidia_gets_cuda_and_directml() -> None:
    options = build_options(make_system(gpus=[NVIDIA_GPU], providers=["CPUExecutionProvider"]))
    assert variants(options) == {"cuda", "directml"}
    cuda = next(option for option in options if option.variant == "cuda")
    assert cuda.recommended is True
    assert cuda.blockedReason is None
    assert any("onnxruntime-gpu" in package for package in cuda.packages)


def test_old_cuda_driver_blocks_cuda() -> None:
    options = build_options(make_system(gpus=[OLD_CUDA_GPU]))
    cuda = next(option for option in options if option.variant == "cuda")
    assert cuda.blockedReason is not None
    assert "11.8" in cuda.blockedReason
    assert cuda.recommended is False
    # DirectML 不依赖 CUDA 版本，仍然可选
    directml = next(option for option in options if option.variant == "directml")
    assert directml.blockedReason is None
    assert directml.recommended is True


def test_amd_gpu_only_offers_directml() -> None:
    options = build_options(make_system(gpus=[]), adapters=[("amd", "Radeon RX 7800 XT")])
    assert variants(options) == {"directml"}


def test_linux_nvidia_has_no_directml() -> None:
    options = build_options(make_system(platform="Linux", gpus=[NVIDIA_GPU]))
    assert variants(options) == {"cuda"}


def test_apple_silicon_needs_no_install() -> None:
    options = build_options(make_system(platform="Darwin", gpus=[APPLE_GPU]))
    assert options == []


def test_frozen_build_marks_options_blocked() -> None:
    options = build_options(make_system(gpus=[NVIDIA_GPU]), frozen=True)
    assert all(option.blockedReason for option in options)


# --------------------------------------------------------------------------- #
# 安装任务
# --------------------------------------------------------------------------- #
@pytest.fixture()
def manager(video_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> AcceleratorInstallManager:
    monkeypatch.setattr(accelerator_install, "install_supported", lambda: (True, None))
    instance = AcceleratorInstallManager(video_settings)
    system = make_system(gpus=[NVIDIA_GPU], providers=["CPUExecutionProvider"])
    monkeypatch.setattr(instance, "system", lambda *args, **kwargs: system)
    monkeypatch.setattr(instance, "adapters", lambda *args, **kwargs: [("nvidia", NVIDIA_GPU.name)])
    return instance


def fake_pip_success(target: Path, packages, on_line, cancel, on_proc=None) -> int:
    on_line("Collecting onnxruntime-gpu\n")
    on_line("Downloading onnxruntime_gpu-1.22.0-cp312-...whl (250.1 MB)\n")
    on_line("Collecting nvidia-cudnn-cu12\n")
    on_line("Installing collected packages: onnxruntime-gpu\n")
    (target / "onnxruntime").mkdir(parents=True, exist_ok=True)
    (target / "onnxruntime" / "__init__.py").write_text("", encoding="utf-8")
    return 0


def wait_for(manager: AcceleratorInstallManager, task_id: str, timeout: float = 5.0):
    deadline = time.perf_counter() + timeout
    current = None
    while time.perf_counter() < deadline:
        current = manager.status().task
        if current is not None and current.id == task_id and current.state in {
            "done",
            "error",
            "cancelled",
        }:
            return current
        time.sleep(0.02)
    return current


def test_install_flow_activates_variant(
    manager: AcceleratorInstallManager,
    video_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator_install, "_pip_install", fake_pip_success)
    monkeypatch.setattr(
        manager, "_verify", lambda variant, path: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    task = manager.start("cuda")
    finished = wait_for(manager, task.id)
    assert finished is not None
    assert finished.state == "done", finished.error
    assert finished.percent == 100.0
    assert finished.restartRequired is True

    target = video_settings.data_dir / "runtime" / "cuda"
    assert (target / "onnxruntime" / "__init__.py").is_file()

    active_file = video_settings.data_dir / "runtime" / "active.json"
    payload = json.loads(active_file.read_text(encoding="utf-8"))
    assert payload["variant"] == "cuda"

    status = manager.status()
    assert status.active == "cuda"
    assert [item.variant for item in status.installed] == ["cuda"]
    assert status.installed[0].active is True


def test_install_failure_keeps_previous_state(
    manager: AcceleratorInstallManager,
    video_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_pip(target: Path, packages, on_line, cancel, on_proc=None) -> int:
        on_line("ERROR: Could not find a version that satisfies the requirement\n")
        return 1

    monkeypatch.setattr(accelerator_install, "_pip_install", failing_pip)
    task = manager.start("cuda")
    finished = wait_for(manager, task.id)
    assert finished is not None
    assert finished.state == "error"
    assert finished.error is not None
    assert not (video_settings.data_dir / "runtime" / "cuda").exists()
    assert not (video_settings.data_dir / "runtime" / ".cuda.tmp").exists()
    assert manager.status().active is None


class _NoCancel:
    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float | None = None) -> bool:
        return False


class _FakePipProcess:
    def __init__(self, code: int) -> None:
        self.code = code
        self.stdout = iter([f"Collecting demo-{code}\n"])

    def wait(self, timeout: float | None = None) -> int:
        return self.code

    def terminate(self) -> None:
        pass


def _make_staging(tmp_path: Path) -> Path:
    staging = tmp_path / ".cuda.tmp"
    (staging / "onnxruntime").mkdir(parents=True)
    (staging / "onnxruntime" / "__init__.py").write_text("", encoding="utf-8")
    return staging


def test_promote_staging_retries_access_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = _make_staging(tmp_path)
    target = tmp_path / "cuda"
    original_rename = Path.rename
    calls: list[str] = []

    def flaky_rename(self: Path, dst) -> None:
        calls.append(str(self))
        if len(calls) == 1:
            raise OSError(errno.EACCES, "拒绝访问")
        return original_rename(self, dst)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr(accelerator_install.time, "sleep", lambda _: None)

    accelerator_install._promote_staging(staging, target)

    assert (target / "onnxruntime" / "__init__.py").is_file()
    assert not staging.exists()
    assert len(calls) == 2


def test_promote_staging_falls_back_to_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = _make_staging(tmp_path)
    target = tmp_path / "cuda"

    def denied_rename(self: Path, dst) -> None:
        raise OSError(errno.EACCES, "拒绝访问")

    monkeypatch.setattr(Path, "rename", denied_rename)
    monkeypatch.setattr(accelerator_install.time, "sleep", lambda _: None)

    accelerator_install._promote_staging(staging, target)

    assert (target / "onnxruntime" / "__init__.py").is_file()
    assert not staging.exists()


def test_pip_install_retries_until_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codes = iter([1, 1, 0])
    monkeypatch.setattr(
        accelerator_install.subprocess,
        "Popen",
        lambda command, **kwargs: _FakePipProcess(next(codes)),
    )
    monkeypatch.setattr(accelerator_install, "_PIP_RETRY_DELAYS", (0.0, 0.0))
    target = tmp_path / "staging"
    target.mkdir()
    lines: list[str] = []

    code = accelerator_install._pip_install(target, ["demo"], lines.append, _NoCancel())  # type: ignore[arg-type]

    assert code == 0
    assert sum(1 for line in lines if line.startswith("[第 ")) == 3
    assert any("第 1/3 次尝试失败" in line for line in lines)


def test_pip_install_reports_timeout_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codes = iter([1, 1, 1])
    monkeypatch.setattr(
        accelerator_install.subprocess,
        "Popen",
        lambda command, **kwargs: _FakePipProcess(next(codes)),
    )
    monkeypatch.setattr(accelerator_install, "_PIP_RETRY_DELAYS", (0.0, 0.0))
    target = tmp_path / "staging"
    target.mkdir()
    lines: list[str] = []

    with pytest.raises(RuntimeError, match="安装超时"):
        accelerator_install._pip_install(target, ["demo"], lines.append, _NoCancel())  # type: ignore[arg-type]

    assert any("第 2/3 次尝试失败" in line for line in lines)


def test_timeout_task_shows_timeout_message(
    manager: AcceleratorInstallManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout_pip(target: Path, packages, on_line, cancel, on_proc=None) -> int:
        on_line("[第 1/3 次尝试] 单次超时 10 分钟")
        raise RuntimeError("安装超时：单次尝试超过 10 分钟未响应，已尝试 3 次仍未完成")

    monkeypatch.setattr(accelerator_install, "_pip_install", timeout_pip)
    task = manager.start("cuda")
    finished = wait_for(manager, task.id)
    assert finished is not None
    assert finished.state == "error"
    assert finished.message == "安装超时"


def test_duplicate_install_returns_same_task(
    manager: AcceleratorInstallManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow_pip(target: Path, packages, on_line, cancel, on_proc=None) -> int:
        release.wait(timeout=3)
        (target / "onnxruntime").mkdir(parents=True, exist_ok=True)
        (target / "onnxruntime" / "__init__.py").write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(accelerator_install, "_pip_install", slow_pip)
    monkeypatch.setattr(manager, "_verify", lambda variant, path: ["CUDAExecutionProvider"])
    first = manager.start("cuda")
    second = manager.start("cuda")
    assert first.id == second.id
    release.set()
    wait_for(manager, first.id)


def test_remove_rolls_back(manager: AcceleratorInstallManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator_install, "_pip_install", fake_pip_success)
    monkeypatch.setattr(manager, "_verify", lambda variant, path: ["CUDAExecutionProvider"])
    task = manager.start("cuda")
    wait_for(manager, task.id)
    assert manager.status().active == "cuda"

    manager.remove("cuda")
    status = manager.status()
    assert status.active is None
    assert status.installed == []


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #
def test_runtime_api(video_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_activation_state()
    monkeypatch.setattr(accelerator_install, "install_supported", lambda: (True, None))
    app = create_app(video_settings)
    with TestClient(app) as client:
        listed = client.get("/api/runtime/accelerators")
        assert listed.status_code == 200
        body = listed.json()
        assert Path(body["runtimeDir"]) == video_settings.data_dir / "runtime"
        assert isinstance(body["options"], list)
        assert body["task"] is None

        invalid = client.post("/api/runtime/accelerators/install", json={"variant": "rocm"})
        assert invalid.status_code == 422

        missing = client.delete("/api/runtime/accelerators/cuda")
        assert missing.status_code == 404

        not_installed = client.post("/api/runtime/accelerators/directml/activate")
        assert not_installed.status_code == 404
