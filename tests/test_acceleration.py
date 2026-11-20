"""加速决策单测：mock 各种硬件组合，断言选择结果（任务 2.3）。"""

from __future__ import annotations

from collections.abc import Sequence

from app.backend.core.settings import Settings
from app.backend.engine.acceleration import decide_acceleration
from app.backend.engine.hardware import CpuInfo, GpuInfo, MemoryInfo, SystemInfo

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
TENSORRT = "TensorrtExecutionProvider"
COREML = "CoreMLExecutionProvider"
DML = "DmlExecutionProvider"


def make_system(
    *,
    providers: Sequence[str] = (CPU,),
    gpus: Sequence[GpuInfo] = (),
    apple: bool = False,
    physical_cores: int = 8,
    logical_cores: int = 16,
    total_mb: int = 16384,
) -> SystemInfo:
    return SystemInfo(
        platform="Darwin" if apple else "Windows",
        platform_release="14.0" if apple else "10.0.22631",
        architecture="arm64" if apple else "AMD64",
        machine="arm64" if apple else "AMD64",
        is_apple_silicon=apple,
        cpu=CpuInfo(
            model="Apple M2 Pro" if apple else "Test CPU",
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            avx2=None if apple else True,
            features_known=not apple,
        ),
        memory=MemoryInfo(total_mb=total_mb, available_mb=total_mb // 2),
        gpus=list(gpus),
        onnx_providers=list(providers),
        directml_available=DML in providers,
    )


NVIDIA_8G = GpuInfo(vendor="nvidia", name="RTX 4070", memory_mb=8192, api="cuda")
NVIDIA_4G = GpuInfo(vendor="nvidia", name="RTX 3050 Laptop", memory_mb=4096, api="cuda")
APPLE_GPU = GpuInfo(vendor="apple", name="Apple M2 Pro", memory_mb=16384, api="metal", unified_memory=True)


def test_cuda_preferred_over_cpu() -> None:
    plan = decide_acceleration(make_system(providers=[CPU, CUDA], gpus=[NVIDIA_8G]))
    assert plan.preferred_provider == CUDA
    assert plan.gpu_accelerated is True
    assert plan.model_tier == "heavy"
    assert plan.device_name == "RTX 4070"
    assert plan.confidence == "high"


def test_tensorrt_wins_over_cuda_and_preserves_spelling() -> None:
    plan = decide_acceleration(make_system(providers=[CPU, CUDA, TENSORRT], gpus=[NVIDIA_8G]))
    assert plan.preferred_provider == TENSORRT
    assert plan.execution_providers[0] == TENSORRT
    assert CUDA in plan.execution_providers
    assert plan.execution_providers[-1] == CPU


def test_coreml_on_apple_silicon_is_heavy() -> None:
    plan = decide_acceleration(make_system(providers=[CPU, COREML], gpus=[APPLE_GPU], apple=True))
    assert plan.preferred_provider == COREML
    assert plan.gpu_accelerated is True
    assert plan.model_tier == "heavy"
    assert plan.device_name == "Apple M2 Pro"


def test_apple_silicon_small_memory_uses_full() -> None:
    gpu = GpuInfo(vendor="apple", name="Apple M1", memory_mb=8192, api="metal", unified_memory=True)
    plan = decide_acceleration(
        make_system(providers=[CPU, COREML], gpus=[gpu], apple=True, total_mb=8192)
    )
    assert plan.model_tier == "full"


def test_directml_on_windows_without_cuda() -> None:
    plan = decide_acceleration(make_system(providers=[CPU, DML]))
    assert plan.preferred_provider == DML
    assert plan.gpu_accelerated is True
    assert plan.model_tier == "full"


def test_pure_cpu_fallback() -> None:
    plan = decide_acceleration(make_system(providers=[CPU], physical_cores=8))
    assert plan.preferred_provider == CPU
    assert plan.gpu_accelerated is False
    assert plan.model_tier == "lite"
    assert plan.threads == 8
    assert plan.confidence == "medium"


def test_gpu_present_but_cpu_ep_still_uses_lite_tier() -> None:
    """有卡但 onnxruntime 没启用 CUDA EP 时不能按显存推荐 heavy。"""
    plan = decide_acceleration(make_system(providers=[CPU], gpus=[NVIDIA_8G]))
    assert plan.gpu_accelerated is False
    assert plan.model_tier == "lite"
    assert any("onnxruntime" in reason for reason in plan.reasons)


def test_no_onnxruntime_still_returns_cpu_plan() -> None:
    plan = decide_acceleration(make_system(providers=[]))
    assert plan.preferred_provider == CPU
    assert plan.execution_providers == [CPU]
    assert plan.confidence == "low"


def test_unknown_providers_are_ignored() -> None:
    plan = decide_acceleration(
        make_system(providers=["AzureExecutionProvider", "OpenVINOExecutionProvider", CPU])
    )
    assert plan.preferred_provider == CPU
    assert plan.gpu_accelerated is False
    assert "AzureExecutionProvider" not in plan.execution_providers


def test_threads_are_bounded() -> None:
    system = make_system(providers=[CPU], physical_cores=32, logical_cores=64)
    plan = decide_acceleration(system, max_threads=6)
    assert plan.threads == 6


def test_gpu_threads_capped_for_ui_responsiveness() -> None:
    system = make_system(
        providers=[CPU, CUDA], gpus=[NVIDIA_4G], physical_cores=32, logical_cores=64
    )
    plan = decide_acceleration(system, max_threads=16)
    assert plan.threads == 4


def test_single_core_never_zero_threads() -> None:
    plan = decide_acceleration(make_system(providers=[CPU], physical_cores=1, logical_cores=1))
    assert plan.threads == 1


def test_preferred_tier_override() -> None:
    plan = decide_acceleration(make_system(providers=[CPU]), preferred_tier="heavy")
    assert plan.model_tier == "heavy"
    assert "用户指定" in plan.model_tier_reason


def test_settings_drive_threads_and_tier() -> None:
    settings = Settings(max_inference_threads=3, prefer_model_tier="lite")
    plan = decide_acceleration(
        make_system(providers=[CPU, CUDA], gpus=[NVIDIA_8G], physical_cores=8),
        settings=settings,
    )
    assert plan.threads <= 3
    assert plan.model_tier == "lite"


def test_fallback_chain_is_provider_order_without_preferred() -> None:
    plan = decide_acceleration(make_system(providers=[CPU, CUDA, TENSORRT]))
    assert plan.fallback_chain == plan.execution_providers[1:]
    assert plan.fallback_chain[-1] == CPU


def test_vram_threshold_boundaries() -> None:
    full = GpuInfo(vendor="nvidia", name="RTX 3050", memory_mb=4000, api="cuda")
    heavy = GpuInfo(vendor="nvidia", name="RTX 4070", memory_mb=8000, api="cuda")
    assert decide_acceleration(make_system(providers=[CPU, CUDA], gpus=[full])).model_tier == "full"
    assert (
        decide_acceleration(make_system(providers=[CPU, CUDA], gpus=[heavy])).model_tier == "heavy"
    )
