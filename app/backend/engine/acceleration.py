"""加速能力决策（任务 2.3）。

优先级（与 TODO 一致）：``TensorRT > CUDA > CoreML > DirectML > CPU(多线程)``。
决策是纯函数，方便用 mock 硬件组合做单元测试。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.settings import Settings, get_settings
from app.backend.engine.hardware import SystemInfo, collect_system_info

ModelTier = Literal["lite", "full", "heavy"]

# onnxruntime 提供器（小写） → (族名, 展示名, 说明)
PROVIDER_META: dict[str, tuple[str, str, str]] = {
    "tensorrtexecutionprovider": ("tensorrt", "TensorRT", "TensorRT 吞吐最高，首次会构建引擎缓存"),
    "cudaexecutionprovider": ("cuda", "CUDA", "NVIDIA CUDA 加速"),
    "coremlexecutionprovider": ("coreml", "CoreML", "Apple 神经引擎 / GPU 加速"),
    "dmlexecutionprovider": ("directml", "DirectML", "Windows DirectML（DX12 GPU）"),
    "cpuexecutionprovider": ("cpu", "CPU", "CPU 多线程兜底"),
}

# 决策优先级顺序（不含 CPU，CPU 始终兜底）
PRIORITY: tuple[str, ...] = (
    "tensorrtexecutionprovider",
    "cudaexecutionprovider",
    "coremlexecutionprovider",
    "dmlexecutionprovider",
)

GPU_FAMILIES = {"tensorrt", "cuda", "coreml", "directml"}

# 模型档位阈值
TIER_HEAVY_VRAM_MB = 8000
TIER_FULL_VRAM_MB = 4000
APPLE_HEAVY_MEMORY_MB = 16 * 1024


class AccelerationPlan(BaseModel):
    preferred_provider: str
    device_name: str | None = None
    gpu_accelerated: bool = False
    execution_providers: list[str] = Field(default_factory=list)
    fallback_chain: list[str] = Field(default_factory=list)
    threads: int = 1
    model_tier: ModelTier = "lite"
    model_tier_reason: str = ""
    confidence: Literal["high", "medium", "low"] = "high"
    reasons: list[str] = Field(default_factory=list)


class HardwareReport(BaseModel):
    generated_at: datetime
    system: SystemInfo
    acceleration: AccelerationPlan


def _find_provider(available: list[str], wanted: str) -> str | None:
    """大小写不敏感匹配（onnxruntime 拼写为 ``TensorrtExecutionProvider``）。"""
    for name in available:
        if name.lower() == wanted:
            return name
    return None


def _threads_for(gpu: bool, system: SystemInfo, max_threads: int) -> int:
    physical = max(1, system.cpu.physical_cores or 1)
    logical = max(1, system.cpu.logical_cores or physical)
    cores = min(physical, logical)
    limit = max(1, max_threads)
    if gpu:
        # GPU 上 CPU 只做预处理，避免抢占 UI 线程
        return max(1, min(cores, 4, limit))
    return max(1, min(cores, limit))


def _choose_tier(
    system: SystemInfo, family: str, preferred_tier: str, gpu_accelerated: bool
) -> tuple[ModelTier, str]:
    if preferred_tier in {"lite", "full", "heavy"}:
        return preferred_tier, f"用户指定模型档位 {preferred_tier}"  # type: ignore[return-value]

    if not gpu_accelerated:
        # 只跑 CPU 时，档位与显卡无关，避免"有卡但没用上"导致误判
        return "lite", "实际使用 CPU 推理，建议 lite 档以保证实时性"

    if system.is_apple_silicon:
        unified = system.memory.total_mb or 0
        if unified >= APPLE_HEAVY_MEMORY_MB:
            return "heavy", f"Apple Silicon 统一内存 {unified} MB，可用 heavy 档"
        return "full", "Apple Silicon 使用 full 档"

    vram = system.max_gpu_memory_mb()
    if vram >= TIER_HEAVY_VRAM_MB:
        return "heavy", f"显存 {vram} MB ≥ {TIER_HEAVY_VRAM_MB} MB，可用 heavy 档"
    if vram >= TIER_FULL_VRAM_MB:
        return "full", f"显存 {vram} MB，可用 full 档"
    if family in GPU_FAMILIES:
        return "full", "GPU 加速可用（显存未知），使用 full 档"
    return "lite", "CPU 推理，建议 lite 档以保证实时性"


def _match_device(system: SystemInfo, family: str) -> str | None:
    if not system.gpus:
        return None
    alias = {"tensorrt": "cuda", "cuda": "cuda", "coreml": "metal", "directml": "directml"}
    wanted = alias.get(family)
    if wanted:
        for gpu in system.gpus:
            if gpu.api.lower() == wanted or (family == "coreml" and gpu.vendor == "apple"):
                return gpu.name
    return system.gpus[0].name


def decide_acceleration(
    system: SystemInfo,
    *,
    max_threads: int | None = None,
    preferred_tier: str | None = None,
    settings: Settings | None = None,
) -> AccelerationPlan:
    settings = settings or get_settings()
    max_threads = max_threads or settings.max_inference_threads
    preferred_tier = preferred_tier or settings.prefer_model_tier

    available = list(system.onnx_providers)
    reasons: list[str] = []

    preferred_provider: str | None = None
    family = "cpu"
    for wanted in PRIORITY:
        real = _find_provider(available, wanted)
        if real:
            preferred_provider = real
            family, label, description = PROVIDER_META[wanted]
            reasons.append(f"命中 {label}：{description}")
            break

    if preferred_provider is None:
        cpu = _find_provider(available, "cpuexecutionprovider")
        preferred_provider = cpu or "CPUExecutionProvider"
        if cpu:
            reasons.append("未发现 GPU 执行提供器，使用 CPU 多线程")
        else:
            reasons.append("onnxruntime 未报告可用执行提供器，兜底 CPU")
        if system.gpus:
            # 有卡但 EP 缺失是常见坑，明确指出可执行的修复动作
            reasons.append(
                f"已检测到 GPU（{system.gpus[0].name}），但当前 onnxruntime 未启用 CUDA/CoreML EP；"
                "安装 onnxruntime-gpu（或 macOS 的 onnxruntime）后可加速"
            )

    gpu_accelerated = family in GPU_FAMILIES

    # EP 顺序：首选 + 其余可用（按优先级），CPU 永远在最后
    execution_providers: list[str] = []
    ordered = [preferred_provider]
    for wanted in PRIORITY:
        real = _find_provider(available, wanted)
        if real:
            ordered.append(real)
    cpu_provider = _find_provider(available, "cpuexecutionprovider") or "CPUExecutionProvider"
    ordered.append(cpu_provider)
    for name in ordered:
        if name not in execution_providers:
            execution_providers.append(name)

    threads = _threads_for(gpu_accelerated, system, max_threads)
    reasons.append(
        f"推理线程 {threads}（物理核 {system.cpu.physical_cores}，上限 {max_threads}）"
    )

    model_tier, tier_reason = _choose_tier(system, family, preferred_tier, gpu_accelerated)
    reasons.append(tier_reason)

    device_name = _match_device(system, family) if gpu_accelerated else system.cpu.model

    confidence: Literal["high", "medium", "low"] = "high"
    if not available:
        confidence = "low"
    elif not gpu_accelerated:
        confidence = "medium"

    return AccelerationPlan(
        preferred_provider=preferred_provider,
        device_name=device_name,
        gpu_accelerated=gpu_accelerated,
        execution_providers=execution_providers,
        fallback_chain=execution_providers[1:],
        threads=threads,
        model_tier=model_tier,
        model_tier_reason=tier_reason,
        confidence=confidence,
        reasons=reasons,
    )


def build_hardware_report(settings: Settings | None = None) -> HardwareReport:
    settings = settings or get_settings()
    system = collect_system_info()
    return HardwareReport(
        generated_at=datetime.now(timezone.utc),
        system=system,
        acceleration=decide_acceleration(system, settings=settings),
    )
