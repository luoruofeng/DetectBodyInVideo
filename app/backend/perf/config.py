"""推理链路运行时配置（任务 9.1 / 9.3）。

所有参数都可以从 ``Settings`` 覆盖，UI 也能按次传参：
- ``queue_size``：生产/消费之间的有界队列长度，满了就背压，不会把内存吃爆；
- ``batch_size``：加速 EP 上的批大小；不支持批处理的引擎会自动退化为 1；
- ``worker_threads``：ONNX intra-op 线程数，0 = 沿用 ``AccelerationPlan.threads``；
- ``fp16``：半精度策略（``auto`` 只在 CUDA/CoreML 上尝试，失败回退 FP32）；
- ``segment_frames``：分段长度，0 = 不分段；用于长视频进度与断点落盘；
- ``checkpoint`` / ``resume``：写 / 读断点，支持"关掉再开继续算"。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.backend.core.settings import Settings


class RuntimeConfig(BaseModel):
    queue_size: int = Field(64, ge=1, le=4096, description="有界队列长度（背压保护）")
    batch_size: int = Field(4, ge=1, le=32, description="加速 EP 上的批大小")
    worker_threads: int = Field(0, ge=0, le=64, description="0 = 沿用加速计划")
    fp16: Literal["auto", "on", "off"] = Field("auto", description="半精度策略")
    segment_frames: int = Field(0, ge=0, description="分段长度（帧），0 = 不分段")
    checkpoint: bool = Field(False, description="把每帧结果落盘，便于中断后续算")
    resume: bool = Field(False, description="读取既有断点并跳过已完成帧")

    @model_validator(mode="after")
    def _resume_implies_checkpoint(self) -> RuntimeConfig:
        if self.resume:
            self.checkpoint = True
        return self

    @property
    def effective_batch_size(self) -> int:
        return max(1, self.batch_size)

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> RuntimeConfig:
        base = cls(
            queue_size=settings.runtime_queue_size,
            batch_size=settings.runtime_batch_size,
            worker_threads=0,
            segment_frames=settings.runtime_segment_frames,
        )
        return base.model_copy(update=overrides) if overrides else base


__all__ = ["RuntimeConfig"]
