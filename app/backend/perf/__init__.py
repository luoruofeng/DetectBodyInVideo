"""推理运行时：生产/消费、批处理、断点续算（任务 9.1 / 9.3）。"""

from __future__ import annotations

from app.backend.perf.checkpoint import FrameCheckpoint, checkpoint_key
from app.backend.perf.config import RuntimeConfig
from app.backend.perf.runner import RuntimeProgress, RuntimeResult, run_inference

__all__ = [
    "FrameCheckpoint",
    "RuntimeConfig",
    "RuntimeProgress",
    "RuntimeResult",
    "checkpoint_key",
    "run_inference",
]
