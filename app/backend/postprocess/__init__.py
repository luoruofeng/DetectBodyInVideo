"""后处理：平滑、补帧、手部稳定、关节角度、质检（任务 5）。

对外入口：
- ``run_postprocess(frames, config)``：完整流水线（选 track → 补帧 → 平滑 → 手部稳定 → 角度 → 质检）。
- ``SmoothingConfig`` / ``GapFillConfig`` / ``HandStabilityConfig`` / ``AngleConfig`` /
  ``QualityConfig``：各段可独立配置。
"""

from __future__ import annotations

from app.backend.postprocess.angles import (
    ANGLE_NAMES,
    JOINT_ANGLE_TRIPLETS,
    AngleConfig,
    angle_deg,
    angle_series,
    attach_angles,
    compute_angles,
    torso_lean,
)
from app.backend.postprocess.gaps import (
    GapFillConfig,
    GapFillReport,
    GapSpan,
    densify,
    fill_short_gaps,
    mark_invalid,
    process_gaps,
)
from app.backend.postprocess.hands import (
    HandSideStats,
    HandStabilityConfig,
    HandStabilityReport,
    stabilize_hands,
)
from app.backend.postprocess.one_euro import OneEuroFilter1D, OneEuroParams, OneEuroVectorFilter
from app.backend.postprocess.pipeline import (
    PostProcessConfig,
    PostProcessResult,
    run_postprocess,
)
from app.backend.postprocess.quality import (
    JUMP_LANDMARKS,
    VERDICT_LABELS,
    QualityConfig,
    QualityReport,
    build_quality_report,
    max_frame_jump,
)
from app.backend.postprocess.smoothing import (
    SmoothingConfig,
    SmoothingInfo,
    SmoothingMethod,
    moving_average,
    savitzky_golay,
    smooth_frames,
)

__all__ = [
    "ANGLE_NAMES",
    "JOINT_ANGLE_TRIPLETS",
    "JUMP_LANDMARKS",
    "VERDICT_LABELS",
    "AngleConfig",
    "GapFillConfig",
    "GapFillReport",
    "GapSpan",
    "HandSideStats",
    "HandStabilityConfig",
    "HandStabilityReport",
    "OneEuroFilter1D",
    "OneEuroParams",
    "OneEuroVectorFilter",
    "PostProcessConfig",
    "PostProcessResult",
    "QualityConfig",
    "QualityReport",
    "SmoothingConfig",
    "SmoothingInfo",
    "SmoothingMethod",
    "angle_deg",
    "angle_series",
    "attach_angles",
    "build_quality_report",
    "compute_angles",
    "densify",
    "fill_short_gaps",
    "mark_invalid",
    "max_frame_jump",
    "moving_average",
    "process_gaps",
    "run_postprocess",
    "savitzky_golay",
    "smooth_frames",
    "stabilize_hands",
    "torso_lean",
]
