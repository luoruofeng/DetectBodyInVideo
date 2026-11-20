"""硬件探测与推理引擎（任务 2 / 任务 4）。

对外入口：
- ``select_engine(plan)``：按加速计划选引擎并完成 prepare，失败自动降级。
- ``PoseFrame`` / ``Landmark`` / ``BoundingBox``：全项目统一的姿态数据结构。
- ``MediaPipePoseEngine`` / ``OnnxPoseEngine``：两种实现（延迟导入依赖）。
"""

from __future__ import annotations

from app.backend.engine.base import (
    LANDMARK_COUNT,
    LANDMARK_NAMES,
    BoundingBox,
    EngineCapabilities,
    EngineError,
    EngineSelection,
    EngineUnavailableError,
    Landmark,
    PoseEngine,
    PoseFrame,
)
from app.backend.engine.consistency import (
    ConsistencyReport,
    compare_engines_on_video,
    compare_frames,
)
from app.backend.engine.enhancements import MediaPipeRefiner, RefinementConfig, RefinementResult
from app.backend.engine.mediapipe_engine import MediaPipePoseEngine
from app.backend.engine.onnx_engine import OnnxPoseEngine
from app.backend.engine.registry import (
    create_engine,
    describe_engines,
    register_engine,
    registered_engines,
    select_engine,
)
from app.backend.engine.tracking import (
    IouTracker,
    TrackSummary,
    filter_track,
    primary_track_id,
    summarize_tracks,
)

__all__ = [
    "LANDMARK_COUNT",
    "LANDMARK_NAMES",
    "BoundingBox",
    "ConsistencyReport",
    "EngineCapabilities",
    "EngineError",
    "EngineSelection",
    "EngineUnavailableError",
    "IouTracker",
    "Landmark",
    "MediaPipePoseEngine",
    "MediaPipeRefiner",
    "OnnxPoseEngine",
    "PoseEngine",
    "PoseFrame",
    "RefinementConfig",
    "RefinementResult",
    "TrackSummary",
    "compare_engines_on_video",
    "compare_frames",
    "create_engine",
    "describe_engines",
    "filter_track",
    "primary_track_id",
    "register_engine",
    "registered_engines",
    "select_engine",
    "summarize_tracks",
]
