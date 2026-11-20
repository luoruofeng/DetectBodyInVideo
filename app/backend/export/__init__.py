"""JSON 导出与 schema（任务 7）。

对外入口：
- ``build_export_document`` / ``document_from_analysis``：装配文档；
- ``write_document`` / ``export_document``：流式写盘 + 回读自校验；
- ``ExportDocument`` / ``SCHEMA_VERSION``：契约本身；
- ``build_motion_clip`` / ``write_motion_clip``：转成 iOS App 的动作 JSON。
"""

from __future__ import annotations

from app.backend.export.builder import (
    DEPTH_CONFIDENCE_WARN_THRESHOLD,
    GRANULARITIES,
    ExportConfig,
    Granularity,
    build_export_document,
    depth_warnings,
    document_from_analysis,
    export_filename,
    joint_names_for,
    safe_stem,
    source_from_probe,
    timestamp_stamp,
)
from app.backend.export.motion_clip import (
    BODY_JOINTS,
    DEFAULT_REFERENCE_MODEL,
    DEFAULT_SAMPLE_INTERVAL_MS,
    MOTION_COORDINATE_SPACE,
    MOTION_SCHEMA_VERSION,
    MotionClipConfig,
    MotionClipDocument,
    MotionFrameOut,
    build_motion_clip,
    motion_filename,
    motion_from_analysis,
    read_motion_clip,
    write_motion_clip,
)
from app.backend.export.schema import (
    SCHEMA_FILE,
    SCHEMA_VERSION,
    ExportDocument,
    ExportFrame,
    JointOut,
    QualityOut,
    RootMotionOut,
    SourceInfo,
    schema_payload,
    write_schema_json,
)
from app.backend.export.writer import (
    ExportArtifact,
    ExportError,
    ExportValidationError,
    export_document,
    latest_export,
    read_document,
    write_document,
)

__all__ = [
    "BODY_JOINTS",
    "DEFAULT_REFERENCE_MODEL",
    "DEPTH_CONFIDENCE_WARN_THRESHOLD",
    "DEFAULT_SAMPLE_INTERVAL_MS",
    "GRANULARITIES",
    "MOTION_COORDINATE_SPACE",
    "MOTION_SCHEMA_VERSION",
    "MotionClipConfig",
    "MotionClipDocument",
    "MotionFrameOut",
    "SCHEMA_FILE",
    "SCHEMA_VERSION",
    "ExportArtifact",
    "ExportConfig",
    "ExportDocument",
    "ExportError",
    "ExportFrame",
    "ExportValidationError",
    "Granularity",
    "JointOut",
    "QualityOut",
    "RootMotionOut",
    "SourceInfo",
    "build_export_document",
    "build_motion_clip",
    "depth_warnings",
    "document_from_analysis",
    "export_document",
    "export_filename",
    "joint_names_for",
    "latest_export",
    "motion_filename",
    "motion_from_analysis",
    "read_document",
    "read_motion_clip",
    "safe_stem",
    "schema_payload",
    "source_from_probe",
    "timestamp_stamp",
    "write_document",
    "write_motion_clip",
    "write_schema_json",
]
