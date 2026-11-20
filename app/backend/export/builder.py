"""把分析结果装配成导出文档（任务 7.1 / 7.2 的"建模"部分）。

三种粒度：
- ``keypoints``：仅关键点（18 个 ARKit 核心关节 + 可选关节），不含角度；
- ``full``：关键点 + 关节角度（默认）；
- ``compact``：精简，只留 18 个 ARKit 关节，无角度、无原始 33 点。

浮点在装配阶段统一压到 ``round_digits``（默认 4）位，既压缩体积又保证
iOS 端数值可比对。真正的大文件流式写盘在 ``writer.py``。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:  # 仅为类型提示，避免导出层反向依赖流水线
    from app.backend.postprocess.pipeline import PostProcessResult

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, LANDMARK_NAMES, PoseFrame
from app.backend.export.schema import (
    DEPTH_IS_ESTIMATED,
    GENERATOR,
    SCHEMA_VERSION,
    ExportDocument,
    ExportFrame,
    JointOut,
    QualityOut,
    RootMotionFrameOut,
    RootMotionOut,
    SourceInfo,
)
from app.backend.media.probe import VideoProbe
from app.backend.postprocess.quality import QualityReport
from app.backend.skeleton.bundle import SkeletonBundle
from app.backend.skeleton.mapping import APPLE_ALL_JOINTS, ARKIT_JOINTS, OPTIONAL_JOINTS
from app.backend.skeleton.transform import ConvertedFrame

logger = get_logger("export.builder")

Granularity = Literal["keypoints", "full", "compact"]
GRANULARITIES: tuple[Granularity, ...] = ("keypoints", "full", "compact")

# 深度可信度低于这个值就在导出 JSON 里写 warnings（任务 5.4）
DEPTH_CONFIDENCE_WARN_THRESHOLD = 0.6

_UNSAFE_NAME = re.compile(r"[^\w\u4e00-\u9fff.\-]+", re.UNICODE)


class ExportConfig(BaseModel):
    granularity: Granularity = "full"
    include_raw_landmarks: bool = Field(
        False, description="附带原始 33 点（便于 iOS 侧二次重映射，体积约翻倍）"
    )
    round_digits: int = Field(4, ge=0, le=6)
    include_root_motion: bool = True


def _round(value: float, digits: int) -> float:
    """四舍五入并消除 ``-0.0``（否则 JSON 里会出现 `-0.0`，影响 diff 稳定性）。"""
    if not math.isfinite(value):
        raise ValueError(f"导出数值必须是有限值，实际 {value!r}")
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def joint_names_for(granularity: Granularity, bundle: SkeletonBundle) -> list[str]:
    if granularity == "compact":
        return list(ARKIT_JOINTS)
    known = set(ARKIT_JOINTS) | set(OPTIONAL_JOINTS) | set(APPLE_ALL_JOINTS)
    unknown = [name for name in bundle.joint_names if name not in known]
    if unknown:
        raise ValueError(f"骨骼映射出现未登记的关节：{unknown}")
    return list(bundle.joint_names)


def source_from_probe(filename: str, probe: VideoProbe) -> SourceInfo:
    width = probe.display_width or probe.width
    height = probe.display_height or probe.height
    return SourceInfo(
        file=Path(filename).name,
        fps=round(float(probe.fps or 0) or 30.0, 4),
        width=int(width) or 1,
        height=int(height) or 1,
        frames=int(probe.frame_count or 0),
        durationSeconds=round(float(probe.duration_s or 0.0), 4) or None,
    )


def _joints_from_frame(
    frame: ConvertedFrame, names: Sequence[str], digits: int
) -> dict[str, JointOut]:
    joints: dict[str, JointOut] = {}
    for name in names:
        point = frame.joints.get(name)
        if point is None:
            continue
        joints[name] = JointOut(
            x=_round(point.x, digits),
            y=_round(point.y, digits),
            z=_round(point.z, digits),
            visibility=_round(float(point.visibility), digits),
        )
    return joints


def _raw_landmarks(frame: PoseFrame | None, digits: int) -> dict[str, JointOut] | None:
    if frame is None or len(frame.landmarks) < LANDMARK_COUNT:
        return None
    return {
        name: JointOut(
            x=_round(point.x, digits),
            y=_round(point.y, digits),
            z=_round(point.z, digits),
            visibility=_round(float(point.visibility), digits),
        )
        for name, point in zip(LANDMARK_NAMES, frame.landmarks, strict=False)
    }


def _quality_out(report: QualityReport, digits: int) -> QualityOut:
    return QualityOut(
        framesTotal=report.frames_total,
        framesValid=report.frames_valid,
        validFrameRatio=_round(report.valid_frame_ratio, min(digits, 4)),
        interpolatedFrames=report.interpolated_frames,
        meanVisibility=_round(report.mean_visibility, min(digits, 4)),
        meanScore=_round(report.mean_score, min(digits, 4)),
        maxJump=_round(report.max_jump, digits),
        maxJumpFrame=report.max_jump_frame,
        zJitterRms=_round(report.z_jitter_rms, digits + 2),
        boneLengthCv=_round(report.bone_length_cv, min(digits, 4)),
        meanReprojError=_round(report.mean_reproj_error, min(digits, 4)),
        depthConfidence=_round(report.depth_confidence, min(digits, 4)),
        verdict=report.verdict,
        verdictLabel=report.verdict_label,
        issues=list(report.issues),
        smoothing=dict(report.smoothing),
        notes=list(report.notes),
    )


def depth_warnings(
    report: QualityReport, *, threshold: float = DEPTH_CONFIDENCE_WARN_THRESHOLD
) -> list[str]:
    """导出时给消费方的可信度提示（任务 5.4）。

    只在深度确实不可信时才写，避免每份文件都带一堆无意义的话；
    消费方（iOS / macOS）应当把 ``warnings`` 原样展示给用户。
    """
    warnings: list[str] = []
    if report.depth_confidence >= threshold:
        return warnings
    warnings.append(
        f"depthConfidence={report.depth_confidence:.2f} < {threshold:.2f}："
        "前后方向（深度）是单目估计值，绝对深度与前后位移不可信，"
        "只可用关节的相对方向与角度"
    )
    if report.z_jitter_rms > 0:
        warnings.append(
            f"深度抖动 {report.z_jitter_rms * 100:.1f} cm/帧、骨长波动 {report.bone_length_cv:.1%}："
            "矢状面动作（下蹲/弓步）建议从正侧面重拍，2D 关节角在侧视图下几乎没有前缩误差"
        )
    if report.mean_reproj_error <= 0 and report.bone_length_cv > 0:
        warnings.append("本次没有可用的 3D/2D 一致性数据（骨长拟合未运行），深度可信度只能按抖动估")
    return warnings


def _root_motion_out(bundle: SkeletonBundle, digits: int) -> RootMotionOut | None:
    track = bundle.root_motion
    if not track.frames:
        return None
    return RootMotionOut(
        mode=track.mode,
        totalTranslation=_round(track.total_translation, digits),
        convention=track.convention,
        frames=[
            RootMotionFrameOut(
                index=item.index,
                t=_round(item.t, digits),
                translation=item.translation.model_copy(
                    update={
                        "x": _round(item.translation.x, digits),
                        "y": _round(item.translation.y, digits),
                        "z": _round(item.translation.z, digits),
                    }
                ),
                rotation=item.rotation.model_copy(
                    update={
                        "x": _round(item.rotation.x, digits),
                        "y": _round(item.rotation.y, digits),
                        "z": _round(item.rotation.z, digits),
                        "w": _round(item.rotation.w, digits),
                    }
                ),
            )
            for item in track.frames
        ],
    )


def build_export_document(
    *,
    source: SourceInfo,
    skeleton: SkeletonBundle,
    quality: QualityReport,
    angles: Mapping[int, Mapping[str, float]] | None = None,
    pose_frames: Sequence[PoseFrame] | None = None,
    config: ExportConfig | None = None,
) -> ExportDocument:
    """按粒度把骨骼 + 角度 + 质检装配成 ``ExportDocument``。"""
    config = config or ExportConfig()
    digits = config.round_digits
    names = joint_names_for(config.granularity, skeleton)
    with_angles = config.granularity == "full"

    raw_by_index: dict[int, PoseFrame] = {}
    if config.include_raw_landmarks:
        raw_by_index = {frame.index: frame for frame in (pose_frames or [])}

    notes: list[str] = list(skeleton.notes)
    notes.append(
        "joints 为髋中心相对位置（米、Y 向上），不是父关节局部坐标；"
        "两关节位置相减即骨段向量，向量长度单位也是米"
    )
    notes.append("personHeightMeters 只是导出元数据（由世界坐标测量），不参与骨骼坐标或缩放计算")
    if config.include_raw_landmarks:
        notes.append(
            "rawLandmarks 是原始归一化坐标，不受顶层 units=meters 约定约束："
            "x/y 相对图像宽/高，Y 向下，Z 为髋相对深度且与 x 同尺度"
        )
    if config.granularity != "compact" and not set(OPTIONAL_JOINTS) <= set(names):
        notes.append(
            "本次骨骼映射未启用可选关节（forearm/toes），导出仅含 18 个核心关节；"
            "如需完整关节请打开 SkeletonConfig.include_optional"
        )
    if config.include_raw_landmarks and not raw_by_index:
        notes.append("请求导出原始 33 点，但分析结果里没有可用的 33 点数据")
    elif config.include_raw_landmarks and len(raw_by_index) < len(skeleton.frames):
        notes.append(f"{len(skeleton.frames) - len(raw_by_index)} 帧缺少原始 33 点，已跳过")

    frames: list[ExportFrame] = []
    for converted in skeleton.frames:
        frame_angles: dict[str, float] = {}
        if with_angles and angles is not None:
            frame_angles = {
                name: _round(value, digits)
                for name, value in angles.get(converted.index, {}).items()
                if math.isfinite(value)
            }
        frames.append(
            ExportFrame(
                index=converted.index,
                t=_round(converted.t, digits),
                valid=converted.valid,
                interpolated=converted.interpolated,
                joints=_joints_from_frame(converted, names, digits),
                angles=frame_angles,
                rawLandmarks=_raw_landmarks(raw_by_index.get(converted.index), digits)
                if config.include_raw_landmarks
                else None,
            )
        )

    person_height = skeleton.person_height_meters
    return ExportDocument(
        version=SCHEMA_VERSION,
        generator=GENERATOR,
        source=source,
        coordinateSystem=skeleton.coordinate_system,
        units=skeleton.units,
        depthIsEstimated=DEPTH_IS_ESTIMATED,
        depthConfidence=_round(quality.depth_confidence, min(digits, 4)),
        warnings=depth_warnings(quality),
        personHeightMeters=_round(person_height, digits),
        jointNames=names,
        frames=frames,
        quality=_quality_out(quality, digits),
        rootMotion=_root_motion_out(skeleton, digits) if config.include_root_motion else None,
        notes=notes,
    )


class AnalysisLike(Protocol):
    """``AnalysisResult`` 的最小契约（避免导出层 import 流水线造成循环依赖）。"""

    skeleton: SkeletonBundle
    postprocess: PostProcessResult


def document_from_analysis(
    result: AnalysisLike,
    source: SourceInfo,
    config: ExportConfig | None = None,
) -> ExportDocument:
    """从 ``AnalysisResult`` 装配文档。"""
    config = config or ExportConfig()
    skeleton = result.skeleton
    postprocess = result.postprocess
    angles = {
        frame.index: values
        for frame, values in zip(postprocess.frames, postprocess.angles, strict=False)
    }
    return build_export_document(
        source=source,
        skeleton=skeleton,
        quality=postprocess.quality,
        angles=angles,
        pose_frames=postprocess.frames,
        config=config,
    )


def safe_stem(original: str) -> str:
    """把源文件名压成可安全落盘的文件名主干（保留中文）。"""
    stem = Path(original).stem or "video"
    return _UNSAFE_NAME.sub("_", stem).strip("._") or "video"


def timestamp_stamp(timestamp: datetime | str | None = None) -> str:
    """把时间戳统一成 ``YYYYmmdd-HHMMSS``；字符串会先做字符清洗。"""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y%m%d-%H%M%S")
    if isinstance(timestamp, str):
        return _UNSAFE_NAME.sub("-", timestamp)
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def export_filename(
    original: str,
    granularity: Granularity,
    *,
    timestamp: datetime | str | None = None,
) -> str:
    """``{原名}_{时间戳}_pose{mode}.json``（任务 7.2）。"""
    return f"{safe_stem(original)}_{timestamp_stamp(timestamp)}_pose{granularity}.json"


__all__ = [
    "DEPTH_CONFIDENCE_WARN_THRESHOLD",
    "GRANULARITIES",
    "ExportConfig",
    "Granularity",
    "build_export_document",
    "depth_warnings",
    "document_from_analysis",
    "export_filename",
    "joint_names_for",
    "safe_stem",
    "source_from_probe",
    "timestamp_stamp",
]
