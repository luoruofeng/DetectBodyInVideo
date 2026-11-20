"""把姿态导出文档转换成 iOS App 使用的动作 JSON（``hip-local-meters-yup``）。

格式以 ``resources/air-squat-motion.json`` 为唯一参考：

- 每帧存 17 个身体关节的**髋中心相对三维坐标**（米，原点 = pelvis，Y 轴向上），
  固定间隔采样（默认 100ms / 10Hz）；
- 运行时**只取"两个关节连成的向量"当骨骼方向**，骨长与位置全部沿用模型自己的
  rest 骨架，所以 JSON 里的骨长比例不参与驱动；
- ``pelvisOffsetMeters`` 单独记录骨盆相对"站立参考帧"的位移（下蹲下沉 / 腾空），
  因为关节坐标是髋局部坐标，不含整体高低。

本模块只做三件事：

1. 关节改名：ARKit 18 关节 → App 的 ``root / pelvis / spine / neck / centerShoulder``
   与四肢命名；
2. 原点归到 pelvis（``pelvis`` / ``root`` 恒为 ``[0,0,0]``），单位米、Y 向上；
3. 按固定间隔重采样：有效帧之间线性插值，无效帧不参与插值（间隔过大则保持前一帧）。

手指关节（48 个）不在此生成：本项目主链路只有 33 点姿态，没有手部数据；
运行时未绑定的关节会继承父骨骼姿态，不会回弹到 T-Pose。

命令行用法（把已有姿态 JSON 转成动作 JSON）::

    python -m app.backend.export.motion_clip clip_posefull.json -o clip_motion.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.backend.core.logging import get_logger
from app.backend.export.builder import (
    AnalysisLike,
    ExportConfig,
    document_from_analysis,
    safe_stem,
    timestamp_stamp,
)
from app.backend.export.schema import (
    ExportDocument,
    FiniteFloat,
    SourceInfo,
)
from app.backend.export.writer import ExportArtifact, ExportValidationError, read_document

logger = get_logger("export.motion_clip")

MOTION_SCHEMA_VERSION = 1
MOTION_COORDINATE_SPACE = "hip-local-meters-yup"
DEFAULT_SAMPLE_INTERVAL_MS = 100
DEFAULT_REFERENCE_MODEL = "t-pose_humanbody_2048.usdz"
DEFAULT_HOLD_GAP_MS = 400
MIN_LEG_SPAN_METERS = 0.2

JOINT_VISIBILITY_NOTE = (
    "jointVisibility 只列 visibility < 1 的关节，缺省视为 1；"
    "数值来自视频逐帧检测，低置信度关节由运行时按权重参与驱动。"
)

# App 动作 JSON 的 17 个身体关节（顺序即 skeleton 顺序）
BODY_JOINTS: tuple[str, ...] = (
    "root",
    "pelvis",
    "spine",
    "neck",
    "centerShoulder",
    "leftShoulder",
    "leftElbow",
    "leftWrist",
    "rightShoulder",
    "rightElbow",
    "rightWrist",
    "leftHip",
    "leftKnee",
    "leftAnkle",
    "rightHip",
    "rightKnee",
    "rightAnkle",
)

# App 关节名 → 本项目导出的 ARKit 关节名
POSE_JOINT_MAP: dict[str, str] = {
    "root": "root",
    "pelvis": "hips_joint",
    "spine": "spine_3_joint",
    "neck": "neck_1_joint",
    "centerShoulder": "spine_7_joint",  # 下方使用两肩中点覆盖，不以脊柱插值点冒充肩中心
    "leftShoulder": "left_shoulder_1_joint",
    "leftElbow": "left_arm_joint",
    "leftWrist": "left_hand_joint",
    "rightShoulder": "right_shoulder_1_joint",
    "rightElbow": "right_arm_joint",
    "rightWrist": "right_hand_joint",
    "leftHip": "left_upLeg_joint",
    "leftKnee": "left_leg_joint",
    "leftAnkle": "left_foot_joint",
    "rightHip": "right_upLeg_joint",
    "rightKnee": "right_leg_joint",
    "rightAnkle": "right_foot_joint",
}

Vec3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Visibility = Annotated[float, Field(ge=0, le=1)]

_SLUG = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# 文档模型（与 air-squat-motion.json 字段一一对应）
# --------------------------------------------------------------------------- #
class MotionFrameOut(BaseModel):
    """一帧动作数据。``t`` 为相对片段起点的毫秒，``joints`` 为 pelvis 局部米制坐标。"""

    model_config = ConfigDict(extra="forbid")

    t: int = Field(ge=0, description="相对片段起点的毫秒时间戳")
    joints: dict[str, Vec3] = Field(default_factory=dict)
    jointVisibility: dict[str, Visibility] = Field(
        default_factory=dict, description="只记录 visibility < 1 的关节，缺省视为 1"
    )
    pelvisOffsetMeters: Vec3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="骨盆相对站立参考帧的位移（米，Y 向上）",
    )
    confidence: Visibility = 1.0


class MotionViewpoint(BaseModel):
    """观看基准：人物正面朝播放器，-Z 为前方。"""

    model_config = ConfigDict(extra="forbid")

    subject: str = "source-video-subject-facing-player"
    space: str = "hip-local-yup"
    forwardAxis: Literal["-Z", "+Z"] = "-Z"
    viewProfile: str = "frontal"
    recommendedViewYawDegrees: float = 0.0
    stageTransformDepthSign: int = -1


class MotionClipDocument(BaseModel):
    """App 侧可解析的动作片段。"""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal[1] = MOTION_SCHEMA_VERSION
    exerciseID: str = Field(min_length=1)
    name: str = Field(min_length=1)
    nameZh: str | None = None
    coordinateSpace: Literal["hip-local-meters-yup"] = MOTION_COORDINATE_SPACE
    bodyOrientation: Literal["upright"] = "upright"
    referenceModel: str = DEFAULT_REFERENCE_MODEL
    mirroredReference: bool = True
    sampleIntervalMs: int = Field(DEFAULT_SAMPLE_INTERVAL_MS, ge=10)
    durationMs: int = Field(0, ge=0)
    loop: bool = False
    demonstrationLoops: int = Field(1, ge=1)
    skeleton: list[str] = Field(default_factory=list)
    keyframesMs: list[int] = Field(default_factory=list)
    playbackHolds: list[int] = Field(default_factory=list)
    frames: list[MotionFrameOut] = Field(min_length=1)
    viewpoint: MotionViewpoint = Field(default_factory=MotionViewpoint)
    jointVisibilityNote: str = JOINT_VISIBILITY_NOTE

    @model_validator(mode="after")
    def _check_document(self) -> MotionClipDocument:
        if len(set(self.skeleton)) != len(self.skeleton):
            raise ValueError("skeleton 存在重复项")
        allowed = set(self.skeleton)
        previous_t: int | None = None
        for frame in self.frames:
            unknown = set(frame.joints) - allowed
            if unknown:
                raise ValueError(f"时间 {frame.t}ms 的帧出现未声明的关节：{sorted(unknown)[:5]}")
            if previous_t is not None and frame.t <= previous_t:
                raise ValueError(f"frames[].t 必须严格递增：{previous_t} → {frame.t}")
            previous_t = frame.t
        if self.durationMs != self.frames[-1].t:
            raise ValueError(
                f"durationMs 必须等于最后一帧时间：{self.durationMs} != {self.frames[-1].t}"
            )
        return self


class MotionClipConfig(BaseModel):
    """转换参数。默认值即 iOS App 期望的 10Hz / 米制 / 髋局部坐标。"""

    sample_interval_ms: int = Field(DEFAULT_SAMPLE_INTERVAL_MS, ge=10, le=1000)
    loop: bool = False
    exercise_id: str | None = Field(None, description="留空则由文件名生成 slug")
    name: str | None = None
    name_zh: str | None = None
    reference_model: str = DEFAULT_REFERENCE_MODEL
    mirrored_reference: bool = True
    round_digits: int = Field(4, ge=0, le=6)
    hold_gap_ms: int = Field(
        DEFAULT_HOLD_GAP_MS,
        ge=0,
        description="两帧有效姿态间隔超过该值时保持前一帧，不做插值",
    )
    include_pelvis_offset: bool = True


# --------------------------------------------------------------------------- #
# 中间结构
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Sample:
    """一帧待重采样的中间数据（未取整）。"""

    index: int
    t_ms: float
    valid: bool
    joints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    visibility: dict[str, float] = field(default_factory=dict)
    pelvis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = 0.0
    leg_span: float = 0.0


def _round(value: float, digits: int) -> float:
    """取整并消除 ``-0.0``；非有限值直接报错（App 端解析会崩）。"""
    if not math.isfinite(value):
        raise ExportValidationError(f"动作 JSON 数值必须是有限值，实际 {value!r}")
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _leg_span(
    joints: dict[str, tuple[float, float, float]], visibility: dict[str, float]
) -> float:
    """骨盆到最低脚踝的竖直距离（骨盆在原点，脚踝 y 为负）。"""
    ys = [
        joints[name][1]
        for name in ("leftAnkle", "rightAnkle")
        if name in joints and visibility.get(name, 0.0) >= 0.3
    ]
    if not ys:
        return 0.0
    span = -min(ys)
    return span if span > 0 else 0.0


def _sample_from_frame(frame) -> _Sample:
    """把一帧姿态导出数据转成 pelvis 局部坐标（未取整）。"""
    raw = frame.joints
    origin_point = raw.get("hips_joint") or raw.get("root")
    if origin_point is None:
        raise ExportValidationError(
            f"第 {frame.index} 帧缺少 pelvis（hips_joint/root），无法生成动作 JSON",
            details={"frame": frame.index},
        )
    origin = (origin_point.x, origin_point.y, origin_point.z)

    joints: dict[str, tuple[float, float, float]] = {}
    visibility: dict[str, float] = {}
    for app_name, pose_name in POSE_JOINT_MAP.items():
        point = raw.get(pose_name)
        if point is None:
            raise ExportValidationError(
                f"第 {frame.index} 帧缺少关节 {pose_name}（App 关节 {app_name}），"
                "请用 keypoints/full 粒度重新导出后再转换",
                details={"frame": frame.index, "joint": pose_name},
            )
        joints[app_name] = (point.x - origin[0], point.y - origin[1], point.z - origin[2])
        visibility[app_name] = float(point.visibility)

    # 原点关节恒为 0，既符合 App 的 hip-local 约定，也避免浮点噪声
    joints["root"] = (0.0, 0.0, 0.0)
    joints["pelvis"] = (0.0, 0.0, 0.0)
    joints["centerShoulder"] = tuple(
        (joints["leftShoulder"][axis] + joints["rightShoulder"][axis]) / 2.0
        for axis in range(3)
    )
    visibility["centerShoulder"] = min(
        visibility["leftShoulder"], visibility["rightShoulder"]
    )

    return _Sample(
        index=frame.index,
        t_ms=float(frame.t) * 1000.0,
        valid=bool(frame.valid),
        joints=joints,
        visibility=visibility,
        pelvis=(0.0, 0.0, 0.0),
        confidence=min(1.0, max(0.0, _mean([visibility[name] for name in BODY_JOINTS]))),
        leg_span=_leg_span(joints, visibility),
    )


def _root_translations(document: ExportDocument) -> dict[int, tuple[float, float, float]]:
    track = document.rootMotion
    if track is None:
        return {}
    return {
        item.index: (item.translation.x, item.translation.y, item.translation.z)
        for item in track.frames
    }


def _apply_pelvis_offsets(
    samples: list[_Sample],
    document: ExportDocument,
    config: MotionClipConfig,
) -> None:
    """计算 ``pelvisOffsetMeters``。

    两套来源，优先级从高到低：

    1. ``rootMotion`` 有非零位移（world 模式下由脚部着地锚定恢复的髋部下沉）：
       直接用 root 位移差分；
    2. 没有可用位移：用"骨盆到脚踝的竖直距离"相对最直立的帧估出下沉量；
       假设脚踝着地，腾空无法还原。
    """
    if not config.include_pelvis_offset or not samples:
        return

    references = [sample for sample in samples if sample.leg_span >= MIN_LEG_SPAN_METERS]
    reference = max(references, key=lambda sample: sample.leg_span) if references else samples[0]
    reference_span = reference.leg_span
    reference_index = reference.index

    translations = _root_translations(document)
    observed = [translations[sample.index] for sample in samples if sample.index in translations]
    # totalTranslation 是首尾净位移，蹲下再站起/走出再返回时为 0。
    # 必须检查整条轨迹，而且只在每个有效样本都有平移时使用该来源。
    has_global = (
        document.rootMotion is not None
        and document.rootMotion.mode == "world"
        and len(observed) == len(samples)
        and any(math.dist(value, observed[0]) > 1e-6 for value in observed[1:])
    )

    if has_global:
        reference_translation = translations[reference_index]
        for sample in samples:
            current = translations[sample.index]
            sample.pelvis = (
                current[0] - reference_translation[0],
                current[1] - reference_translation[1],
                current[2] - reference_translation[2],
            )
        return

    for sample in samples:
        # 骨盆离脚越近（leg_span 变小）→ 偏移越负，即下蹲下沉
        if references and sample.leg_span >= MIN_LEG_SPAN_METERS:
            sample.pelvis = (0.0, sample.leg_span - reference_span, 0.0)


def _lerp(a: float, b: float, weight: float) -> float:
    return a + (b - a) * weight


def _blend(left: _Sample, right: _Sample, weight: float) -> _Sample:
    joints = {
        name: tuple(
            _lerp(left.joints[name][axis], right.joints[name][axis], weight) for axis in range(3)
        )
        for name in BODY_JOINTS
    }
    visibility = {
        name: _lerp(left.visibility[name], right.visibility[name], weight) for name in BODY_JOINTS
    }
    pelvis = tuple(
        _lerp(left.pelvis[axis], right.pelvis[axis], weight) for axis in range(3)
    )
    return _Sample(
        index=left.index,
        t_ms=_lerp(left.t_ms, right.t_ms, weight),
        valid=True,
        joints=joints,
        visibility=visibility,
        pelvis=pelvis,  # type: ignore[arg-type]
        confidence=_lerp(left.confidence, right.confidence, weight),
        leg_span=_lerp(left.leg_span, right.leg_span, weight),
    )


def _copy(sample: _Sample) -> _Sample:
    return _Sample(
        index=sample.index,
        t_ms=sample.t_ms,
        valid=True,
        joints=dict(sample.joints),
        visibility=dict(sample.visibility),
        pelvis=sample.pelvis,
        confidence=sample.confidence,
        leg_span=sample.leg_span,
    )


def _resample(samples: list[_Sample], config: MotionClipConfig) -> list[MotionFrameOut]:
    """按固定间隔重采样；无效帧不参与插值，长缺口保持前一帧。"""
    usable = sorted((sample for sample in samples if sample.valid), key=lambda item: item.t_ms)
    if not usable:
        raise ExportValidationError("没有可用帧，无法生成动作 JSON")

    timestamps = [sample.t_ms for sample in usable]
    start = timestamps[0]
    end = timestamps[-1]
    interval = float(config.sample_interval_ms)

    targets: list[float] = []
    cursor = start
    while cursor <= end + 0.5:
        targets.append(cursor)
        cursor += interval
    if not targets:
        targets = [start]

    frames: list[MotionFrameOut] = []
    for target in targets:
        position = bisect.bisect_left(timestamps, target)
        if position < len(usable) and abs(timestamps[position] - target) < 0.5:
            sample = _copy(usable[position])
        elif position == 0:
            sample = _copy(usable[0])
        elif position >= len(usable):
            sample = _copy(usable[-1])
        else:
            left, right = usable[position - 1], usable[position]
            gap = right.t_ms - left.t_ms
            weight = 0.0 if gap <= 0 else (target - left.t_ms) / gap
            if gap > config.hold_gap_ms:
                sample = _copy(left)
            else:
                sample = _blend(left, right, min(1.0, max(0.0, weight)))

        digits = config.round_digits
        joints = {
            name: [_round(value, digits) for value in sample.joints[name]] for name in BODY_JOINTS
        }
        visibility = {
            name: _round(value, 2)
            for name, value in sample.visibility.items()
            if round(value, 2) < 1.0
        }
        frames.append(
            MotionFrameOut(
                t=int(round(target - start)),
                joints=joints,
                jointVisibility=visibility,
                pelvisOffsetMeters=[_round(value, digits) for value in sample.pelvis],
                confidence=_round(sample.confidence, 3),
            )
        )
    return frames


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #
def build_motion_clip(
    document: ExportDocument,
    config: MotionClipConfig | None = None,
) -> MotionClipDocument:
    """把姿态导出文档转换为 App 动作 JSON 文档。"""
    config = config or MotionClipConfig()
    if not document.frames:
        raise ExportValidationError("导出文档没有帧，无法生成动作 JSON")

    # 无效帧可能连髋/四肢都不存在，必须在映射及站立参考计算前过滤。
    samples = [_sample_from_frame(frame) for frame in document.frames if frame.valid]
    if not samples:
        raise ExportValidationError("没有有效姿态帧，无法生成动作 JSON")
    _apply_pelvis_offsets(samples, document, config)
    frames = _resample(samples, config)

    name = (config.name or Path(document.source.file).stem or "").strip() or "motion"
    exercise_id = (config.exercise_id or _slug(name)).strip() or "motion"
    duration = frames[-1].t
    keyframes = [0, duration] if duration > 0 else [0]

    clip = MotionClipDocument(
        schemaVersion=MOTION_SCHEMA_VERSION,
        exerciseID=exercise_id,
        name=name,
        nameZh=config.name_zh,
        coordinateSpace=MOTION_COORDINATE_SPACE,
        referenceModel=config.reference_model,
        mirroredReference=config.mirrored_reference,
        sampleIntervalMs=config.sample_interval_ms,
        durationMs=duration,
        loop=config.loop,
        demonstrationLoops=1,
        skeleton=list(BODY_JOINTS),
        keyframesMs=keyframes,
        playbackHolds=[],
        frames=frames,
    )

    return clip


def motion_from_analysis(
    result: AnalysisLike,
    source: SourceInfo,
    config: MotionClipConfig | None = None,
) -> MotionClipDocument:
    """从分析结果直接生成动作 JSON。"""
    document = document_from_analysis(
        result,
        source,
        ExportConfig(
            granularity="keypoints",
            include_raw_landmarks=False,
            include_root_motion=True,
        ),
    )
    return build_motion_clip(document, config)


# --------------------------------------------------------------------------- #
# 落盘 / 回读
# --------------------------------------------------------------------------- #
def _write_stream(clip: MotionClipDocument, sink: object) -> int:
    header = clip.model_dump(mode="json", exclude={"frames"}, exclude_none=True)
    header_text = json.dumps(header, ensure_ascii=False)
    if not header_text.endswith("}"):
        raise ExportValidationError("动作 JSON 头序列化失败")
    sink.write(header_text[:-1])  # type: ignore[attr-defined]
    sink.write(', "frames": [')  # type: ignore[attr-defined]

    written = 0
    first = True
    for frame in clip.frames:
        sink.write(("" if first else ",") + frame.model_dump_json())  # type: ignore[attr-defined]
        first = False
        written += 1
    sink.write("]}")  # type: ignore[attr-defined]
    return written


def read_motion_clip(path: Path | str) -> MotionClipDocument:
    """回读并做 Pydantic 校验（导出后自检、测试共用）。"""
    target = Path(path)
    try:
        return MotionClipDocument.model_validate_json(target.read_text(encoding="utf-8"))
    except ExportValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转成业务错误
        raise ExportValidationError(
            f"动作 JSON 校验失败：{exc}", details={"path": str(target)}
        ) from exc


def write_motion_clip(
    clip: MotionClipDocument,
    path: Path | str,
    *,
    validate: bool = True,
) -> ExportArtifact:
    """原子写盘（先写 ``*.part``）+ 可选回读校验。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    started = time.perf_counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as sink:
            written = _write_stream(clip, sink)
        if validate:
            checked = read_motion_clip(temporary)
            if len(checked.frames) != written:
                raise ExportValidationError(
                    "回读帧数与写入帧数不一致",
                    details={"written": written, "read": len(checked.frames)},
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    artifact = ExportArtifact(
        path=str(target),
        filename=target.name,
        bytes=target.stat().st_size,
        frames=len(clip.frames),
        mode="motion",
        validated=validate,
        elapsedSeconds=round(time.perf_counter() - started, 3),
    )
    logger.info(
        "动作 JSON 导出完成：%s（%s 帧，%.1f KiB，%.2fs）",
        artifact.filename,
        artifact.frames,
        artifact.bytes / 1024,
        artifact.elapsedSeconds,
    )
    return artifact


def motion_filename(
    original: str,
    *,
    timestamp: datetime | str | None = None,
) -> str:
    """``{原名}_{时间戳}_motion.json``。"""
    return f"{safe_stem(original)}_{timestamp_stamp(timestamp)}_motion.json"


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把姿态导出 JSON 转换为 iOS App 动作 JSON（hip-local-meters-yup）"
    )
    parser.add_argument("source", help="姿态导出 JSON 文件")
    parser.add_argument("-o", "--output", help="输出文件（默认与源文件同目录）")
    parser.add_argument("--interval", type=int, default=DEFAULT_SAMPLE_INTERVAL_MS, help="采样间隔 ms")
    parser.add_argument("--loop", action="store_true", help="标记为循环动作")
    parser.add_argument("--name", help="动作名（默认取源文件名）")
    parser.add_argument("--name-zh", dest="name_zh", help="中文动作名")
    args = parser.parse_args(argv)

    source = Path(args.source)
    document = read_document(source)
    clip = build_motion_clip(
        document,
        MotionClipConfig(
            sample_interval_ms=args.interval,
            loop=args.loop,
            name=args.name,
            name_zh=args.name_zh,
        ),
    )
    target = Path(args.output) if args.output else source.with_name(motion_filename(source.name))
    artifact = write_motion_clip(clip, target)
    print(f"动作 JSON 已写入 {artifact.path}（{artifact.frames} 帧，{artifact.bytes} 字节）")
    return 0


__all__ = [
    "BODY_JOINTS",
    "DEFAULT_REFERENCE_MODEL",
    "DEFAULT_SAMPLE_INTERVAL_MS",
    "MOTION_COORDINATE_SPACE",
    "MOTION_SCHEMA_VERSION",
    "MotionClipConfig",
    "MotionClipDocument",
    "MotionFrameOut",
    "MotionViewpoint",
    "POSE_JOINT_MAP",
    "build_motion_clip",
    "main",
    "motion_filename",
    "motion_from_analysis",
    "read_motion_clip",
    "write_motion_clip",
]


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    raise SystemExit(main())
