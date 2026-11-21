"""任务 6 的对外装配：33 点 → ARKit 关节 → 米制坐标 → rootMotion。

一步产出导出层（任务 7）需要的东西，避免调用方手工串联三个模块。
米制只来自世界坐标（world landmarks）；没有该输出的引擎不再回退任何"身高标定"。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame
from app.backend.skeleton.ground import (
    FOOT_JOINTS,
    GroundContactConfig,
    GroundReference,
    anchor_feet_to_ground,
)
from app.backend.skeleton.mapping import (
    APPLE_ALL_JOINTS,
    ARKIT_JOINTS,
    OPTIONAL_JOINTS,
    joint_names,
    map_frames,
    validate_mapping,
)
from app.backend.skeleton.root_motion import RootMotionConfig, RootMotionTrack, build_root_motion
from app.backend.skeleton.transform import (
    COORDINATE_SYSTEM,
    UNITS,
    ConvertedFrame,
    convert_frames,
)

logger = get_logger("skeleton.bundle")

# 参考身高取较直立帧的分位数，避免深蹲帧把"头到脚"量短。
HEIGHT_QUANTILE = 0.9
# 明显的检测错误排除（只留正常人形跨度）
MIN_HEIGHT_SPAN_METERS = 0.2
MAX_HEIGHT_SPAN_METERS = 2.6
# 量不到时写进导出 JSON 的兜底值，仅作元数据
FALLBACK_PERSON_HEIGHT_METERS = 1.70
# 导出 schema 的合法区间（gt 0.5 / le 2.5）：超出说明测量被遮挡或误检，直接回退
EXPORT_MIN_HEIGHT_METERS = 0.5
EXPORT_MAX_HEIGHT_METERS = 2.5


class SkeletonConfig(BaseModel):
    include_optional: bool = False
    include_all_joints: bool = Field(
        False, description="输出 resources/AppleModel.usda 的全部 91 个关节（前端默认开启）"
    )
    root_motion_mode: Literal["world", "in_place"] = "world"
    normalize_root_at_start: bool = True
    anchor_feet_to_ground: bool = Field(
        True,
        description="按脚踝/脚尖着地锚定地面：恢复 world 模式下被髋原点抹掉的下蹲下沉，"
        "并把支撑脚钉在平滑后的地面高度上（原地动作时脚不会随关键点噪声上下跳）",
    )


class SkeletonBundle(BaseModel):
    joint_names: list[str] = Field(default_factory=list)
    coordinate_system: str = COORDINATE_SYSTEM
    units: str = UNITS
    person_height_meters: float = Field(
        FALLBACK_PERSON_HEIGHT_METERS,
        description="由世界坐标自动测量的参考高度（头中心→脚底），不参与任何缩放",
    )
    frames: list[ConvertedFrame] = Field(default_factory=list)
    root_motion: RootMotionTrack = Field(default_factory=RootMotionTrack)
    notes: list[str] = Field(default_factory=list)


def measure_person_height(frames: Sequence[ConvertedFrame]) -> float | None:
    """整段世界坐标里「头中心 → 最低脚点」的竖直跨度（米），量不到返回 ``None``。

    纯元数据：导出 JSON 的 ``personHeightMeters`` 用它，骨骼坐标不受影响。
    头中心是两耳中点/鼻尖，不含头顶，因此比真实身高偏小约 10%。
    """
    spans: list[float] = []
    for frame in frames:
        if not frame.valid:
            continue
        head = frame.joints.get("head_joint")
        if head is None:
            continue
        lows = [
            point.y
            for name in FOOT_JOINTS
            if (point := frame.joints.get(name)) is not None and point.visibility >= 0.3
        ]
        if not lows:
            continue
        span = head.y - min(lows)
        if MIN_HEIGHT_SPAN_METERS <= span <= MAX_HEIGHT_SPAN_METERS:
            spans.append(span)
    if not spans:
        return None
    ordered = sorted(spans)
    position = (len(ordered) - 1) * HEIGHT_QUANTILE
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_skeleton_bundle(
    frames: Sequence[PoseFrame], config: SkeletonConfig | None = None
) -> SkeletonBundle:
    config = config or SkeletonConfig()
    problems = validate_mapping()
    if problems:  # 映射表写错属于开发期错误，必须立刻暴露
        raise ValueError("ARKit 映射表校验失败：" + "；".join(problems))

    ordered = sorted(frames, key=lambda item: item.index)
    map_kwargs = {
        "include_optional": config.include_optional,
        "include_all": config.include_all_joints,
    }
    has_world = any(len(frame.world_landmarks) >= LANDMARK_COUNT for frame in ordered)
    # 缺 world landmarks 的帧会得到 valid=False / 空关节，绝不会拿归一化坐标冒充米制。
    world = map_frames(ordered, use_world=True, **map_kwargs)
    converted = convert_frames(world)
    ground_references: list[GroundReference] = []
    ground_config = GroundContactConfig()
    if config.anchor_feet_to_ground:
        converted, ground_references = anchor_feet_to_ground(converted, ground_config)
    root_motion = build_root_motion(
        converted,
        RootMotionConfig(
            mode=config.root_motion_mode,
            normalize_start=config.normalize_root_at_start,
        ),
    )

    notes: list[str] = []
    for reference in ground_references:
        notes.append(
            f"人物轨迹 {reference.track_id}：按 {reference.frames_used} 帧脚踝/脚尖着地锚定"
            f"估计站立参考腿长 {reference.foot_span_meters:.2f} m，"
            "world 模式的 root.y 已恢复下蹲时的髋部下沉（脚底保持在地面）"
        )
        if (
            reference.span_jitter_before_meters > 0
            and ground_config.smooth_cutoff_hz > 0
        ):
            notes.append(
                f"支撑脚跨度已做 {ground_config.smooth_cutoff_hz:.1f} Hz 零相位高斯低通 + "
                f"{ground_config.switch_margin_meters * 100:.0f} cm 支撑脚切换滞回："
                f"逐帧抖动 {reference.span_jitter_before_meters * 100:.1f} cm → "
                f"{reference.span_jitter_after_meters * 100:.1f} cm，"
                "原地动作的髋部上下抖动随之被压掉"
            )
        if ground_config.lock_support_foot:
            notes.append(
                "支撑脚已按平滑后的跨度锁定（两骨 IK 重排膝，骨长不变）："
                "脚底高度在整段里恒定，不再随关键点噪声离地/落地"
            )
    if not has_world:
        notes.append(
            "当前引擎没有输出 world landmarks（米制世界坐标），3D 关节为空；"
            "请改用 MediaPipe 引擎"
        )

    measured = measure_person_height(converted)
    if measured is None or not (EXPORT_MIN_HEIGHT_METERS < measured <= EXPORT_MAX_HEIGHT_METERS):
        person_height = FALLBACK_PERSON_HEIGHT_METERS
        detail = f"实测 {measured:.2f} m" if measured is not None else "世界坐标不足"
        notes.append(
            f"参考身高不可用（{detail}），personHeightMeters 回退 {person_height:.2f} m"
            "（仅作导出元数据，不参与坐标计算）"
        )
    else:
        person_height = measured
        notes.append(
            f"personHeightMeters={person_height:.2f} m 由世界坐标自动测量"
            "（头中心→脚踝/脚趾跨度，不含头顶），仅作缩放核对参考，不参与骨骼坐标计算"
        )

    all_names = joint_names(config.include_optional, config.include_all_joints)
    if config.include_all_joints:
        missing_names = sorted(set(APPLE_ALL_JOINTS) - set(all_names))
        if missing_names:
            notes.append(f"AppleModel 全关节清单有 {len(missing_names)} 个关节未参与映射")
    return SkeletonBundle(
        joint_names=all_names,
        coordinate_system=COORDINATE_SYSTEM,
        units=UNITS,
        person_height_meters=person_height,
        frames=converted,
        root_motion=root_motion,
        notes=notes,
    )


__all__ = [
    "ARKIT_JOINTS",
    "OPTIONAL_JOINTS",
    "SkeletonBundle",
    "SkeletonConfig",
    "build_skeleton_bundle",
    "measure_person_height",
]
