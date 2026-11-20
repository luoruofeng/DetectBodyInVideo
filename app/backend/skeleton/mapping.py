"""MediaPipe 关键点 → AppleModel.usda 关节映射（任务 6.1 + 全关节覆盖）。

``resources/AppleModel.usda`` 的 Skeleton 声明了 **91 个关节**；本模块按该顺序
输出同名关节，可直接驱动该 USDZ 骨架。分层：

- ``ARKIT_JOINTS``（18）：躯干主链 + 四肢，默认粒度，保持向后兼容；
- ``OPTIONAL_JOINTS``（4）：前臂中点 + 脚趾，旧版可选扩展；
- ``EXTRA_JOINTS``（69）：脊柱/颈部细分、下颌/眼/虹膜、脚尖末端与双手 48 个指关节。

需要全部 91 个关节时使用 ``include_all=True``（``SkeletonConfig.include_all_joints``）。
数据来源：33 点姿态 + 每手 21 点 + 面部语义点（由 ``MediaPipeRefiner`` 提供）。
无增强数据时指关节/面部细分会记入 ``missing``，不伪造坐标。

缺失关节的推导：
- ``hips_joint`` / ``root``：两髋中点；``spine_*``：髋中点 → 颈根线性插值；
- ``neck_*``：肩中点 → 头中心线性插值；``head_joint``：两耳中点（缺耳用鼻尖）；
- ``*_toesEnd_joint``：踝 → 脚尖方向按 0.22 外推；
- 指端 ``*End_joint``：用指尖 - 上一指节方向按 0.35 外推。

左右绝不混用：``validate_mapping()`` 断言 left_* 只来自左侧关键点/左手，
单测与运行时（导出前）都可以调用。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LANDMARK_NAMES,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HIP,
    LEFT_INDICES,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HIP,
    RIGHT_INDICES,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
    PoseFrame,
)
from app.backend.engine.enhancements import FACE_SUBSET_NAMES, HAND_LANDMARK_COUNT

logger = get_logger("skeleton.mapping")

# 18 个核心关节（顺序即 JSON 中 jointNames 的顺序）
ARKIT_JOINTS: tuple[str, ...] = (
    "root",
    "hips_joint",
    "spine_3_joint",
    "spine_7_joint",
    "neck_1_joint",
    "head_joint",
    "left_shoulder_1_joint",
    "left_arm_joint",
    "left_hand_joint",
    "right_shoulder_1_joint",
    "right_arm_joint",
    "right_hand_joint",
    "left_upLeg_joint",
    "left_leg_joint",
    "left_foot_joint",
    "right_upLeg_joint",
    "right_leg_joint",
    "right_foot_joint",
)

# 可选扩展关节（旧版"完整"粒度）
OPTIONAL_JOINTS: tuple[str, ...] = (
    "left_forearm_joint",
    "right_forearm_joint",
    "left_toes_joint",
    "right_toes_joint",
)

# resources/AppleModel.usda 中 Skeleton.joints 的全部 91 个关节，顺序与模型一致。
# 前 22 个即 ARKIT_JOINTS + OPTIONAL_JOINTS（顺序不同但集合一致），其余为细分关节。
_FINGERS: tuple[str, ...] = ("Pinky", "Ring", "Mid", "Index")
APPLE_ALL_JOINTS: tuple[str, ...] = (
    "root",
    "hips_joint",
    "left_upLeg_joint",
    "left_leg_joint",
    "left_foot_joint",
    "left_toes_joint",
    "left_toesEnd_joint",
    "right_upLeg_joint",
    "right_leg_joint",
    "right_foot_joint",
    "right_toes_joint",
    "right_toesEnd_joint",
    "spine_1_joint",
    "spine_2_joint",
    "spine_3_joint",
    "spine_4_joint",
    "spine_5_joint",
    "spine_6_joint",
    "spine_7_joint",
    "neck_1_joint",
    "neck_2_joint",
    "neck_3_joint",
    "neck_4_joint",
    "head_joint",
    "jaw_joint",
    "chin_joint",
    "nose_joint",
    "right_eye_joint",
    "right_eyeUpperLid_joint",
    "right_eyeLowerLid_joint",
    "right_eyeball_joint",
    "left_eye_joint",
    "left_eyeUpperLid_joint",
    "left_eyeLowerLid_joint",
    "left_eyeball_joint",
    "right_shoulder_1_joint",
    "right_arm_joint",
    "right_forearm_joint",
    "right_hand_joint",
    *(
        f"right_hand{finger}{part}_joint"
        for finger in _FINGERS
        for part in ("Start", "_1", "_2", "_3", "End")
    ),
    "right_handThumbStart_joint",
    "right_handThumb_1_joint",
    "right_handThumb_2_joint",
    "right_handThumbEnd_joint",
    "left_shoulder_1_joint",
    "left_arm_joint",
    "left_forearm_joint",
    "left_hand_joint",
    *(
        f"left_hand{finger}{part}_joint"
        for finger in _FINGERS
        for part in ("Start", "_1", "_2", "_3", "End")
    ),
    "left_handThumbStart_joint",
    "left_handThumb_1_joint",
    "left_handThumb_2_joint",
    "left_handThumbEnd_joint",
)

# 旧 22 关节之外新增的关节（供需要"只加细分、不加旧可选"的场景参考）
EXTRA_JOINTS: tuple[str, ...] = tuple(
    name for name in APPLE_ALL_JOINTS if name not in set(ARKIT_JOINTS) | set(OPTIONAL_JOINTS)
)

# MediaPipe 手部 21 点 → AppleModel 指关节索引。
# MediaPipe 顺序：0 腕；1-4 拇指 CMC/MCP/IP/TIP；5-8 食指；9-12 中指；13-16 无名指；17-20 小指。
# AppleModel 每指有 Start/_1/_2/_3/End 五节，拇指只有 Start/_1/_2/End 四节。
# 值：int = 直接取该索引；tuple(tip, previous) = 指尖沿末节方向外推。
_FINGER_SEGMENTS: dict[str, tuple[int, int, int, int]] = {
    "Pinky": (17, 18, 19, 20),
    "Ring": (13, 14, 15, 16),
    "Mid": (9, 10, 11, 12),
    "Index": (5, 6, 7, 8),
    "Thumb": (1, 2, 3, 4),
}
HandSource = int | tuple[int, int]


def _hand_joint_indices(side: str) -> dict[str, HandSource]:
    result: dict[str, HandSource] = {}
    for finger, (mcp, pip, dip, tip) in _FINGER_SEGMENTS.items():
        parts = ("Start", "_1", "_2", "_3", "End") if finger != "Thumb" else ("Start", "_1", "_2", "End")
        sources: tuple[HandSource, ...] = (
            (mcp, pip, dip, tip, (tip, dip)) if finger != "Thumb" else (mcp, pip, dip, tip)
        )
        for part, source in zip(parts, sources, strict=False):
            result[f"{side}_hand{finger}{part}_joint"] = source
    return result


HAND_JOINT_SOURCES: dict[str, HandSource] = {
    **_hand_joint_indices("left"),
    **_hand_joint_indices("right"),
}
FINGER_TIP_EXTRAPOLATION = 0.35

# 直接映射：ARKit 关节 → MediaPipe 索引
DIRECT_MAP: dict[str, int] = {
    "left_shoulder_1_joint": LEFT_SHOULDER,
    "left_arm_joint": LEFT_ELBOW,
    "left_hand_joint": LEFT_WRIST,
    "right_shoulder_1_joint": RIGHT_SHOULDER,
    "right_arm_joint": RIGHT_ELBOW,
    "right_hand_joint": RIGHT_WRIST,
    "left_upLeg_joint": LEFT_HIP,
    "left_leg_joint": LEFT_KNEE,
    "left_foot_joint": LEFT_ANKLE,
    "right_upLeg_joint": RIGHT_HIP,
    "right_leg_joint": RIGHT_KNEE,
    "right_foot_joint": RIGHT_ANKLE,
}

# 可选关节的源索引（用于左右校验）
OPTIONAL_MAP: dict[str, int] = {
    "left_toes_joint": LEFT_FOOT_INDEX,
    "right_toes_joint": RIGHT_FOOT_INDEX,
}

DERIVED_JOINTS: tuple[str, ...] = (
    "root",
    "hips_joint",
    "spine_3_joint",
    "spine_7_joint",
    "neck_1_joint",
    "head_joint",
)

LEFT_JOINTS: tuple[str, ...] = tuple(name for name in (*ARKIT_JOINTS, *OPTIONAL_JOINTS) if name.startswith("left_"))
RIGHT_JOINTS: tuple[str, ...] = tuple(name for name in (*ARKIT_JOINTS, *OPTIONAL_JOINTS) if name.startswith("right_"))

SPINE_3_RATIO = 0.40
SPINE_7_RATIO = 0.75
NECK_RATIO = 0.35
TOES_END_RATIO = 0.22
DERIVED_VISIBILITY_FACTOR = 0.9


class JointPoint(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0
    source: Literal["direct", "derived", "estimated"] = "direct"


class SkeletonFrame(BaseModel):
    index: int = 0
    timestamp_ms: float = 0.0
    valid: bool = True
    interpolated: bool = False
    track_id: int | None = None
    coordinate_space: Literal["normalized", "world"] = "normalized"
    joints: dict[str, JointPoint] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)

    def joint(self, name: str) -> JointPoint | None:
        return self.joints.get(name)


def joint_names(include_optional: bool = False, include_all: bool = False) -> list[str]:
    if include_all:
        return list(APPLE_ALL_JOINTS)
    return list(ARKIT_JOINTS) + (list(OPTIONAL_JOINTS) if include_optional else [])


def _midpoint(a: Landmark, b: Landmark, *, source: str = "derived") -> JointPoint:
    return JointPoint(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0,
        visibility=min(a.visibility, b.visibility) * DERIVED_VISIBILITY_FACTOR,
        source=source,  # type: ignore[arg-type]
    )


def _lerp(a: Landmark | JointPoint, b: Landmark | JointPoint, ratio: float) -> JointPoint:
    return JointPoint(
        x=a.x + (b.x - a.x) * ratio,
        y=a.y + (b.y - a.y) * ratio,
        z=a.z + (b.z - a.z) * ratio,
        visibility=min(a.visibility, b.visibility) * DERIVED_VISIBILITY_FACTOR,
        source="derived",
    )


def _to_point(landmark: Landmark) -> JointPoint:
    return JointPoint(
        x=landmark.x,
        y=landmark.y,
        z=landmark.z,
        visibility=landmark.visibility,
        source="direct",
    )


def _extrapolate(a: Landmark, b: Landmark, ratio: float) -> JointPoint:
    """从 a→b 方向按 ratio 外推，生成末端关节（趾尖/指尖）。"""
    return JointPoint(
        x=b.x + (b.x - a.x) * ratio,
        y=b.y + (b.y - a.y) * ratio,
        z=b.z + (b.z - a.z) * ratio,
        visibility=min(a.visibility, b.visibility) * DERIVED_VISIBILITY_FACTOR,
        source="derived",
    )


def _local_world_scale(frame: PoseFrame) -> float | None:
    """估计 1 归一化单位对应的米数（各向同性近似），用于把手/脸归一化点锚定到 world 空间。"""
    if len(frame.landmarks) < LANDMARK_COUNT or len(frame.world_landmarks) < LANDMARK_COUNT:
        return None
    norm = frame.landmarks
    world = frame.world_landmarks
    norm_span = abs(norm[NOSE].y - (norm[LEFT_HIP].y + norm[RIGHT_HIP].y) / 2.0)
    world_span = abs(world[NOSE].y - (world[LEFT_HIP].y + world[RIGHT_HIP].y) / 2.0)
    if norm_span > 1e-4 and world_span > 1e-4:
        return world_span / norm_span
    return None


def _anchored_point(
    point: Landmark,
    anchor_norm: Landmark,
    anchor_world: Landmark | None,
    scale: float | None,
    use_world: bool,
) -> JointPoint:
    """归一化点 → 目标空间：world 模式以锚点的 world 位置 + 相对偏移 × 尺度换算。"""
    if not use_world or anchor_world is None or scale is None:
        return _to_point(point)
    return JointPoint(
        x=anchor_world.x + (point.x - anchor_norm.x) * scale,
        y=anchor_world.y + (point.y - anchor_norm.y) * scale,
        z=anchor_world.z + (point.z - anchor_norm.z) * scale,
        visibility=point.visibility,
        source="estimated",
    )


_SPINE_RATIOS: dict[int, float] = {1: 0.08, 2: 0.22, 4: 0.55, 5: 0.65, 6: 0.72}
_NECK_RATIOS: dict[int, float] = {2: 0.55, 3: 0.75, 4: 0.92}


def _add_face_joints(
    joints: dict[str, JointPoint],
    frame: PoseFrame,
    points: list[Landmark],
    *,
    use_world: bool,
    scale: float | None,
) -> None:
    if not frame.face or (use_world and scale is None):
        return
    face = dict(zip(FACE_SUBSET_NAMES, frame.face, strict=False))
    nose_norm = frame.landmarks[NOSE] if frame.landmarks else Landmark()
    nose_world = frame.world_landmarks[NOSE] if use_world and len(frame.world_landmarks) > NOSE else None

    def put(name: str, point: Landmark | None) -> None:
        if point is not None:
            joints[name] = _anchored_point(point, nose_norm, nose_world, scale, use_world)

    chin = face.get("chin")
    put("chin_joint", chin)
    put("jaw_joint", face.get("jaw") or chin)
    put("nose_joint", face.get("nose_tip") or (points[NOSE] if points else None))
    for side in ("left", "right"):
        inner = face.get(f"{side}_eye_inner")
        outer = face.get(f"{side}_eye_outer")
        eye = _midpoint(inner, outer, source="estimated") if inner and outer else None
        if eye is not None:
            joints[f"{side}_eye_joint"] = _anchored_point(
                _joint_to_landmark(eye), nose_norm, nose_world, scale, use_world
            )
        put(f"{side}_eyeUpperLid_joint", face.get(f"{side}_eye_upper_lid"))
        put(f"{side}_eyeLowerLid_joint", face.get(f"{side}_eye_lower_lid"))
        put(f"{side}_eyeball_joint", face.get(f"{side}_iris") or eye)


def _add_hand_joints(
    joints: dict[str, JointPoint],
    frame: PoseFrame,
    *,
    use_world: bool,
    scale: float | None,
) -> None:
    if not frame.hands or (use_world and scale is None):
        return
    for hand_index, hand in enumerate(frame.hands):
        if len(hand) < HAND_LANDMARK_COUNT:
            continue
        side = (frame.hand_sides[hand_index].lower() if hand_index < len(frame.hand_sides) else "")
        if side not in {"left", "right"}:
            # 没有 handedness 时按与姿态腕点的距离回退分配
            norm = frame.landmarks
            if len(norm) <= LEFT_WRIST:
                continue
            d_left = _distance(hand[0], norm[LEFT_WRIST])
            d_right = _distance(hand[0], norm[RIGHT_WRIST])
            side = "left" if d_left <= d_right else "right"
        wrist_index = LEFT_WRIST if side == "left" else RIGHT_WRIST
        anchor_norm = hand[0]
        anchor_world = (
            frame.world_landmarks[wrist_index]
            if use_world and len(frame.world_landmarks) > wrist_index
            else None
        )
        for name, source in HAND_JOINT_SOURCES.items():
            if not name.startswith(f"{side}_"):
                continue
            if isinstance(source, int):
                point = hand[source]
            else:
                tip, previous = source
                point = _joint_to_landmark(_extrapolate(hand[previous], hand[tip], FINGER_TIP_EXTRAPOLATION))
            joints[name] = _anchored_point(point, anchor_norm, anchor_world, scale, use_world)


def _distance(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def map_frame(
    frame: PoseFrame,
    *,
    use_world: bool = False,
    include_optional: bool = False,
    include_all: bool = False,
    world_scale: float | None = None,
) -> SkeletonFrame:
    """把一帧关键点映射为关节字典；``include_all`` 输出 AppleModel 全部 91 关节。

    ``world_scale`` 是"1 归一化单位 = 多少米"的**整段共用**常数（见 ``map_frames``）；
    留空则退回逐帧反解，手指/面部关节会随帧伸缩。
    """
    # 请求世界点时不能用归一化点补位，否则下游会把无量纲数值误当成米。
    source_landmarks = frame.world_landmarks if use_world else frame.landmarks
    points = list(source_landmarks[:LANDMARK_COUNT])
    names = joint_names(include_optional, include_all)
    skeleton = SkeletonFrame(
        index=frame.index,
        timestamp_ms=frame.timestamp_ms,
        valid=frame.valid and len(points) == LANDMARK_COUNT,
        interpolated=frame.interpolated,
        track_id=frame.track_id,
        coordinate_space="world" if use_world else "normalized",
    )
    if len(points) < LANDMARK_COUNT:
        skeleton.missing = names
        return skeleton

    joints: dict[str, JointPoint] = {}
    for name, index in DIRECT_MAP.items():
        joints[name] = _to_point(points[index])

    hips_mid = _midpoint(points[LEFT_HIP], points[RIGHT_HIP])
    shoulders_mid = _midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])

    ears_visible = min(points[LEFT_EAR].visibility, points[RIGHT_EAR].visibility) >= 0.3
    head_reference = (
        _midpoint(points[LEFT_EAR], points[RIGHT_EAR], source="derived")
        if ears_visible
        else _to_point(points[NOSE])
    )
    neck = _lerp(shoulders_mid, head_reference, NECK_RATIO)
    neck_landmark = _joint_to_landmark(neck)
    hips_landmark = _joint_to_landmark(hips_mid)
    shoulders_landmark = _joint_to_landmark(shoulders_mid)

    joints["root"] = hips_mid.model_copy(update={"source": "derived"})
    joints["hips_joint"] = hips_mid.model_copy(update={"source": "derived"})
    joints["spine_3_joint"] = _lerp(hips_landmark, neck_landmark, SPINE_3_RATIO)
    joints["spine_7_joint"] = _lerp(hips_landmark, neck_landmark, SPINE_7_RATIO)
    joints["neck_1_joint"] = neck
    joints["head_joint"] = JointPoint(
        x=head_reference.x,
        y=head_reference.y,
        z=head_reference.z,
        visibility=head_reference.visibility,
        source="estimated",
    )

    if include_optional or include_all:
        joints["left_forearm_joint"] = _lerp(points[LEFT_ELBOW], points[LEFT_WRIST], 0.5)
        joints["right_forearm_joint"] = _lerp(points[RIGHT_ELBOW], points[RIGHT_WRIST], 0.5)
        for name, index in OPTIONAL_MAP.items():
            joints[name] = _to_point(points[index])

    if include_all:
        for level, ratio in _SPINE_RATIOS.items():
            joints[f"spine_{level}_joint"] = _lerp(hips_landmark, neck_landmark, ratio)
        for level, ratio in _NECK_RATIOS.items():
            joints[f"neck_{level}_joint"] = _lerp(shoulders_landmark, head_reference, ratio)
        for side, ankle, toes in (
            ("left", LEFT_ANKLE, LEFT_FOOT_INDEX),
            ("right", RIGHT_ANKLE, RIGHT_FOOT_INDEX),
        ):
            joints[f"{side}_toesEnd_joint"] = _extrapolate(points[ankle], points[toes], TOES_END_RATIO)
        scale = world_scale
        if scale is None and use_world:
            scale = _local_world_scale(frame)
        _add_face_joints(joints, frame, points, use_world=use_world, scale=scale)
        _add_hand_joints(joints, frame, use_world=use_world, scale=scale)

    ordered = {name: joints[name] for name in names if name in joints}
    skeleton.joints = ordered
    skeleton.missing = [name for name in names if name not in ordered]
    return skeleton


def _joint_to_landmark(point: JointPoint) -> Landmark:
    return Landmark(x=point.x, y=point.y, z=point.z, visibility=point.visibility)


def map_frames(
    frames: Sequence[PoseFrame],
    *,
    use_world: bool = False,
    include_optional: bool = False,
    include_all: bool = False,
) -> list[SkeletonFrame]:
    """整段映射。世界尺度用整段中位数，保证手/面关节的段长不随帧伸缩。"""
    ordered = sorted(frames, key=lambda item: item.index)
    world_scale: float | None = None
    if use_world:
        scales = [
            value
            for frame in ordered
            if (value := _local_world_scale(frame)) is not None and value > 1e-6
        ]
        if scales:
            world_scale = statistics.median(scales)
    return [
        map_frame(
            frame,
            use_world=use_world,
            include_optional=include_optional,
            include_all=include_all,
            world_scale=world_scale,
        )
        for frame in ordered
    ]


def validate_mapping() -> list[str]:
    """校验左右不串 + 关节齐全；返回问题列表（空表示通过）。"""
    problems: list[str] = []

    for name, index in DIRECT_MAP.items():
        if name.startswith("left_") and index not in LEFT_INDICES:
            problems.append(f"{name} 映射到了非左侧关键点 {LANDMARK_NAMES[index]}")
        if name.startswith("right_") and index not in RIGHT_INDICES:
            problems.append(f"{name} 映射到了非右侧关键点 {LANDMARK_NAMES[index]}")

    for name, index in OPTIONAL_MAP.items():
        if name.startswith("left_") and index not in LEFT_INDICES:
            problems.append(f"{name} 映射到了非左侧关键点 {LANDMARK_NAMES[index]}")
        if name.startswith("right_") and index not in RIGHT_INDICES:
            problems.append(f"{name} 映射到了非右侧关键点 {LANDMARK_NAMES[index]}")

    duplicated = {index for index in DIRECT_MAP.values() if list(DIRECT_MAP.values()).count(index) > 1}
    if duplicated:
        problems.append(f"多个关节共用同一关键点：{sorted(duplicated)}")

    if not set(DERIVED_JOINTS) <= set(ARKIT_JOINTS):
        problems.append("推导关节未包含在 ARKIT_JOINTS 中")

    if len(ARKIT_JOINTS) != 18:
        problems.append(f"核心关节数量应为 18，实际 {len(ARKIT_JOINTS)}")
    if len(APPLE_ALL_JOINTS) != 91:
        problems.append(f"AppleModel 全关节数应为 91，实际 {len(APPLE_ALL_JOINTS)}")
    if len(set(APPLE_ALL_JOINTS)) != len(APPLE_ALL_JOINTS):
        problems.append("AppleModel 全关节清单存在重复项")
    if not (set(ARKIT_JOINTS) | set(OPTIONAL_JOINTS)) <= set(APPLE_ALL_JOINTS):
        problems.append("旧 22 关节未全部包含在 AppleModel 全关节清单中")
    for name in HAND_JOINT_SOURCES:
        if name not in APPLE_ALL_JOINTS:
            problems.append(f"手指关节 {name} 不在 AppleModel 全关节清单中")

    return problems


__all__ = [
    "APPLE_ALL_JOINTS",
    "ARKIT_JOINTS",
    "DERIVED_JOINTS",
    "DIRECT_MAP",
    "EXTRA_JOINTS",
    "HAND_JOINT_SOURCES",
    "LEFT_JOINTS",
    "OPTIONAL_JOINTS",
    "OPTIONAL_MAP",
    "RIGHT_JOINTS",
    "JointPoint",
    "SkeletonFrame",
    "joint_names",
    "map_frame",
    "map_frames",
    "validate_mapping",
]
