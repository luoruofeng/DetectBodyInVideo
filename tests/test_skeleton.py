"""骨骼映射 / 坐标转换 / rootMotion 单测（任务 6）。"""

from __future__ import annotations

import math

import pytest

from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.skeleton.bundle import SkeletonConfig, build_skeleton_bundle
from app.backend.skeleton.live import LiveSkeletonMapper
from app.backend.skeleton.mapping import (
    ARKIT_JOINTS,
    map_frame,
    validate_mapping,
)
from app.backend.skeleton.root_motion import (
    RootMotionConfig,
    build_root_motion,
    quaternion_from_yaw,
    unwrap_angles,
    yaw_between,
)
from app.backend.skeleton.transform import (
    ConvertedFrame,
    CoordinateTransform,
    Vector3,
    roundtrip_error,
    to_realitykit_vector,
)

# 标准正面站姿（归一化坐标，y 向下）
STANDARD_POINTS: dict[int, tuple[float, float]] = {
    0: (0.50, 0.18),  # nose
    7: (0.47, 0.18),  # left_ear
    8: (0.53, 0.18),  # right_ear
    11: (0.40, 0.30),
    12: (0.60, 0.30),
    13: (0.35, 0.45),
    14: (0.65, 0.45),
    15: (0.33, 0.60),
    16: (0.67, 0.60),
    23: (0.45, 0.55),
    24: (0.55, 0.55),
    25: (0.45, 0.72),
    26: (0.55, 0.72),
    27: (0.45, 0.90),
    28: (0.55, 0.90),
    29: (0.45, 0.93),
    30: (0.55, 0.93),
    31: (0.45, 0.96),
    32: (0.55, 0.96),
}


def make_pose_frame(
    index: int = 0,
    *,
    dx: float = 0.0,
    world: bool = False,
    visibility: float = 0.95,
) -> PoseFrame:
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
    for landmark_index, (x, y) in STANDARD_POINTS.items():
        landmarks[landmark_index] = Landmark(
            x=x + dx, y=y, z=0.0, visibility=visibility
        )
    frame = PoseFrame(
        index=index,
        timestamp_ms=index * 33.0,
        landmarks=landmarks,
        score=0.95,
    )
    if world:
        # 米制世界坐标：原点≈两髋中点，y 向下
        world_landmarks = []
        for x, y in STANDARD_POINTS.values():
            world_landmarks.append(
                Landmark(
                    x=(x - 0.5) * 0.5 + dx,
                    y=(y - 0.55) * 1.7 / 0.78,
                    z=0.0,
                    visibility=visibility,
                )
            )
        padded = [Landmark() for _ in range(LANDMARK_COUNT)]
        for index_, item in zip(STANDARD_POINTS, world_landmarks, strict=True):
            padded[index_] = item
        frame.world_landmarks = padded
    return frame


# --------------------------------------------------------------------------- #
# 6.1 映射
# --------------------------------------------------------------------------- #
def test_validate_mapping_passes() -> None:
    assert validate_mapping() == []
    assert len(ARKIT_JOINTS) == 18


def test_map_frame_direct_and_derived_joints() -> None:
    frame = make_pose_frame()
    skeleton = map_frame(frame)
    assert set(skeleton.joints) == set(ARKIT_JOINTS)
    assert skeleton.missing == []

    # 直接映射
    assert skeleton.joints["left_shoulder_1_joint"].x == pytest.approx(0.40)
    assert skeleton.joints["right_shoulder_1_joint"].x == pytest.approx(0.60)
    assert skeleton.joints["left_arm_joint"].x == pytest.approx(0.35)
    assert skeleton.joints["right_arm_joint"].x == pytest.approx(0.65)
    assert skeleton.joints["left_foot_joint"].y == pytest.approx(0.90)

    # 推导关节
    assert skeleton.joints["hips_joint"].x == pytest.approx(0.50)
    assert skeleton.joints["root"].y == pytest.approx(0.55)
    spine3 = skeleton.joints["spine_3_joint"]
    spine7 = skeleton.joints["spine_7_joint"]
    assert 0.30 < spine3.y < 0.55
    assert spine3.y > spine7.y  # y 向下：spine_7 更靠上
    assert skeleton.joints["head_joint"].x == pytest.approx(0.50)


def test_left_right_never_swapped() -> None:
    skeleton = map_frame(make_pose_frame())
    left_pairs = [
        ("left_shoulder_1_joint", "right_shoulder_1_joint"),
        ("left_arm_joint", "right_arm_joint"),
        ("left_hand_joint", "right_hand_joint"),
        ("left_upLeg_joint", "right_upLeg_joint"),
        ("left_leg_joint", "right_leg_joint"),
        ("left_foot_joint", "right_foot_joint"),
    ]
    for left, right in left_pairs:
        assert skeleton.joints[left].x < skeleton.joints[right].x, f"{left}/{right} 左右串了"


def test_mirrored_input_swaps_sides() -> None:
    """把画面水平镜像后，左右必须整体互换（不是只在某一侧乱跳）。"""
    original = map_frame(make_pose_frame())
    landmarks = [
        Landmark(x=1.0 - item.x, y=item.y, z=item.z, visibility=item.visibility)
        for item in make_pose_frame().landmarks
    ]
    mirrored_frame = PoseFrame(landmarks=landmarks, score=0.95)
    mirrored = map_frame(mirrored_frame)

    assert mirrored.joints["left_arm_joint"].x == pytest.approx(
        original.joints["right_arm_joint"].x, abs=0.02
    )
    assert mirrored.joints["right_arm_joint"].x == pytest.approx(
        original.joints["left_arm_joint"].x, abs=0.02
    )


# --------------------------------------------------------------------------- #
# 6.2 坐标转换
# --------------------------------------------------------------------------- #
def test_roundtrip_error_is_negligible() -> None:
    point = Vector3(x=1.234, y=-0.456, z=2.789)
    assert roundtrip_error(point) < 1e-6
    flipped = to_realitykit_vector(point)
    assert flipped.x == pytest.approx(point.x)
    assert flipped.y == pytest.approx(-point.y)
    assert flipped.z == pytest.approx(-point.z)


def test_normalized_only_frame_has_no_metric_joints() -> None:
    """没有 world landmarks 就没有米制尺度：不猜、不标定，直接留空。"""
    world = map_frame(make_pose_frame(), use_world=True)
    assert world.coordinate_space == "world"
    converted = CoordinateTransform().convert(world)
    assert converted.valid is False
    assert converted.joints == {}
    assert converted.root.length == pytest.approx(0.0)


def test_world_mode_flips_axes_and_zero_root() -> None:
    frame = make_pose_frame(world=True)
    world = map_frame(frame, use_world=True)
    assert world.coordinate_space == "world"

    converted = CoordinateTransform().convert(world)
    assert converted.valid is True
    assert converted.root.length == pytest.approx(0.0)
    # 头部在髋上方：world y 向下为负，翻转后应为正
    assert converted.joints["head_joint"].y > 0
    assert converted.joints["left_foot_joint"].y < 0


# --------------------------------------------------------------------------- #
# 6.3 rootMotion
# --------------------------------------------------------------------------- #
def test_root_motion_world_mode_tracks_translation() -> None:
    # root.y 由脚部着地锚定给回下蹲下沉；这里直接构造带 root 位移的帧验证装配。
    frames = [
        ConvertedFrame(index=index, t=index * 0.1, root=Vector3(y=-index * 0.05))
        for index in range(5)
    ]
    track = build_root_motion(frames, RootMotionConfig(mode="world"))

    assert len(track.frames) == 5
    assert track.frames[0].translation.length == pytest.approx(0.0)
    assert track.frames[-1].translation.y < 0
    assert track.total_translation > 0
    for item in track.frames:
        length = math.sqrt(item.rotation.x**2 + item.rotation.y**2 + item.rotation.z**2 + item.rotation.w**2)
        assert length == pytest.approx(1.0, abs=1e-4)


def test_root_motion_in_place_zeroes_translation() -> None:
    frames = [
        ConvertedFrame(index=index, t=index * 0.1, root=Vector3(y=-index * 0.05))
        for index in range(4)
    ]
    track = build_root_motion(frames, RootMotionConfig(mode="in_place"))
    assert all(item.translation.length == pytest.approx(0.0) for item in track.frames)
    assert track.mode == "in_place"
    assert track.notes


def test_yaw_and_quaternion() -> None:
    assert yaw_between(Vector3(x=-1), Vector3(x=1)) == pytest.approx(0.0)
    assert yaw_between(Vector3(), Vector3()) is None

    yaw = math.pi / 2
    quaternion = quaternion_from_yaw(yaw)
    assert quaternion.y == pytest.approx(math.sin(yaw / 2))
    assert quaternion.w == pytest.approx(math.cos(yaw / 2))

    # ±180° 处展开成连续角度
    unwrapped = unwrap_angles([math.radians(179), math.radians(-179)])
    assert unwrapped[1] > math.pi


# --------------------------------------------------------------------------- #
# 6.1~6.3 装配
# --------------------------------------------------------------------------- #
def test_bundle_without_world_has_no_metric_joints() -> None:
    """引擎没有 world landmarks 时不再按身高反解尺度，只留提示。"""
    frames = [make_pose_frame(index) for index in range(3)]
    bundle = build_skeleton_bundle(frames)
    assert bundle.coordinate_system == "arkit-right-handed-y-up"
    assert bundle.units == "meters"
    assert len(bundle.frames) == 3
    assert len(bundle.joint_names) == 18
    assert all(frame.valid is False and not frame.joints for frame in bundle.frames)
    assert any("world landmarks" in note for note in bundle.notes)
    assert bundle.person_height_meters == pytest.approx(1.70)  # 量不到时的兜底元数据
    assert bundle.root_motion.frames == []


def test_bundle_with_world_landmarks_uses_world_mode() -> None:
    frames = [make_pose_frame(index, world=True) for index in range(3)]
    bundle = build_skeleton_bundle(frames)
    assert bundle.frames[0].root.length == pytest.approx(0.0)
    # 参考身高由世界坐标实测：头中心（耳中点）→脚踝，比真实身高偏小
    assert bundle.person_height_meters == pytest.approx(1.57, abs=0.01)
    assert any("自动测量" in note for note in bundle.notes)


# --------------------------------------------------------------------------- #
# 推理中的增量映射（3D 预览 / 白膜 Mesh 边算边看）
# --------------------------------------------------------------------------- #
def test_live_mapper_is_incremental_and_matches_transform() -> None:
    frames = [make_pose_frame(index, world=True) for index in range(4)]
    mapper = LiveSkeletonMapper(SkeletonConfig())

    mapper.extend(frames[:2])
    assert mapper.consumed == 2
    mapper.extend(frames)  # 已处理过的帧不重算
    assert mapper.consumed == 4

    entries = mapper.snapshot()
    assert [entry.index for entry in entries] == [0, 1, 2, 3]

    expected = CoordinateTransform().convert(map_frame(frames[0], use_world=True))
    joints = {name: (x, y, z, visibility) for name, x, y, z, visibility in entries[0].as_joints(mapper.joint_names)}
    assert set(joints) == set(mapper.joint_names)
    for name in ("hips_joint", "left_shoulder_1_joint", "head_joint"):
        point = expected.joints[name]
        assert joints[name][0] == pytest.approx(point.x, abs=1e-6)
        assert joints[name][1] == pytest.approx(point.y, abs=1e-6)
        assert joints[name][2] == pytest.approx(point.z, abs=1e-6)
    # 推理中也要给出关节角度，3D 卡片的角度格才有内容
    assert entries[0].angles["left_elbow"] > 0


def test_live_mapper_omits_joints_for_missing_detection() -> None:
    mapper = LiveSkeletonMapper()
    mapper.extend([PoseFrame(index=0, timestamp_ms=0.0, valid=False)])

    entry = mapper.snapshot()[0]
    assert len(mapper.joint_names) == 18
    assert entry.angles == {}
    assert all(visibility <= 0 for _, _, _, _, visibility in entry.as_joints(mapper.joint_names))
