"""脚部着地锚定：深蹲时髋下沉、脚底固定的回归测试。"""

from __future__ import annotations

import math

import pytest

from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.skeleton.bundle import SkeletonConfig, build_skeleton_bundle
from app.backend.skeleton.ground import (
    GroundContactConfig,
    anchor_feet_to_ground,
)
from app.backend.skeleton.transform import ConvertedFrame, JointPosition

# 髋为原点的米制 world landmarks（MediaPipe：y 向下），脚踝在髋下方 0.84 m。
STANDING_WORLD: dict[int, tuple[float, float, float]] = {
    0: (0.00, -0.70, 0.0),
    7: (0.07, -0.74, 0.0),
    8: (-0.07, -0.74, 0.0),
    11: (0.19, -0.45, 0.0),
    12: (-0.19, -0.45, 0.0),
    23: (0.09, 0.00, 0.0),
    24: (-0.09, 0.00, 0.0),
    25: (0.09, 0.42, 0.0),
    26: (-0.09, 0.42, 0.0),
    27: (0.09, 0.84, 0.0),
    28: (-0.09, 0.84, 0.0),
    29: (0.09, 0.88, 0.06),
    30: (-0.09, 0.88, 0.06),
    31: (0.09, 0.90, -0.04),
    32: (-0.09, 0.90, -0.04),
}

# 深蹲：MediaPipe 的髋原点不变，膝/踝向髋靠近（这正是"脚离地"的来源）。
SQUAT_Y: dict[int, float] = {
    25: 0.20,
    26: 0.20,
    27: 0.44,
    28: 0.44,
    29: 0.47,
    30: 0.47,
    31: 0.48,
    32: 0.48,
}


def _make_pose_frame(index: int, *, squat: bool = False) -> PoseFrame:
    points = dict(STANDING_WORLD)
    if squat:
        for landmark_index, y in SQUAT_Y.items():
            x, _, z = points[landmark_index]
            points[landmark_index] = (x, y, z)

    world = [Landmark() for _ in range(LANDMARK_COUNT)]
    normalized = [Landmark() for _ in range(LANDMARK_COUNT)]
    for landmark_index, (x, y, z) in points.items():
        world[landmark_index] = Landmark(x=x, y=y, z=z, visibility=0.95)
        normalized[landmark_index] = Landmark(
            x=0.5 + x * 0.5, y=0.55 + y * 0.45, z=0.0, visibility=0.95
        )
    return PoseFrame(
        index=index,
        timestamp_ms=index * 100.0,
        landmarks=normalized,
        world_landmarks=world,
        score=0.95,
    )


def _converted(index: int, foot_y: float) -> ConvertedFrame:
    return ConvertedFrame(
        index=index,
        t=index * 0.1,
        joints={
            "left_foot_joint": JointPosition(y=foot_y, visibility=0.95),
            "right_foot_joint": JointPosition(y=foot_y, visibility=0.95),
            "left_toes_joint": JointPosition(y=foot_y - 0.04, visibility=0.95),
        },
    )


def test_anchor_feet_plants_ankle_and_lowers_hip() -> None:
    frames = [_converted(0, -0.90), _converted(1, -0.90), _converted(2, -0.50)]
    anchored, references = anchor_feet_to_ground(frames)

    assert len(references) == 1
    assert references[0].foot_span_meters == pytest.approx(0.94)
    assert anchored[0].root.y == pytest.approx(0.0, abs=1e-4)
    # 深蹲帧髋部下沉约 0.4 m，而不是脚踝上抬。
    assert anchored[2].root.y == pytest.approx(-0.40, abs=1e-3)

    ankle_heights = [frame.root.y + frame.joints["left_foot_joint"].y for frame in anchored]
    assert max(ankle_heights) - min(ankle_heights) < 1e-4
    # 关节本身保持髋局部坐标，只有 root 被修正。
    assert anchored[2].joints["left_foot_joint"].y == pytest.approx(-0.50)


def test_anchor_feet_can_be_disabled() -> None:
    frames = [_converted(0, -0.90), _converted(1, -0.50)]
    anchored, references = anchor_feet_to_ground(
        frames, GroundContactConfig(enabled=False)
    )
    assert references == []
    assert all(frame.root.length == pytest.approx(0.0) for frame in anchored)


def test_anchor_feet_ignores_frames_without_feet() -> None:
    frames = [ConvertedFrame(index=0), ConvertedFrame(index=1, valid=False)]
    anchored, references = anchor_feet_to_ground(frames)
    assert references == []
    assert all(frame.root.length == pytest.approx(0.0) for frame in anchored)


def _converted_pair(index: int, left_span: float, right_span: float) -> ConvertedFrame:
    """双脚各给一条踝点；span = 髋到脚的竖直距离（米）。"""
    return ConvertedFrame(
        index=index,
        t=index / 30.0,
        joints={
            "left_foot_joint": JointPosition(y=-left_span, visibility=0.95),
            "right_foot_joint": JointPosition(y=-right_span, visibility=0.95),
        },
    )


def test_anchor_suppresses_span_jitter_without_lagging_squat() -> None:
    """原地动作：脚没动，但逐帧跨度噪声曾原样变成整段髋部上下抖动。"""
    frames = [
        _converted_pair(index, 0.90 + (0.02 if index % 2 else -0.02), 0.90 + (0.02 if index % 2 else -0.02))
        for index in range(31)
    ]
    anchored, references = anchor_feet_to_ground(frames)
    heights = [frame.root.y for frame in anchored]

    # 平滑前 ±2 cm 的逐帧抖动，平滑后应基本消失。
    assert max(heights) - min(heights) < 0.005
    assert references[0].span_jitter_after_meters < references[0].span_jitter_before_meters * 0.5


def test_support_foot_hysteresis_prevents_root_bob() -> None:
    """两脚差不多高时，支撑脚不能在噪声里来回切换（否则 root.y 会阶跃）。"""
    frames = [
        _converted_pair(index, 0.94, 0.94 if index % 2 == 0 else 0.955)
        for index in range(21)
    ]
    anchored, _ = anchor_feet_to_ground(frames)
    heights = [frame.root.y for frame in anchored]
    assert max(heights) - min(heights) < 1e-3

    # 关掉滞回与平滑时，同一段输入必然抖起来 —— 证明上一条不是恒等断言。
    loose = GroundContactConfig(switch_margin_meters=0.0, smooth_cutoff_hz=0.0)
    raw, _ = anchor_feet_to_ground(frames, loose)
    raw_heights = [frame.root.y for frame in raw]
    assert max(raw_heights) - min(raw_heights) > 0.005


def test_anchor_reports_smoothed_ground_span_for_preview() -> None:
    anchored, _ = anchor_feet_to_ground([_converted(0, -0.90), _converted(1, -0.90)])
    assert anchored[0].ground_span == pytest.approx(0.94, abs=1e-5)

    skipped, _ = anchor_feet_to_ground([_converted(0, -0.90), ConvertedFrame(index=1, valid=False)])
    assert skipped[1].ground_span is None


def test_anchor_holds_span_when_foot_temporarily_hidden() -> None:
    """脚被遮挡一帧时沿用邻居的跨度，而不是把 root.y 弹回 0（整个身体蹦一下）。"""
    hidden = ConvertedFrame(index=3, t=3 / 30.0, joints={})
    frames = [
        hidden if frame.index == 3 else frame
        for frame in (_converted_pair(index, 0.60, 0.60) for index in range(7))
    ]
    anchored, _ = anchor_feet_to_ground(frames)
    heights = [frame.root.y for frame in anchored]
    assert max(heights) - min(heights) < 1e-3


def _distance(a: JointPosition, b: JointPosition) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _converted_leg(index: int, jitter: float) -> ConvertedFrame:
    """一条微屈的腿：髋 → 膝（略向前）→ 踝 → 脚尖，竖直方向带 jitter。"""
    return ConvertedFrame(
        index=index,
        t=index / 30.0,
        joints={
            "left_upLeg_joint": JointPosition(x=0.09, y=0.0, visibility=0.95),
            "left_leg_joint": JointPosition(x=0.09, y=-0.42, z=0.12, visibility=0.95),
            "left_foot_joint": JointPosition(x=0.09, y=-0.80 + jitter, z=0.02, visibility=0.95),
            "left_toes_joint": JointPosition(
                x=0.09, y=-0.84 + jitter, z=0.08, visibility=0.95
            ),
        },
    )


def test_lock_support_foot_keeps_contact_height_constant() -> None:
    """只调 root.y 时脚自身仍随噪声上下跳；锁定后脚底高度应恒定，骨长不变。"""
    frames = [_converted_leg(index, 0.02 if index % 2 else -0.02) for index in range(31)]
    anchored, _ = anchor_feet_to_ground(frames)

    contact_heights = [
        frame.root.y + frame.joints["left_toes_joint"].y for frame in anchored
    ]
    assert max(contact_heights) - min(contact_heights) < 5e-4

    for before, after in zip(frames, anchored, strict=True):
        thigh_before = _distance(before.joints["left_upLeg_joint"], before.joints["left_leg_joint"])
        thigh_after = _distance(after.joints["left_upLeg_joint"], after.joints["left_leg_joint"])
        assert thigh_after == pytest.approx(thigh_before, abs=1e-6)
        shank_before = _distance(before.joints["left_leg_joint"], before.joints["left_foot_joint"])
        shank_after = _distance(after.joints["left_leg_joint"], after.joints["left_foot_joint"])
        assert shank_after == pytest.approx(shank_before, abs=1e-6)

    unlocked, _ = anchor_feet_to_ground(frames, GroundContactConfig(lock_support_foot=False))
    unlocked_heights = [
        frame.root.y + frame.joints["left_toes_joint"].y for frame in unlocked
    ]
    assert max(unlocked_heights) - min(unlocked_heights) > 0.005


def test_bundle_squat_restores_hip_descent_and_plants_feet() -> None:
    frames = [_make_pose_frame(0), _make_pose_frame(1), _make_pose_frame(2, squat=True)]
    bundle = build_skeleton_bundle(frames, SkeletonConfig(include_optional=True))

    standing, squat = bundle.frames[0], bundle.frames[2]
    assert squat.root.y < standing.root.y - 0.3

    def foot_height(frame: ConvertedFrame, name: str) -> float:
        return frame.root.y + frame.joints[name].y

    # 真正接触地面的是脚尖；脚踝比脚尖高约 6 cm 属正常解剖结构。
    assert foot_height(squat, "left_toes_joint") == pytest.approx(
        foot_height(standing, "left_toes_joint"), abs=0.01
    )
    assert abs(foot_height(squat, "left_foot_joint") - foot_height(standing, "left_foot_joint")) < 0.07
    translations = [item.translation.y for item in bundle.root_motion.frames]
    assert min(translations) < -0.3
    assert any("着地锚定" in note for note in bundle.notes)
    assert any("水平脚底" in note for note in bundle.notes)


# --------------------------------------------------------------------------- #
# 水平脚底锚定：原地站立时脚不能跟着髋在地面上漂
# --------------------------------------------------------------------------- #


def _standing_pair(
    index: int,
    sway: float,
    left_jitter: float = 0.0,
    right_jitter: float = 0.0,
    lift: float = 0.0,
) -> ConvertedFrame:
    """原地站立（髋为原点）：身体左晃 sway 时，脚在髋局部坐标里反向移动 -sway。

    ``lift`` 抬起左脚（span 变小），用来验证单脚离地不会把基准点拽走。
    """
    return ConvertedFrame(
        index=index,
        t=index / 30.0,
        joints={
            "left_foot_joint": JointPosition(
                x=-0.10 - sway + left_jitter,
                y=-0.85 + lift,
                z=0.02 - sway,
                visibility=0.95,
            ),
            "right_foot_joint": JointPosition(
                x=0.10 - sway + right_jitter,
                y=-0.85,
                z=0.02 - sway,
                visibility=0.95,
            ),
        },
    )


def _sway_frames(count: int = 61) -> list[ConvertedFrame]:
    frames: list[ConvertedFrame] = []
    for index in range(count):
        seconds = index / 30.0
        sway = 0.03 * math.sin(2.0 * math.pi * 0.4 * seconds)
        jitter = 0.008 if index % 2 else -0.008
        frames.append(_standing_pair(index, sway, jitter, -jitter))
    return frames


def test_horizontal_anchor_keeps_standing_feet_from_drifting() -> None:
    """髋局部坐标下脚会跟着身体晃；水平锚定后脚在世界里基本不动，髋去晃。"""
    frames = _sway_frames()
    anchored, references = anchor_feet_to_ground(frames)

    raw_foot_x = [frame.joints["left_foot_joint"].x for frame in frames]
    foot_x = [frame.root.x + frame.joints["left_foot_joint"].x for frame in anchored]
    # 锚定前左右各晃 ±3 cm（叠 ±0.8 cm 噪声）；锚定后脚只剩被低通滤掉的噪声。
    assert max(raw_foot_x) - min(raw_foot_x) > 0.06
    assert max(foot_x) - min(foot_x) < 0.02
    # 髋必须真的在动，否则说明只是把整段位移清零了。
    root_x = [frame.root.x for frame in anchored]
    assert max(root_x) - min(root_x) > 0.05

    reference = references[0]
    assert reference.horizontal_locked
    assert (
        reference.horizontal_drift_after_meters
        < reference.horizontal_drift_before_meters * 0.6
    )


def test_horizontal_anchor_can_be_disabled() -> None:
    frames = _sway_frames(21)
    anchored, references = anchor_feet_to_ground(
        frames, GroundContactConfig(anchor_horizontal=False)
    )
    assert all(frame.root.x == 0.0 for frame in anchored)
    assert not references[0].horizontal_locked


def test_horizontal_anchor_ignores_lifted_foot() -> None:
    """抬起一只脚时，基准点应跟着支撑脚走，而不是被抬起那只脚拽偏半个站距。"""
    frames = [_standing_pair(index, 0.0, lift=0.25 if index >= 10 else 0.0) for index in range(31)]
    anchored, _ = anchor_feet_to_ground(frames)
    before = anchored[9].root.x
    after = anchored[-1].root.x
    assert abs(after - before) < 0.03


def test_horizontal_anchor_skipped_when_only_one_foot_visible() -> None:
    frames = []
    for index in range(21):
        frame = _standing_pair(index, 0.0)
        frame.joints.pop("right_foot_joint")
        frames.append(frame)
    anchored, references = anchor_feet_to_ground(frames)
    assert all(frame.root.x == 0.0 for frame in anchored)
    assert not references[0].horizontal_locked


def test_bundle_reports_horizontal_anchor_in_notes() -> None:
    bundle = build_skeleton_bundle(
        [_make_pose_frame(index) for index in range(5)], SkeletonConfig(include_optional=True)
    )
    assert any("水平脚底" in note for note in bundle.notes)
    assert all(item.translation.x == pytest.approx(0.0, abs=1e-6) for item in bundle.root_motion.frames)
