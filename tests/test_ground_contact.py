"""脚部着地锚定：深蹲时髋下沉、脚底固定的回归测试。"""

from __future__ import annotations

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
