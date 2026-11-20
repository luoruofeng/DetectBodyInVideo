"""AppleModel.usda 全关节覆盖测试（91 关节 + 手/脸增强映射）。"""

from __future__ import annotations

import re
from pathlib import Path

from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.engine.enhancements import FACE_SUBSET_NAMES, HAND_LANDMARK_COUNT
from app.backend.skeleton.bundle import SkeletonConfig, build_skeleton_bundle
from app.backend.skeleton.mapping import (
    APPLE_ALL_JOINTS,
    HAND_JOINT_SOURCES,
    joint_names,
    map_frame,
    validate_mapping,
)

USDA_PATH = Path(__file__).resolve().parents[1] / "resources" / "AppleModel.usda"


def _frame_with_enhancements() -> PoseFrame:
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
    for index in range(LANDMARK_COUNT):
        landmarks[index] = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)
    landmarks[15] = Landmark(x=0.33, y=0.60, z=0.0, visibility=0.9)  # left wrist
    landmarks[16] = Landmark(x=0.67, y=0.60, z=0.0, visibility=0.9)  # right wrist
    frame = PoseFrame(index=0, landmarks=landmarks, valid=True, score=0.9)
    frame.hands = [
        [Landmark(x=0.33 + i * 0.001, y=0.60, z=0.0, visibility=0.9) for i in range(HAND_LANDMARK_COUNT)],
        [Landmark(x=0.67 + i * 0.001, y=0.60, z=0.0, visibility=0.9) for i in range(HAND_LANDMARK_COUNT)],
    ]
    frame.hand_sides = ["Left", "Right"]
    frame.face = [Landmark(x=0.5, y=0.3, z=0.0, visibility=0.9) for _ in FACE_SUBSET_NAMES]
    return frame


def test_apple_joint_list_matches_usda() -> None:
    assert validate_mapping() == []
    assert len(APPLE_ALL_JOINTS) == 91
    if not USDA_PATH.is_file():
        return
    line = next(
        line for line in USDA_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        if "uniform token[] joints" in line
    )
    usda = tuple(part.split("/")[-1] for part in re.findall(r'"([^"]+)"', line))
    assert usda == APPLE_ALL_JOINTS


def test_joint_names_modes() -> None:
    assert len(joint_names()) == 18
    assert len(joint_names(include_optional=True)) == 22
    assert len(joint_names(include_all=True)) == 91


def test_map_frame_all_joints_with_enhancements() -> None:
    frame = _frame_with_enhancements()
    skeleton = map_frame(frame, include_all=True)
    assert set(skeleton.joints) == set(APPLE_ALL_JOINTS)
    assert skeleton.missing == []
    assert skeleton.joints["left_handIndex_1_joint"].source == "direct"
    assert skeleton.joints["right_handThumbEnd_joint"].source == "direct"
    assert skeleton.joints["left_eyeball_joint"].source in {"direct", "estimated"}
    assert skeleton.joints["spine_4_joint"].source == "derived"


def test_handedness_assigns_left_and_right() -> None:
    frame = _frame_with_enhancements()
    frame.hands[0] = [Landmark(x=0.10, y=0.10, z=0.0, visibility=0.9) for _ in range(HAND_LANDMARK_COUNT)]
    skeleton = map_frame(frame, include_all=True)
    # 左手数据落在画面左上，但 handedness 声明为 Left，必须按声明分配
    left = skeleton.joints["left_hand_joint"]
    right = skeleton.joints["right_hand_joint"]
    assert left.x != right.x


def test_world_mode_enhancement_points_are_anchored() -> None:
    frame = _frame_with_enhancements()
    # 归一化空间也要有头-髋跨度，否则 _local_world_scale 无法估计尺度
    frame.landmarks[0] = Landmark(x=0.5, y=0.18, z=0.0, visibility=0.9)  # nose
    frame.landmarks[23] = Landmark(x=0.45, y=0.55, z=0.0, visibility=0.9)  # left hip
    frame.landmarks[24] = Landmark(x=0.55, y=0.55, z=0.0, visibility=0.9)  # right hip
    # 构造有真实头-髋跨度的 world landmarks（米制，y 向下），否则无法估计尺度
    frame.world_landmarks = [
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
    ]
    frame.world_landmarks[0] = Landmark(x=0.5, y=-0.7, z=0.0, visibility=0.9)  # nose
    frame.world_landmarks[23] = Landmark(x=0.45, y=0.0, z=0.0, visibility=0.9)  # left hip
    frame.world_landmarks[24] = Landmark(x=0.55, y=0.0, z=0.0, visibility=0.9)  # right hip
    skeleton = map_frame(frame, use_world=True, include_all=True)
    assert len(skeleton.joints) == 91
    assert skeleton.joints["left_handIndex_1_joint"].source == "estimated"
    # 以 world 腕点为锚点换算，值应围绕 1.7 m 量级，而不是 [0,1] 归一化原值
    assert abs(skeleton.joints["left_handIndex_1_joint"].x) < 2.0


def test_world_mode_without_scale_skips_enhancement() -> None:
    frame = _frame_with_enhancements()
    frame.world_landmarks = [
        Landmark(x=0.5, y=0.0, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
    ]
    skeleton = map_frame(frame, use_world=True, include_all=True)
    # 无跨度无法估计尺度：增强关节必须记入 missing，绝不拿归一化值冒充米制
    assert "left_handIndex_1_joint" in skeleton.missing
    assert "left_eyeball_joint" in skeleton.missing


def test_bundle_all_joints_and_export_names() -> None:
    frame = _frame_with_enhancements()
    # 3D 关节只来自世界坐标，必须给出头-髋跨度才能把手/面增强点锚定到米制空间
    frame.landmarks[0] = Landmark(x=0.5, y=0.18, z=0.0, visibility=0.9)
    frame.landmarks[23] = Landmark(x=0.45, y=0.55, z=0.0, visibility=0.9)
    frame.landmarks[24] = Landmark(x=0.55, y=0.55, z=0.0, visibility=0.9)
    frame.world_landmarks = [
        Landmark(x=0.5, y=0.0, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
    ]
    frame.world_landmarks[0] = Landmark(x=0.5, y=-0.7, z=0.0, visibility=0.9)
    bundle = build_skeleton_bundle([frame], SkeletonConfig(include_all_joints=True))
    assert bundle.joint_names == list(APPLE_ALL_JOINTS)
    assert len(bundle.frames[0].joints) == 91
    assert "left_handPinky_2_joint" in bundle.frames[0].joints


def test_hand_joint_sources_cover_all_finger_joints() -> None:
    assert len(HAND_JOINT_SOURCES) == 48
    for name in HAND_JOINT_SOURCES:
        assert name in APPLE_ALL_JOINTS
        assert name.startswith("left_hand") or name.startswith("right_hand")
