"""手部 21 点时序稳定测试（掉帧补全 / 姿态迟滞 / 左右手配对 / 人物归属）。"""

from __future__ import annotations

import math

import pytest

from app.backend.engine.base import LANDMARK_COUNT, LEFT_WRIST, RIGHT_WRIST, Landmark, PoseFrame
from app.backend.engine.enhancements import (
    HAND_LANDMARK_COUNT,
    RefinementResult,
    apply_refinements,
)
from app.backend.postprocess.hands import (
    HandStabilityConfig,
    _pose_delta,
    stabilize_hands,
)
from app.backend.postprocess.ik_fit import FitConfig
from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess
from app.backend.postprocess.smoothing import SmoothingConfig
from app.backend.skeleton.mapping import map_frame

HAND_SCALE = 0.05

# 五个手指的展开方向（腕点为原点，y 向下）：拇指在前，其余四指朝下散开
_FINGER_DIRECTIONS: tuple[tuple[float, float], ...] = (
    (-0.9, -1.0),
    (-0.35, 1.0),
    (0.0, 1.05),
    (0.35, 1.0),
    (0.7, 0.9),
)
# 每根手指四个关节到腕点的距离（单位：手长）；curl=1 时折叠回掌心 = 握拳
_OPEN_DEPTHS = (1.0, 1.6, 2.1, 2.5)
_FIST_DEPTHS = (1.0, 1.25, 1.1, 0.9)


def make_hand(
    curl: float, wrist: tuple[float, float] = (0.3, 0.5), scale: float = HAND_SCALE
) -> list[Landmark]:
    """合成一只 21 点的手：``curl=0`` 张开、``curl=1`` 握拳。"""
    points = [Landmark(x=wrist[0], y=wrist[1], z=0.0, visibility=0.95)]
    for direction in _FINGER_DIRECTIONS:
        norm = math.hypot(*direction)
        ux, uy = direction[0] / norm, direction[1] / norm
        for open_depth, fist_depth in zip(_OPEN_DEPTHS, _FIST_DEPTHS, strict=False):
            depth = open_depth + (fist_depth - open_depth) * curl
            points.append(
                Landmark(
                    x=wrist[0] + ux * depth * scale,
                    y=wrist[1] + uy * depth * scale,
                    z=0.0,
                    visibility=0.95,
                )
            )
    assert len(points) == HAND_LANDMARK_COUNT
    return points


def make_frame(
    index: int,
    *,
    left_hand: list[Landmark] | None = None,
    right_hand: list[Landmark] | None = None,
    labels: tuple[str, str] = ("Left", "Right"),
    left_wrist: tuple[float, float] = (0.3, 0.5),
    right_wrist: tuple[float, float] = (0.7, 0.5),
) -> PoseFrame:
    landmarks = [
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
    ]
    landmarks[LEFT_WRIST] = Landmark(x=left_wrist[0], y=left_wrist[1], z=0.0, visibility=0.9)
    landmarks[RIGHT_WRIST] = Landmark(x=right_wrist[0], y=right_wrist[1], z=0.0, visibility=0.9)

    hands: list[list[Landmark]] = []
    sides: list[str] = []
    for hand, label in ((left_hand, labels[0]), (right_hand, labels[1])):
        if hand is not None:
            hands.append(hand)
            sides.append(label)
    return PoseFrame(
        index=index,
        timestamp_ms=index * 33.0,
        landmarks=landmarks,
        score=0.9,
        valid=True,
        hands=hands or None,
        hand_sides=sides,
    )


def hand_of(frame: PoseFrame, side: str = "left") -> list[Landmark] | None:
    for index, label in enumerate(frame.hand_sides):
        if label.lower() == side and index < len(frame.hands or []):
            return frame.hands[index]
    return None


# --------------------------------------------------------------------------- #
# 掉帧补全
# --------------------------------------------------------------------------- #
def test_missing_frame_reuses_fist_pose_instead_of_opening() -> None:
    """中间掉一帧：补出来的必须是握拳，而不是"没检测到 = 张开"。"""
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist) for index in range(6)]
    frames[3].hands = None
    frames[3].hand_sides = []

    report = stabilize_hands(frames, HandStabilityConfig(smooth=False))

    filled = hand_of(frames[3])
    assert filled is not None
    assert _pose_delta(filled, fist) < 1e-6
    assert report.applied is True
    assert report.side("left") is not None
    assert report.side("left").frames_interpolated == 1  # type: ignore[union-attr]


def test_held_hand_follows_current_body_wrist() -> None:
    """缺口超过插值窗口后沿用旧手型，但腕点要贴到当前帧的人体腕点（手跟着手臂走）。"""
    fist = make_hand(curl=1.0, wrist=(0.30, 0.5))
    frames = []
    for index in range(9):
        wrist_x = 0.30 + index * 0.01
        frames.append(
            make_frame(
                index,
                left_hand=fist if index <= 2 else None,
                left_wrist=(wrist_x, 0.5),
            )
        )

    report = stabilize_hands(
        frames, HandStabilityConfig(max_gap_frames=1, hold_max_frames=24, smooth=False)
    )

    held = hand_of(frames[6])
    assert held is not None
    body_wrist = frames[6].landmarks[LEFT_WRIST]
    assert held[0].x == pytest.approx(body_wrist.x, abs=1e-6)
    assert held[0].y == pytest.approx(body_wrist.y, abs=1e-6)
    # 手型本身不变
    assert _pose_delta(held, fist) < 1e-6
    assert report.side("left").frames_held >= 1  # type: ignore[union-attr]


def test_long_gap_is_left_missing_instead_of_fabricated() -> None:
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist if index <= 2 else None) for index in range(40)]

    stabilize_hands(
        frames, HandStabilityConfig(max_gap_frames=2, hold_max_frames=5, smooth=False)
    )

    assert hand_of(frames[2]) is not None
    assert frames[10].hands is None
    assert frames[39].hands is None


def test_leading_gap_is_not_backfilled() -> None:
    """片头没检测到手就不补：无法判断手是刚入画还是一直存在。"""
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist if index >= 2 else None) for index in range(8)]

    stabilize_hands(frames, HandStabilityConfig(smooth=False))

    assert frames[0].hands is None
    assert frames[1].hands is None
    assert hand_of(frames[2]) is not None


# --------------------------------------------------------------------------- #
# 姿态迟滞：单帧跳变不采纳，连续确认才切换
# --------------------------------------------------------------------------- #
def test_single_frame_open_hand_is_rejected() -> None:
    fist = make_hand(curl=1.0)
    open_hand = make_hand(curl=0.0)
    frames = [make_frame(index, left_hand=fist) for index in range(8)]
    frames[4].hands = [open_hand]
    frames[4].hand_sides = ["Left"]

    report = stabilize_hands(frames, HandStabilityConfig(smooth=False))

    kept = hand_of(frames[4])
    assert kept is not None
    assert _pose_delta(kept, fist) < 1e-6
    assert report.side("left").rejected_jumps == 1  # type: ignore[union-attr]


def test_sustained_pose_change_is_accepted_with_ramp() -> None:
    """连续两帧都张开 → 认可换手型，并且在确认窗口内过渡（不会一帧硬跳）。"""
    fist = make_hand(curl=1.0)
    open_hand = make_hand(curl=0.0)
    frames = []
    for index in range(13):
        frames.append(make_frame(index, left_hand=fist if index < 8 else open_hand))

    report = stabilize_hands(frames, HandStabilityConfig(smooth=False))

    assert _pose_delta(hand_of(frames[7]), fist) < 1e-6  # type: ignore[arg-type]
    accepted = hand_of(frames[9])
    assert accepted is not None and _pose_delta(accepted, open_hand) < 1e-6
    final = hand_of(frames[12])
    assert final is not None and _pose_delta(final, open_hand) < 1e-6
    # 第 8 帧是确认窗口内的过渡帧：既不完全是拳，也不完全是张开
    transitional = hand_of(frames[8])
    assert transitional is not None
    assert _pose_delta(transitional, fist) > 0.1
    assert _pose_delta(transitional, open_hand) > 0.1
    assert report.side("left").rejected_jumps == 0  # type: ignore[union-attr]


def test_confirm_frames_one_disables_hysteresis() -> None:
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist) for index in range(6)]
    frames[3].hands = [make_hand(curl=0.0)]
    frames[3].hand_sides = ["Left"]

    stabilize_hands(
        frames, HandStabilityConfig(smooth=False, confirm_frames=1, motion_threshold=0.2)
    )

    assert _pose_delta(hand_of(frames[3]), fist) > 0.5  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 左右手配对
# --------------------------------------------------------------------------- #
def test_handedness_flip_is_corrected_by_wrist_position() -> None:
    """handedness 逐帧翻转时，同一只手不能被认成左右两只不同的手。"""
    fist = make_hand(curl=1.0, wrist=(0.3, 0.5))
    frames = [
        make_frame(index, left_hand=fist, labels=("Left" if index % 2 == 0 else "Right", "Right"))
        for index in range(6)
    ]

    report = stabilize_hands(frames, HandStabilityConfig(smooth=False))

    for frame in frames:
        assert frame.hand_sides == ["Left"]
        assert hand_of(frame, "left") is not None
        assert hand_of(frame, "right") is None
    assert report.side("left").frames_observed == 6  # type: ignore[union-attr]


def test_two_hands_stay_on_their_own_sides() -> None:
    left = make_hand(curl=0.2, wrist=(0.3, 0.5))
    right = make_hand(curl=0.9, wrist=(0.7, 0.5))
    frames = [
        make_frame(index, left_hand=left, right_hand=right) for index in range(4)
    ]

    stabilize_hands(frames, HandStabilityConfig(smooth=False))

    for frame in frames:
        assert frame.hand_sides == ["Left", "Right"]
        assert _pose_delta(hand_of(frame, "left"), left) < 1e-6  # type: ignore[arg-type]
        assert _pose_delta(hand_of(frame, "right"), right) < 1e-6  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 人物归属（增强阶段）
# --------------------------------------------------------------------------- #
def _person(wrist_x: float, index: int = 0) -> PoseFrame:
    frame = make_frame(index, left_wrist=(wrist_x, 0.5), right_wrist=(wrist_x + 0.05, 0.5))
    frame.track_id = 1 if wrist_x < 0.5 else 2
    return frame


def test_refinements_attach_hand_to_nearest_person() -> None:
    """多人画面：手要挂到腕点最近的那个人，而不是永远 frames[0]。"""
    first, second = _person(0.2), _person(0.8)
    hand = make_hand(curl=1.0, wrist=(0.80, 0.5))
    result = RefinementResult(hands=[hand], hand_sides=["Left"])

    apply_refinements([first, second], result)

    assert first.hands is None
    assert second.hands is not None and len(second.hands) == 1
    assert second.hand_sides == ["Left"]


def test_refinements_drop_hand_far_from_every_person() -> None:
    first, second = _person(0.2), _person(0.8)
    stranger = make_hand(curl=1.0, wrist=(0.5, 0.1))
    result = RefinementResult(hands=[stranger], hand_sides=["Left"])

    apply_refinements([first, second], result)

    assert first.hands is None
    assert second.hands is None


def test_refinements_keep_hand_for_single_person_even_if_wrist_drifts() -> None:
    """只有一个人时保留兜底：姿态腕点抖动不能把整只手丢掉。"""
    person = _person(0.2)
    hand = make_hand(curl=1.0, wrist=(0.2, 0.35))
    result = RefinementResult(hands=[hand], hand_sides=["Left"])

    apply_refinements([person], result)

    assert person.hands is not None and len(person.hands) == 1


# --------------------------------------------------------------------------- #
# 与后处理流水线 / 导出的衔接
# --------------------------------------------------------------------------- #
def test_pipeline_reports_hand_stage_and_fills_finger_joints() -> None:
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist) for index in range(10)]
    frames[5].hands = None
    frames[5].hand_sides = []

    result = run_postprocess(
        frames,
        PostProcessConfig(
            smoothing=SmoothingConfig(method="none"),
            ik_fit=FitConfig(enabled=False),
        ),
    )

    assert result.hands.applied is True
    assert hand_of(result.frames[5]) is not None
    # 补出来的帧必须能映射出指关节，否则导出端还是会缺手指
    skeleton = map_frame(result.frames[5], include_all=True)
    assert "left_handIndexEnd_joint" in skeleton.joints
    assert "left_handThumbEnd_joint" in skeleton.joints


def test_hand_stability_can_be_disabled_for_legacy_output() -> None:
    fist = make_hand(curl=1.0)
    frames = [make_frame(index, left_hand=fist) for index in range(6)]
    frames[3].hands = None
    frames[3].hand_sides = []

    report = stabilize_hands(frames, HandStabilityConfig(enabled=False))

    assert report.applied is False
    assert frames[3].hands is None
