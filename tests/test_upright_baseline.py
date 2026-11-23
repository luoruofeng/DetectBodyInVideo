"""有符号躯干俯仰角与直立基准校准的单测。"""

from __future__ import annotations

import pytest

from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Landmark,
    PoseFrame,
)
from app.backend.export.schema import ExportFrame
from app.backend.postprocess.angles import compute_angles
from app.backend.postprocess.upright import (
    UprightBaselineConfig,
    UprightBaselineReport,
    calibrate_upright_baseline,
)
from app.backend.skeleton.orientation import YawEstimate


def make_world_frame(
    index: int,
    *,
    shoulder_z: float,
    hip_z: float = 0.0,
    hip_x: float = 0.5,
    ear_z: float = 0.0,
    ear_visibility: float = 0.9,
) -> PoseFrame:
    frame = PoseFrame(
        index=index,
        timestamp_ms=index * 33.0,
        landmarks=[
            Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
        ],
        score=0.9,
    )
    world = [
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
    ]
    for landmark_index in (LEFT_SHOULDER, RIGHT_SHOULDER):
        world[landmark_index] = Landmark(x=0.5, y=0.35, z=shoulder_z, visibility=0.9)
    for landmark_index in (LEFT_HIP, RIGHT_HIP):
        world[landmark_index] = Landmark(x=hip_x, y=0.75, z=hip_z, visibility=0.9)
    for landmark_index in (LEFT_HIP, RIGHT_HIP):
        frame.landmarks[landmark_index] = Landmark(
            x=hip_x, y=0.75, z=0.0, visibility=0.9
        )
    # 两耳：头部关键点，虽不参与拟合，但属于上半身，必须跟着躯干基准一起平移
    for landmark_index, x in ((LEFT_EAR, 0.47), (RIGHT_EAR, 0.53)):
        world[landmark_index] = Landmark(
            x=x, y=0.18, z=ear_z, visibility=ear_visibility
        )
    frame.world_landmarks = world
    return frame


def frontal_yaws(count: int, *, width_ratio: float = 0.97) -> list[YawEstimate]:
    return [
        YawEstimate(index=index, yaw_rad=0.0, confidence=0.4, source="both", width_ratio=width_ratio)
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# 有符号俯仰角
# --------------------------------------------------------------------------- #
def test_signed_pitch_is_positive_forward_negative_backward() -> None:
    forward = compute_angles(make_world_frame(0, shoulder_z=-0.10))[
        "torso_pitch_signed_deg"
    ]
    backward = compute_angles(make_world_frame(1, shoulder_z=0.10))[
        "torso_pitch_signed_deg"
    ]
    upright = compute_angles(make_world_frame(2, shoulder_z=0.0))[
        "torso_pitch_signed_deg"
    ]
    assert forward > 0 and backward < 0
    assert forward == pytest.approx(-backward, abs=1e-6)
    assert upright == pytest.approx(0.0, abs=1e-6)


def test_export_allows_negative_signed_pitch() -> None:
    frame = ExportFrame(index=0, t=0.0, angles={"torso_pitch_signed_deg": -12.3})
    assert frame.angles["torso_pitch_signed_deg"] == -12.3
    with pytest.raises(ValueError):
        ExportFrame(index=0, t=0.0, angles={"left_elbow": -5.0})


# --------------------------------------------------------------------------- #
# 直立基准校准
# --------------------------------------------------------------------------- #
def test_calibration_removes_forward_bias_from_upper_body_only() -> None:
    frames = [make_world_frame(index, shoulder_z=-0.12) for index in range(20)]
    report = calibrate_upright_baseline(frames, frontal_yaws(20))

    assert report.applied is True
    assert report.eligible_frames == 20
    assert report.baseline_meters == pytest.approx(-0.12, abs=1e-6)
    assert report.adjusted_frames == 20
    for frame in frames:
        assert frame.world_landmarks[LEFT_SHOULDER].z == pytest.approx(0.0, abs=1e-6)
        assert frame.world_landmarks[RIGHT_SHOULDER].z == pytest.approx(0.0, abs=1e-6)
        assert frame.world_landmarks[LEFT_HIP].z == pytest.approx(0.0, abs=1e-6)
        assert frame.world_landmarks[RIGHT_HIP].z == pytest.approx(0.0, abs=1e-6)


def test_calibration_skips_when_view_is_not_frontal() -> None:
    frames = [make_world_frame(index, shoulder_z=-0.12) for index in range(20)]
    report = calibrate_upright_baseline(frames, frontal_yaws(20, width_ratio=0.5))

    assert report.applied is False
    assert report.eligible_frames == 0
    assert frames[0].world_landmarks[LEFT_SHOULDER].z == pytest.approx(-0.12)
    assert isinstance(report, UprightBaselineReport)


def test_calibration_skips_when_subject_is_moving() -> None:
    frames = [make_world_frame(index, shoulder_z=-0.12, hip_x=0.5 + index * 0.05) for index in range(20)]
    report = calibrate_upright_baseline(frames, frontal_yaws(20))

    assert report.applied is False
    assert report.eligible_frames < UprightBaselineConfig().min_frames


def test_calibration_skips_implausibly_large_bias() -> None:
    frames = [make_world_frame(index, shoulder_z=-0.60) for index in range(20)]
    report = calibrate_upright_baseline(frames, frontal_yaws(20))

    assert report.applied is False
    assert "超过" in "".join(report.notes)
    assert frames[0].world_landmarks[LEFT_SHOULDER].z == pytest.approx(-0.60)


def test_calibration_can_be_disabled() -> None:
    frames = [make_world_frame(index, shoulder_z=-0.12) for index in range(20)]
    report = calibrate_upright_baseline(
        frames, frontal_yaws(20), UprightBaselineConfig(enabled=False)
    )

    assert report.applied is False
    assert frames[0].world_landmarks[LEFT_SHOULDER].z == pytest.approx(-0.12)


def test_calibration_moves_ears_with_the_torso() -> None:
    """两耳不在骨长拓扑里，但属于上半身：躯干基准必须把它们的 z 一起平移。

    否则胸/肩/鼻都被挪回去了、耳却留在原地，任何消费 world 点的下游都会看到
    "头相对胸"凭空多出一段相对位移。
    """
    frames = [make_world_frame(index, shoulder_z=-0.12, ear_z=-0.12) for index in range(20)]
    report = calibrate_upright_baseline(frames, frontal_yaws(20))

    assert report.applied is True
    for frame in frames:
        assert frame.world_landmarks[LEFT_EAR].z == pytest.approx(0.0, abs=1e-6)
        assert frame.world_landmarks[RIGHT_EAR].z == pytest.approx(0.0, abs=1e-6)

    # 头相对躯干的"前探量"在这一步里既不产生、也不消除（那是 mapping/ik_fit 的职责）
    assert report.baseline_meters == pytest.approx(-0.12, abs=1e-6)
