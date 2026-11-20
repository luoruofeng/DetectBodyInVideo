"""肢体互相遮挡时的姿态连续性测试。"""

from __future__ import annotations

import math

import numpy as np
import synth_pose

from app.backend.engine.base import LEFT_ELBOW, LEFT_SHOULDER
from app.backend.postprocess.angles import AngleConfig, attach_angles
from app.backend.postprocess.ik_fit import FitConfig, fit_frames


def _direction(frame, vertex: int) -> np.ndarray:
    points = frame.world_landmarks
    vector = np.array(
        [points[vertex].x - points[LEFT_SHOULDER].x,
         points[vertex].y - points[LEFT_SHOULDER].y,
         points[vertex].z - points[LEFT_SHOULDER].z],
        dtype=np.float64,
    )
    return vector / np.linalg.norm(vector)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def _turn_matrix(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array([[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]])


def _occlusion_frames() -> list:
    before = synth_pose.make_frames(3, degrees=0.0)
    turned = synth_pose.make_frames(5, degrees=35.0)
    for index, frame in enumerate(turned, start=3):
        frame.index = index
        frame.timestamp_ms = index * 1000.0 / 30.0
    frames = before + turned
    for frame in frames[3:6]:
        for points in (frame.landmarks, frame.world_landmarks):
            point = points[LEFT_ELBOW]
            point.visibility = 0.0
            point.x += 0.9
            point.y -= 0.4
            point.z += 0.6
    return frames


def test_occluded_limb_keeps_previous_relative_angle():
    frames = _occlusion_frames()
    config = FitConfig(refine_passes=1, temporal_weight=0.0)
    fitted, report = fit_frames(frames, config)

    assert report.frames_occlusion_held >= 3
    reference = _direction(fitted[2], LEFT_ELBOW)
    expected = _turn_matrix(35.0) @ reference
    for frame in fitted[3:6]:
        assert _angle_between(_direction(frame, LEFT_ELBOW), expected) < math.radians(12)


def test_without_occlusion_hold_bad_detection_moves_limb():
    frames = _occlusion_frames()
    config = FitConfig(
        refine_passes=1, temporal_weight=0.0, hold_occluded_limbs=False
    )
    fitted, report = fit_frames(frames, config)

    assert report.frames_occlusion_held == 0
    reference = _direction(fitted[2], LEFT_ELBOW)
    expected = _turn_matrix(35.0) @ reference
    assert _angle_between(_direction(fitted[3], LEFT_ELBOW), expected) > math.radians(30)


def test_angle_series_holds_short_occlusion_then_releases():
    frames = synth_pose.make_frames(10)
    for frame in frames[2:5]:
        frame.landmarks[LEFT_ELBOW].visibility = 0.0
        frame.world_landmarks[LEFT_ELBOW].visibility = 0.0

    raw = attach_angles(frames, AngleConfig(hold_occluded_frames=0))
    held = attach_angles(frames, AngleConfig(hold_occluded_frames=4))
    assert "left_elbow" in raw[1]
    assert all("left_elbow" not in raw[index] for index in (2, 3, 4))
    assert all(
        held[index]["left_elbow"] == raw[1]["left_elbow"] for index in (2, 3, 4)
    )
    assert "left_elbow" in held[5]  # 恢复观测后立即采用新读数
