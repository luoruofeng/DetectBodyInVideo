"""摆动腿识别与踢腿相位（``docs/todo-kick-depth-accuracy.md`` §5.2 / §5.3 / §5.4）。

三条硬约束：

1. **摆动腿能分清**：单腿前踢 → 判出踢的那条；双腿同步 → 如实退回 ``none``（不猜）；
2. **相位误差 ≤3 帧**：合成三角速度剖面下，起脚 / 最大伸展 / 回落的帧差都在 3 帧内；
3. **权重只影响先验力度**：相位与摆动腿只缩放置信度，不改变任何坐标。
"""

from __future__ import annotations

import math

import pytest
import synth_pose

from app.backend.postprocess.depth_sign import (
    SUPPORT_LEG_CONFIDENCE_SCALE,
    _sign_confidence,
)
from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess
from app.backend.postprocess.swing import (
    KickPhaseReport,
    SwingConfig,
    detect_kick_phases,
    direction_weight_scale,
    identify_swing_leg,
    sagittal_angles,
)

ASPECT = synth_pose.ASPECT


# --------------------------------------------------------------------------- #
# 1. 摆动腿识别
# --------------------------------------------------------------------------- #
def test_single_leg_kick_identifies_swing_leg():
    frames = synth_pose.make_kick_frames(40, peak_deg=70.0, ramp=0.45, side="left")
    report = identify_swing_leg(frames)
    assert report.swing_leg == "left"
    assert report.applied is True
    assert report.confidence > 0.5
    assert report.energy["left"] > report.energy["right"]


def test_symmetric_kick_falls_back_to_none():
    """双腿同步前踢：两侧能量接近，任何单侧判定都会制造单侧伪影。"""
    frames = synth_pose.make_kick_frames(40, peak_deg=70.0, ramp=0.45, side="both")
    report = identify_swing_leg(frames)
    assert report.swing_leg == "none"
    assert report.applied is False
    assert any("对称" in note or "交替" in note for note in report.notes), report.notes


def test_static_sequence_reports_no_motion():
    frames = synth_pose.make_frames(40)
    report = identify_swing_leg(frames)
    assert report.swing_leg == "none"
    assert report.applied is False


def test_short_sequence_is_rejected():
    frames = synth_pose.make_kick_frames(5, peak_deg=60.0)
    report = identify_swing_leg(frames, SwingConfig(min_frames=8))
    assert report.swing_leg == "none"
    assert any("帧数不足" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# 2. 相位检测
# --------------------------------------------------------------------------- #
def test_phase_error_within_three_frames():
    frames = synth_pose.make_kick_frames(60, peak_deg=80.0, ramp=0.5)
    labels = synth_pose.kick_labels(frames)
    truth = labels["phases"]
    report = detect_kick_phases(frames)
    assert report.detected is True
    assert report.side == "left"
    for key in ("start", "peak", "return"):
        assert abs(report.phases[key] - truth[key]) <= 3, (key, report.phases, truth)


def test_pipeline_phase_error_within_three_frames():
    """整条后处理链给出的相位也要 ≤3 帧。

    这条比"在原始帧上调检测器"更接近真实用法：平滑会把动作起点抹开 4~5 帧，
    所以链路必须用**平滑前**的 2D 轨迹定相位（见 ``pipeline.py`` 第 2.5 步注释）。
    """
    frames = synth_pose.make_kick_frames(60, peak_deg=80.0, ramp=0.4)
    labels = synth_pose.kick_labels(frames)
    base = PostProcessConfig(fps=30.0, aspect_ratio=ASPECT)
    config = base.model_copy(
        update={
            "depth_sign": base.depth_sign.model_copy(
                update={"enabled": True, "mode": "temporal"}
            )
        }
    )
    result = run_postprocess(frames, config)
    assert result.kick_phases.detected is True
    for key in ("start", "peak", "return"):
        assert abs(result.kick_phases.phases[key] - labels["phases"][key]) <= 3, (
            key,
            result.kick_phases.phases,
            labels["phases"],
        )


def test_phase_detection_fails_gracefully_on_static_input():
    frames = synth_pose.make_frames(40)
    report = detect_kick_phases(frames, "left")
    assert report.detected is False
    assert report.notes


def test_direction_weight_scale_follows_phases():
    phases = KickPhaseReport(
        applied=True, detected=True, side="left",
        phases={"start": 10, "acceleration": 16, "peak": 24, "return": 38},
    )
    assert direction_weight_scale(24, phases) == pytest.approx(1.5)
    assert direction_weight_scale(20, phases) == pytest.approx(1.5)  # 峰值 ±25% 内
    assert direction_weight_scale(2, phases) == pytest.approx(0.6)  # 起脚之前
    assert direction_weight_scale(34, phases) == pytest.approx(1.0)
    # 没检测到相位时不能改变既有行为
    assert direction_weight_scale(24, KickPhaseReport()) == 1.0


# --------------------------------------------------------------------------- #
# 3. 权重只影响先验力度
# --------------------------------------------------------------------------- #
def test_support_leg_confidence_is_discounted():
    confidence = 0.8
    swing = _sign_confidence("left_ankle", 24, confidence, "left", None)
    support = _sign_confidence("right_ankle", 24, confidence, "left", None)
    assert swing == pytest.approx(confidence)
    assert support == pytest.approx(confidence * SUPPORT_LEG_CONFIDENCE_SCALE)
    # 手臂与躯干不受摆动腿影响
    assert _sign_confidence("left_elbow", 24, confidence, "left", None) == pytest.approx(confidence)


def test_confidence_never_exceeds_one():
    phases = KickPhaseReport(
        applied=True, detected=True, side="left", phases={"start": 0, "peak": 10, "return": 20}
    )
    assert _sign_confidence("left_knee", 10, 0.9, "left", phases) <= 1.0
    assert _sign_confidence("left_knee", 10, 0.5, "none", phases) <= 1.0


def test_pipeline_reports_swing_and_phases_without_touching_coordinates():
    frames = synth_pose.make_kick_frames(45, peak_deg=70.0, ramp=0.45)
    config = PostProcessConfig(fps=30.0, aspect_ratio=ASPECT)
    with_swing = run_postprocess(frames, config)
    without = run_postprocess(frames, config.model_copy(update={"swing": SwingConfig(enabled=False)}))

    assert with_swing.swing.swing_leg == "left"
    assert with_swing.kick_phases.detected is True
    assert without.swing.swing_leg == "none"
    assert without.kick_phases.detected is False
    # 关掉摆动腿识别 = 不改变任何坐标（因为深度符号先验本身也是关的）
    for left, right in zip(with_swing.frames, without.frames, strict=True):
        assert left.world_landmarks == right.world_landmarks


# --------------------------------------------------------------------------- #
# 4. 矢状面角度口径
# --------------------------------------------------------------------------- #
def test_sagittal_angles_at_rest_and_peak():
    rest = synth_pose.make_kick_frames(3, peak_deg=0.0)[0]
    elevation, flexion = sagittal_angles(rest, "left")
    assert elevation == pytest.approx(0.0, abs=1e-6)
    assert flexion == pytest.approx(0.0, abs=1e-6)

    peak = synth_pose.make_kick_frames(3, peak_deg=90.0, ramp=0.01)[1]
    elevation, _flexion = sagittal_angles(peak, "left")
    assert elevation == pytest.approx(90.0, abs=1.0)


def test_sagittal_angle_is_independent_of_y_axis_sign():
    """同一姿态把 y 取反（换坐标系约定）后，抬起角必须不变。"""
    from app.backend.engine.base import PoseFrame

    frame = synth_pose.make_kick_frames(3, peak_deg=60.0, ramp=0.01)[1]
    flipped = PoseFrame(
        index=frame.index,
        timestamp_ms=frame.timestamp_ms,
        landmarks=frame.landmarks,
        world_landmarks=[
            type(point)(x=point.x, y=-point.y, z=point.z, visibility=point.visibility)
            for point in frame.world_landmarks
        ],
        score=frame.score,
        valid=frame.valid,
        track_id=frame.track_id,
    )
    before, _ = sagittal_angles(frame, "left")
    after, _ = sagittal_angles(flipped, "left")
    # 抬起角是"从身体向下方向量起"的有向角，取反 y 后垂直基准随之翻转，
    # 因此同一个姿态应给出同一个角（数值同号同量级）
    assert math.copysign(1.0, before) == math.copysign(1.0, after)
    assert abs(before - after) < 1e-6
