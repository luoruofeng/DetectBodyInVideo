"""深度符号先验（``postprocess.depth_sign``）与 ik_fit 分支翻转的测试。

核心命题只有一条：弱透视 + 已知骨长下，每根骨段的 ``ΔZ`` 大小被 2D 投影长度锁死，
**只剩符号**。所以本文件构造的"前踢腿"故障是**等距镜像**：
把整条腿关于 ``z = 髋.z`` 平面反射 —— 2D 投影一个像素都不变、骨长一点不变，
只有深度符号反了。这正好复现"前踢腿被解成后踢腿"的真实故障形态，
也保证任何断言都不会被"重投影误差顺带变化"污染。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import synth_pose

from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    Landmark,
    PoseFrame,
)
from app.backend.postprocess import depth_sign, ik_fit
from app.backend.postprocess.depth_sign import DepthSignConfig
from app.backend.postprocess.pipeline import (
    PostProcessConfig,
    legacy_postprocess_config,
    run_postprocess,
)
from app.backend.postprocess.quality import build_quality_report, sagittal_motion_share

ASPECT = synth_pose.ASPECT
LEFT_LEG = (LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX)


# --------------------------------------------------------------------------- #
# 合成"前踢腿"序列
# --------------------------------------------------------------------------- #
def _kick_frames(count: int = 40, peak_deg: float = 75.0, fps: float = 30.0) -> list[PoseFrame]:
    """左腿绕左髋向镜头方向摆动（两端各 20% 过渡、中间保持），2D 与 3D 严格自洽。"""
    base = synth_pose.standing_points()
    frames: list[PoseFrame] = []
    for index in range(count):
        phase = index / max(1, count - 1)
        ramp = max(0.0, min(1.0, phase / 0.2, (1.0 - phase) / 0.2))
        angle = math.radians(peak_deg) * ramp
        points = {key: value.copy() for key, value in base.items()}
        hip = points[LEFT_HIP].copy()
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for index_of_leg in LEFT_LEG:
            offset = points[index_of_leg] - hip
            # 绕 x 轴旋转：dy 为负（腿向下），正向旋转后 dz 变负（靠近相机）
            points[index_of_leg] = hip + np.array(
                [offset[0], offset[1] * cos_a, offset[1] * sin_a]
            )
        normalized = synth_pose.project_to_normalized(points)
        frames.append(
            PoseFrame(
                index=index,
                timestamp_ms=round(index * 1000.0 / fps, 4),
                landmarks=[
                    Landmark(
                        x=normalized[i][0],
                        y=normalized[i][1],
                        z=normalized[i][2],
                        visibility=synth_pose.VISIBILITY,
                    )
                    for i in range(LANDMARK_COUNT)
                ],
                world_landmarks=[
                    Landmark(
                        x=float(points[i][0]),
                        y=float(points[i][1]),
                        z=float(points[i][2]),
                        visibility=synth_pose.VISIBILITY,
                    )
                    for i in range(LANDMARK_COUNT)
                ],
                score=synth_pose.VISIBILITY,
                valid=True,
                track_id=1,
            )
        )
    return frames


def _mirror_leg_depth(frames: list[PoseFrame]) -> list[PoseFrame]:
    """把左腿关于 ``z = 髋.z`` 反射：等距变换，2D 与骨长完全不变，只有 Δz 反号。"""
    mirrored: list[PoseFrame] = []
    for frame in frames:
        copy = frame.model_copy(deep=True)
        hip_z = copy.world_landmarks[LEFT_HIP].z
        for index in LEFT_LEG:
            point = copy.world_landmarks[index]
            point.z = 2.0 * hip_z - point.z
        mirrored.append(copy)
    return mirrored


def _leg_dz_cm(frame: PoseFrame) -> float:
    return (frame.world_landmarks[LEFT_KNEE].z - frame.world_landmarks[LEFT_HIP].z) * 100.0


def _reproj_error(frames: list[PoseFrame]) -> float:
    from scripts.bench_depth import reprojection_error_px

    values = [
        value
        for value in (
            reprojection_error_px(frame, aspect=ASPECT, width=1080, height=1920)
            for frame in frames
        )
        if value is not None
    ]
    return float(np.mean(values)) if values else 0.0


def _forward_signs(frames: list[PoseFrame]) -> dict[int, dict[str, tuple[int, float]]]:
    """正确方向：前踢腿时 ``Δz < 0``（``-z`` 朝相机）。"""
    return {
        frame.index: {"left_knee": (-1, 1.0), "left_ankle": (-1, 1.0)}
        for frame in frames
    }


# --------------------------------------------------------------------------- #
# 构造的故障必须是"只有符号错"
# --------------------------------------------------------------------------- #
def test_mirror_is_an_isometry_of_2d_and_bone_length():
    frames = _kick_frames(12)
    mirrored = _mirror_leg_depth(frames)
    assert _leg_dz_cm(frames[6]) < -20.0
    assert _leg_dz_cm(mirrored[6]) > 20.0
    # 2D 一个像素都不变 → 重投影误差不变 → 任何差异只能来自符号
    assert _reproj_error(frames) == pytest.approx(_reproj_error(mirrored), abs=1e-9)
    for key, length in synth_pose.bone_lengths(frames[6]).items():
        assert synth_pose.bone_lengths(mirrored[6])[key] == pytest.approx(length, abs=1e-9)


def test_depth_sign_flips_metric():
    """指标口径：干净序列 ≈ 0；逐帧交替镜像后明显大于 0。"""
    from scripts.bench_depth import depth_sign_flips

    clean = depth_sign_flips(_kick_frames(20))
    assert clean["total"] == 0, clean

    frames = _kick_frames(20)
    for position, frame in enumerate(frames):
        if position % 2 == 0 and abs(_leg_dz_cm(frame)) > 5.0:
            frame.world_landmarks[LEFT_KNEE].z = -frame.world_landmarks[LEFT_KNEE].z
    assert depth_sign_flips(frames)["total"] > 0


def test_sagittal_share_flags_front_view_kick():
    """正面机位下的前踢腿必须被识别为"前后方向不可观测"，并给出补拍建议。"""
    frames = _kick_frames(40, peak_deg=75.0)
    share = sagittal_motion_share(frames, aspect=ASPECT, threshold=0.5)
    assert share > 0.25, share
    report = build_quality_report(frames, aspect=ASPECT)
    assert any("正侧机位" in issue for issue in report.issues), report.issues


def test_standing_pose_is_not_flagged():
    """站直/下蹲这类骨段躺在画面平面内的动作不该被误报。"""
    frames = synth_pose.make_frames(10)
    assert sagittal_motion_share(frames, aspect=ASPECT) < 0.25


# --------------------------------------------------------------------------- #
# 符号先验本身
# --------------------------------------------------------------------------- #
def test_despike_merges_short_runs():
    # 长度 1 的尖刺被两侧一致的邻居吞并
    assert depth_sign._despike([1, 1, -1, 1, 1, 1], min_run=3) == [1, 1, 1, 1, 1, 1]
    # 达到 min_run 的游程是真实方向，原样保留
    assert depth_sign._despike([1, 1, -1, -1, -1, 1], min_run=3) == [1, 1, -1, -1, -1, 1]


def test_temporal_prior_removes_sporadic_flips():
    """零星翻转 → 时序投票后符号稳定（这是"腿突然翻面"的直接解药）。"""
    frames = _kick_frames(40, peak_deg=75.0)
    for position, frame in enumerate(frames):
        if position % 7 == 3 and abs(_leg_dz_cm(frame)) > 5.0:
            frame.world_landmarks[LEFT_KNEE].z = -frame.world_landmarks[LEFT_KNEE].z

    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True), aspect=ASPECT
    )
    assert report.applied is True
    assert report.flips_before > 0
    assert report.flips_after == 0, report
    left_knee = [
        value["left_knee"][0] for _, value in sorted(signs.items()) if "left_knee" in value
    ]
    assert left_knee and len(set(left_knee)) == 1


def test_depth_sign_requires_enough_frames():
    frames = _kick_frames(12)
    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True), aspect=ASPECT
    )
    assert signs == {}
    assert report.applied is False
    assert any("时序证据不足" in note for note in report.notes)


def test_depth_sign_without_world_landmarks():
    frames = synth_pose.make_frames(40)
    for frame in frames:
        frame.world_landmarks = []
    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True), aspect=ASPECT
    )
    assert signs == {}
    assert report.applied is False


def test_disabled_config_produces_nothing():
    frames = _kick_frames(40)
    signs, report = depth_sign.estimate_depth_signs(frames, DepthSignConfig(), aspect=ASPECT)
    assert signs == {}
    assert report.mode == "off"
    assert report.applied is False


def test_model_mode_without_weights_falls_back_to_temporal(monkeypatch):
    """缺权重必须静默退回时序模式：分析链路不能因为可选模型缺失而失败。"""
    frames = _kick_frames(40)
    monkeypatch.setattr(depth_sign, "resolve_model_path", lambda config, lifter=None: None)
    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )
    assert report.applied is True
    assert report.mode == "temporal"
    assert report.lifter == depth_sign.resolve_lifter_name()
    assert any("权重" in note and "退回时序" in note for note in report.notes), report.notes


# --------------------------------------------------------------------------- #
# ik_fit：分支翻转 + hinge
# --------------------------------------------------------------------------- #
def test_hinge_flips_the_depth_branch():
    """等距镜像输入 + 正确符号先验 → 输出必须落在正确分支，且 2D 不变。"""
    frames = _mirror_leg_depth(_kick_frames(10, peak_deg=70.0))
    signs = _forward_signs(frames)

    baseline, _ = ik_fit.fit_frames(frames)
    fitted, report = ik_fit.fit_frames(frames, signs=signs)

    assert report.applied is True
    assert np.mean([_leg_dz_cm(frame) for frame in baseline]) > 10.0
    assert np.mean([_leg_dz_cm(frame) for frame in fitted]) < -20.0
    # 只翻 z 不动 2D：重投影误差必须基本不变（镜像是等距变换，2D 残差没有理由变差）
    assert _reproj_error(fitted) <= max(_reproj_error(baseline) + 2.0, _reproj_error(baseline) * 1.2)
    assert any("深度符号先验" in note for note in report.notes)


def test_correct_sign_keeps_orientation_and_stops_depth_collapse():
    """输入符号本来就对时：不能翻转方向，而且**不该被时域项拉回 0**。

    后半句是这一层的附加收益：时域 z 残差原本是"深度没有观测"的替代品，
    一旦有方向先验，它就该退位，否则动作起始帧的深度会被压向上一帧（"踢腿慢半拍"）。
    """
    frames = _kick_frames(10, peak_deg=70.0)
    signs = _forward_signs(frames)
    baseline, _ = ik_fit.fit_frames(frames)
    fitted, _ = ik_fit.fit_frames(frames, signs=signs)

    graded = [
        (before, after)
        for before, after in zip(baseline, fitted, strict=True)
        if abs(_leg_dz_cm(before)) > 3.0
    ]
    assert graded
    for before, after in graded:
        # 方向不许被改反
        assert _leg_dz_cm(before) * _leg_dz_cm(after) > 0, (before.index, _leg_dz_cm(before), _leg_dz_cm(after))
        # 深度幅度只许保留或变大（抑制"慢半拍"的塌陷）
        assert abs(_leg_dz_cm(after)) >= abs(_leg_dz_cm(before)) - 0.5, (
            before.index,
            _leg_dz_cm(before),
            _leg_dz_cm(after),
        )
    # 2.5 px 的余量：这条是"别为了符号把画面内拟合搞崩"的粗筛，不是精度断言。
    # （合成骨架的头部只有 6 cm 前伸、比解剖值 9 cm 扁，头部深度目标会在这里花掉
    #  约 2.2 px —— 那是夹具与人体测量学不一致，不是符号先验的代价。）
    assert _reproj_error(fitted) <= max(_reproj_error(baseline) + 2.5, _reproj_error(baseline) * 1.2)


def test_sign_prior_weight_zero_is_a_noop():
    frames = _mirror_leg_depth(_kick_frames(8, peak_deg=70.0))
    signs = _forward_signs(frames)
    baseline, _ = ik_fit.fit_frames(frames)
    fitted, _ = ik_fit.fit_frames(
        frames,
        ik_fit.FitConfig(depth_sign_weight=0.0),
        signs=signs,
    )
    for before, after in zip(baseline, fitted, strict=True):
        assert before.world_landmarks[LEFT_KNEE].z == pytest.approx(
            after.world_landmarks[LEFT_KNEE].z, abs=1e-9
        )


def test_flip_depth_branch_is_involutive():
    d0 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    idx = np.array([0], dtype=np.int64)
    flipped = ik_fit._flip_depth_branch(d0, idx, np.array([-1.0]))
    assert flipped[0, 2] == pytest.approx(-1.0)
    assert np.linalg.norm(flipped[0]) == pytest.approx(1.0)
    restored = ik_fit._flip_depth_branch(flipped, idx, np.array([1.0]))
    assert restored[0, 2] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_lift_stage_progress_and_quality_notes():
    """§2.1 的可观测性：后处理阶段要上报 ``lift``，质检 notes 要能看到符号先验的结论。"""
    frames = _mirror_leg_depth(_kick_frames(40, peak_deg=70.0))
    stages: list[tuple[str, str]] = []
    config = PostProcessConfig(
        fps=30.0, aspect_ratio=ASPECT, depth_sign=DepthSignConfig(enabled=True)
    )
    result = run_postprocess(
        frames,
        config,
        progress=lambda stage, message, done, total: stages.append((stage, message)),
    )
    assert ("lift", "修正前后方向") in stages
    assert result.depth_sign.mode == "temporal"
    assert result.depth_sign.flips_after <= result.depth_sign.flips_before
    assert any("深度方向修正" in note for note in result.quality.notes), result.quality.notes


def test_postprocess_reports_lift_stage():
    frames = _mirror_leg_depth(_kick_frames(40, peak_deg=70.0))
    config = PostProcessConfig(
        fps=30.0, aspect_ratio=ASPECT, depth_sign=DepthSignConfig(enabled=True)
    )
    result = run_postprocess(frames, config)
    assert result.depth_sign.applied is True
    assert result.depth_sign.segments > 0
    assert result.fit.applied is True
    for frame in result.frames:
        for key, length in synth_pose.bone_lengths(frame).items():
            target = result.fit.bone_lengths.get(key, length)
            assert abs(length - target) / target < 0.02, (frame.index, key)


def test_legacy_config_disables_depth_sign():
    config = legacy_postprocess_config()
    assert config.depth_sign.enabled is False
    assert config.ik_fit.depth_sign_weight == pytest.approx(0.0)
    result = run_postprocess(synth_pose.make_frames(6), config)
    assert result.depth_sign.applied is False
    assert result.depth_sign.mode == "off"
