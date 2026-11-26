"""Z 轴独立强平滑（任务 4 的测试）。

覆盖范围：
1. 逐轴参数：``min_cutoff_z`` / ``beta_z`` 的默认值把 z 压得比 x/y 更狠；
2. 向后兼容：不传新参数时与旧版逐位一致；
3. 效果：z 的相邻帧 RMS 下降 ≥ 30%，同时 x/y 变化 < 5%；
4. 世界点独立平滑：``world_landmarks`` 单独一趟，缺世界坐标的帧不被写坏。
"""

from __future__ import annotations

import numpy as np
import pytest
import synth_pose

from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.postprocess.one_euro import (
    OneEuroFilter1D,
    OneEuroParams,
    default_one_euro_params,
    default_world_one_euro_params,
    smooth_array,
    smooth_array_zero_phase,
)
from app.backend.postprocess.pipeline import (
    PostProcessConfig,
    legacy_postprocess_config,
    run_postprocess,
)
from app.backend.postprocess.smoothing import SmoothingConfig, smooth_frames, smooth_world_pass

JITTER_LANDMARKS = (11, 12, 13, 14, 23, 24, 25, 26, 27, 28)


# --------------------------------------------------------------------------- #
# 4.1 / 4.2 逐轴参数
# --------------------------------------------------------------------------- #
def test_z_params_default_to_weaker_cutoff() -> None:
    """z 的 cutoff / beta 默认取 x/y 的 30%（1.0 → 0.3，0.5 → 0.15）。"""
    params = default_one_euro_params()
    assert params.min_cutoff_z == pytest.approx(params.min_cutoff * 0.3)
    assert params.beta_z == pytest.approx(params.beta * 0.3)
    assert params.d_cutoff_z == pytest.approx(params.d_cutoff)


def test_world_params_use_metric_beta() -> None:
    """米制世界点的 beta 要按"米/秒"重新标，不能沿用归一化那套。"""
    world = default_world_one_euro_params()
    normalized = default_one_euro_params()
    assert world.beta > normalized.beta
    assert world.min_cutoff_z == pytest.approx(world.min_cutoff * 0.3)


def test_vector_filter_uses_separate_z_filter() -> None:
    """同一个关键点的 z 轴衰减应当比 x 轴更慢（更强的平滑）。"""
    params = default_one_euro_params()
    x_filter = OneEuroFilter1D(params.axis_params(0))
    z_filter = OneEuroFilter1D(params.axis_params(2))
    assert x_filter.params.min_cutoff > z_filter.params.min_cutoff


# --------------------------------------------------------------------------- #
# 4.5 向后兼容
# --------------------------------------------------------------------------- #
def _legacy_smooth_array(array: np.ndarray, timestamps_ms: list[float]) -> np.ndarray:
    """改造前的实现（三轴共用一组参数），用于验证逐位兼容。"""
    from app.backend.postprocess.one_euro import timestamps_to_dt

    data = np.asarray(array, dtype=np.float32)
    deltas = timestamps_to_dt(timestamps_ms)
    result = data.copy()
    legacy = OneEuroParams(min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
    for landmark_index in range(LANDMARK_COUNT):
        filters = [OneEuroFilter1D(legacy) for _ in range(3)]
        for t in range(data.shape[0]):
            for axis, flt in enumerate(filters):
                result[t, landmark_index, axis] = flt.filter(
                    float(data[t, landmark_index, axis]), deltas[t]
                )
    return result


def test_new_fields_are_backward_compatible() -> None:
    """不传 ``*_z`` 时输出与旧版逐位一致（None = 沿用 x/y 的参数）。"""
    rng = np.random.default_rng(11)
    array = np.zeros((12, LANDMARK_COUNT, 4), dtype=np.float32)
    array[:, :, :3] = rng.normal(0.0, 0.02, size=(12, LANDMARK_COUNT, 3))
    array[:, :, 3] = 0.9
    stamps = [index * 33.0 for index in range(12)]

    modern = smooth_array(array, stamps, OneEuroParams(min_cutoff=1.0, beta=0.5))
    legacy = _legacy_smooth_array(array, stamps)
    assert np.array_equal(modern, legacy)


# --------------------------------------------------------------------------- #
# 4.5 z 抖动下降 / x-y 不受影响
# --------------------------------------------------------------------------- #
def _adjacent_rms(frames: list[PoseFrame], axes: tuple[int, ...]) -> float:
    squared: list[float] = []
    for previous, current in zip(frames, frames[1:], strict=False):
        for index in JITTER_LANDMARKS:
            a = previous.world_landmarks[index]
            b = current.world_landmarks[index]
            values = (a.x, a.y, a.z)
            others = (b.x, b.y, b.z)
            squared.append(sum((others[axis] - values[axis]) ** 2 for axis in axes))
    return float(np.sqrt(np.mean(squared))) if squared else 0.0


def test_z_smoothing_cuts_z_jitter_without_touching_xy() -> None:
    noisy = synth_pose.add_depth_noise(synth_pose.make_frames(30), z_sigma=0.02, xy_sigma=0.006)

    legacy = SmoothingConfig(
        one_euro=OneEuroParams(min_cutoff=1.0, beta=0.5),
        world_one_euro=OneEuroParams(min_cutoff=1.0, beta=1.0),
    )
    default = SmoothingConfig()  # x/y 1.0/0.5 + z 0.3/0.15

    before, _ = smooth_world_pass(noisy, legacy)
    after, _ = smooth_world_pass(noisy, default)

    z_before = _adjacent_rms(before, (2,))
    z_after = _adjacent_rms(after, (2,))
    xy_before = _adjacent_rms(before, (0, 1))
    xy_after = _adjacent_rms(after, (0, 1))

    assert z_after < z_before * 0.7, f"z 抖动应下降 ≥30%（{z_before:.5f} → {z_after:.5f}）"
    assert abs(xy_after - xy_before) / max(1e-9, xy_before) < 0.05


def test_smooth_frames_does_not_write_metric_values_into_normalized_landmarks() -> None:
    """没有世界坐标的帧在 use_world 模式下必须原样返回，不能被米制数值污染。"""
    frames = synth_pose.make_frames(6)
    for frame in frames[2:4]:
        frame.world_landmarks = []

    smoothed = smooth_frames(frames, SmoothingConfig(), use_world=True)[0]
    for frame in smoothed[2:4]:
        assert frame.world_landmarks == []
        assert frame.landmarks[11].x == pytest.approx(frames[frame.index].landmarks[11].x)


# --------------------------------------------------------------------------- #
# 4.3 世界点独立平滑接入流水线
# --------------------------------------------------------------------------- #
def test_pipeline_smooths_world_landmarks_separately() -> None:
    frames = synth_pose.add_depth_noise(synth_pose.make_frames(24), z_sigma=0.02, xy_sigma=0.006)
    result = run_postprocess(
        frames,
        PostProcessConfig(
            ik_fit=PostProcessConfig().ik_fit.model_copy(update={"enabled": False})
        ),
    )
    assert result.smoothing.applied is True
    assert any(key.startswith("world") for key in result.smoothing.params)
    assert any("world_landmarks" in note for note in result.smoothing.notes)


def test_pipeline_world_smoothing_can_be_disabled() -> None:
    frames = synth_pose.make_frames(12)
    config = PostProcessConfig(
        smoothing=SmoothingConfig(extra_world_pass=False),
        ik_fit=PostProcessConfig().ik_fit.model_copy(update={"enabled": False}),
    )
    result = run_postprocess(frames, config)
    assert not any(key.startswith("world") for key in result.smoothing.params)


def test_legacy_config_turns_off_every_new_switch() -> None:
    """DoD：可回滚 —— 一个入口把所有新开关关掉。"""
    config = legacy_postprocess_config()
    assert config.smoothing.one_euro.min_cutoff_z is None
    assert config.smoothing.one_euro.beta_z is None
    assert config.smoothing.extra_world_pass is False
    assert config.ik_fit.depth_prior_weight == pytest.approx(0.05)
    assert config.ik_fit.temporal_axes == "xyz"
    assert config.estimate_orientation is False
    assert config.quality.enable_depth_gate is False
    assert config.quality.enable_framing_rules is False


def test_legacy_pipeline_has_no_world_pass_and_no_yaw() -> None:
    frames = synth_pose.add_depth_noise(synth_pose.make_frames(12), z_sigma=0.02)
    result = run_postprocess(frames, legacy_postprocess_config())
    assert result.yaw == []
    assert not any(key.startswith("world") for key in result.smoothing.params)
    assert not any("world_landmarks" in note for note in result.smoothing.notes)


def test_zero_phase_world_pass_is_default_and_reported() -> None:
    """零相位是默认档，且要能在 ``quality.smoothing`` 里看到（可复现、可回滚）。"""
    assert SmoothingConfig().zero_phase_world is True
    frames = synth_pose.add_depth_noise(synth_pose.make_frames(24), z_sigma=0.02)
    result = run_postprocess(
        frames,
        PostProcessConfig(
            ik_fit=PostProcessConfig().ik_fit.model_copy(update={"enabled": False})
        ),
    )
    assert result.smoothing.params.get("worldZeroPhase") is True


def test_zero_phase_cuts_more_jitter_than_causal() -> None:
    """前向+后向应当比单向压得更狠（同一组参数、同一段噪声）。"""
    frames = synth_pose.add_depth_noise(synth_pose.make_frames(40), z_sigma=0.02, xy_sigma=0.006)
    causal = smooth_frames(frames, SmoothingConfig(), use_world=True, zero_phase=False)[0]
    zero_phase = smooth_frames(frames, SmoothingConfig(), use_world=True, zero_phase=True)[0]

    z_causal = _adjacent_rms(causal, (2,))
    z_zero = _adjacent_rms(zero_phase, (2,))
    assert z_zero < z_causal, f"零相位应更平滑（{z_causal:.5f} → {z_zero:.5f}）"


def test_zero_phase_does_not_lag_a_constant_velocity_ramp() -> None:
    """单向滤波必然滞后；零相位在匀速段应当把位置还原回来（误差更小）。"""
    count = 40
    array = np.zeros((count, LANDMARK_COUNT, 4), dtype=np.float32)
    array[:, :, 3] = 0.95
    ramp = np.linspace(0.0, 0.5, count, dtype=np.float32)
    array[:, :, 0] = ramp[:, None]
    stamps = [index * 33.0 for index in range(count)]

    interior = slice(10, count - 10)
    causal = smooth_array(array, stamps, default_world_one_euro_params())[interior, 0, 0]
    zero = smooth_array_zero_phase(array, stamps, default_world_one_euro_params())[interior, 0, 0]
    truth = ramp[interior]

    causal_error = float(np.mean(np.abs(causal - truth)))
    zero_error = float(np.mean(np.abs(zero - truth)))
    assert zero_error < causal_error, f"零相位不该滞后（{causal_error:.5f} vs {zero_error:.5f}）"


def test_acceleration_prior_is_off_by_default() -> None:
    """二阶项默认 0：关掉时输出与改动前逐位一致（只作为可选旋钮）。"""
    from app.backend.postprocess.ik_fit import FitConfig

    assert FitConfig().acceleration_weight == 0.0


def test_landmark_helper_keeps_visibility() -> None:
    """平滑永远不动 visibility（置信度是观测事实）。"""
    frame = PoseFrame(
        index=0,
        timestamp_ms=0.0,
        landmarks=[Landmark(x=0.1, y=0.2, z=0.3, visibility=0.42) for _ in range(LANDMARK_COUNT)],
        world_landmarks=[
            Landmark(x=0.1, y=0.2, z=0.3, visibility=0.37) for _ in range(LANDMARK_COUNT)
        ],
        valid=True,
    )
    frames = [frame.model_copy(deep=True) for _ in range(6)]
    for index, item in enumerate(frames):
        item.index = index
        item.timestamp_ms = index * 33.0
    smoothed = smooth_frames(frames, SmoothingConfig(), use_world=True)[0]
    assert smoothed[0].world_landmarks[11].visibility == pytest.approx(0.37)
