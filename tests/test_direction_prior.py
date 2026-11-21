"""方向向量软先验（``docs/todo-kick-depth-accuracy.md`` §4）。

四条约束：

1. **向后兼容**：``(符号, 置信度)`` 与 ``(符号, 置信度, 方向)`` 两种条目都能用，
   旧的调用方/单测不用改；
2. **互斥**：同一帧同一骨段只走一套先验（有方向就不再叠加 hinge），否则两个先验互相拉扯；
3. **可对齐**：骨段方向已经对齐"方向线段"时，方向残差与梯度都为 0（不抢 2D 自由度）；
4. **可回滚**：``depth_direction_weight=0`` / ``depth_sparsity_weight=0`` 时，
   残差数组为空、输出与改造前逐位一致。
"""

from __future__ import annotations

import numpy as np
import pytest
import synth_pose

from app.backend.postprocess.ik_fit import (
    FitConfig,
    _direction_context,
    _entry_parts,
    _orthonormal_basis,
    _sign_context,
    calibrate_bone_lengths,
    fit_frames,
    node_positions,
)
from app.backend.skeleton.topology import node_map


# --------------------------------------------------------------------------- #
# 1. 条目解包
# --------------------------------------------------------------------------- #
def test_entry_parts_accepts_both_shapes():
    sign, confidence, direction = _entry_parts((-1, 0.8))
    assert sign == -1
    assert confidence == pytest.approx(0.8)
    assert direction is None

    sign, confidence, direction = _entry_parts((1, 0.9, np.array([0.0, 0.0, -2.0])))
    assert sign == 1
    assert confidence == pytest.approx(0.9)
    assert direction is not None
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[2] == pytest.approx(-1.0)


def test_entry_parts_ignores_zero_direction():
    _sign, _confidence, direction = _entry_parts((1, 0.9, np.zeros(3)))
    assert direction is None


def test_orthonormal_basis_is_orthonormal():
    for direction in (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.6, 0.0, -0.8]),
    ):
        first, second = _orthonormal_basis(direction)
        assert abs(float(np.dot(first, direction))) < 1e-9
        assert abs(float(np.dot(second, direction))) < 1e-9
        assert abs(float(np.dot(first, second))) < 1e-9
        assert np.linalg.norm(first) == pytest.approx(1.0)
        assert np.linalg.norm(second) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. 互斥与门槛
# --------------------------------------------------------------------------- #
def test_direction_disables_hinge_for_same_bone():
    signs = {
        "left_knee": (-1, 0.9, np.array([0.0, 0.0, -1.0])),
        "left_ankle": (-1, 0.9),
    }
    config = FitConfig(depth_direction_weight=0.3)
    sign_idx, *_ = _sign_context(signs, config, scale=0.45)
    dir_idx, *_ = _direction_context(signs, config)
    assert dir_idx.tolist() == [11]  # left_knee 在 NODES 中的下标
    assert sign_idx.tolist() == [12]  # 只有没有方向的 left_ankle 留在 hinge 里


def test_low_confidence_direction_is_dropped():
    signs = {"right_ankle": (-1, 0.3, np.array([0.0, 0.0, -1.0]))}
    config = FitConfig(depth_direction_weight=0.3, depth_direction_min_confidence=0.5)
    dir_idx, *_ = _direction_context(signs, config)
    assert dir_idx.size == 0


def test_direction_is_off_by_default():
    signs = {"left_knee": (-1, 0.9, np.array([0.0, 0.0, -1.0]))}
    dir_idx, *_ = _direction_context(signs, FitConfig())
    sign_idx, *_ = _sign_context(signs, FitConfig(), scale=0.45)
    assert dir_idx.size == 0
    # 默认配置下方向项不生效，所以 hinge 照旧接管这根骨段
    assert sign_idx.tolist() == [11]


def test_contradicting_signal_and_direction_is_ignored():
    """符号说"朝后"、方向说"朝前"：两者矛盾时不施加方向先验（宁可不改）。"""
    signs = {"left_knee": (-1, 0.9, np.array([0.0, 0.0, 1.0]))}
    dir_idx, *_ = _direction_context(signs, FitConfig(depth_direction_weight=0.3))
    assert dir_idx.size == 0


# --------------------------------------------------------------------------- #
# 3. 对齐时残差为 0
# --------------------------------------------------------------------------- #
def test_residual_is_zero_when_bone_already_aligned():
    frames = synth_pose.make_frames(1)
    frame = frames[0]
    positions = node_positions(frame.world_landmarks)
    sign_map = {}
    for key in ("left_knee", "left_ankle", "right_knee", "right_ankle"):
        node = node_map()[key]
        vector = positions[key] - positions[node.parent]
        unit = vector / np.linalg.norm(vector)
        sign_map[key] = (1 if unit[2] >= 0 else -1, 1.0, unit)

    lengths = calibrate_bone_lengths(frames)
    config = FitConfig(depth_direction_weight=0.5, depth_flat_weight=0.0)
    fitted, _report = fit_frames([frame], config, lengths=lengths, signs={0: sign_map})
    assert fitted
    # 输入方向已经是先验目标：拟合几乎不应移动关节（阈值留一点数值余量）
    shift = max(
        float(
            np.linalg.norm(
                np.array([a.x, a.y, a.z]) - np.array([b.x, b.y, b.z])
            )
        )
        for a, b in zip(fitted[0].world_landmarks, frame.world_landmarks, strict=True)
    )
    assert shift < 0.02


# --------------------------------------------------------------------------- #
# 4. 稀疏项与默认关闭
# --------------------------------------------------------------------------- #
def test_sparsity_shrinks_depth_when_enabled():
    frames = synth_pose.make_frames(1)
    frame = frames[0]
    # 人为给踝点加 20 cm 深度（2D 不变），这是"胯下凭空多出深度"的形态
    for index in (27, 28):
        point = frame.world_landmarks[index]
        frame.world_landmarks[index] = type(point)(
            x=point.x, y=point.y, z=point.z + 0.2, visibility=point.visibility
        )
    lengths = calibrate_bone_lengths(frames)

    def depth_of(config: FitConfig) -> float:
        fitted, _ = fit_frames([frame], config, lengths=lengths)
        left_ankle = fitted[0].world_landmarks[27]
        left_knee = fitted[0].world_landmarks[25]
        return abs(left_ankle.z - left_knee.z)

    base = FitConfig(depth_flat_weight=0.0, depth_sparsity_weight=0.0)
    sparsity = FitConfig(depth_flat_weight=0.0, depth_sparsity_weight=0.6)
    assert depth_of(sparsity) < depth_of(base)


def test_direction_prior_fit_is_idempotent():
    """重复拟合必须收敛（与 ``test_fit_is_stable_when_reapplied`` 同判据）。

    方向先验是"以本轮输入为锚点"的近似不动点迭代：允许小幅微调，但增量必须逐步缩小。
    """
    frames = synth_pose.add_depth_noise(synth_pose.make_frames(6))
    lengths = calibrate_bone_lengths(frames)
    positions = node_positions(frames[0].world_landmarks)
    signs = {}
    for index in range(len(frames)):
        signs[index] = {}
        for key in ("left_knee", "left_ankle"):
            node = node_map()[key]
            vector = positions[key] - positions[node.parent]
            unit = vector / np.linalg.norm(vector)
            signs[index][key] = (1 if unit[2] >= 0 else -1, 0.9, unit)

    config = FitConfig(depth_direction_weight=0.3, depth_flat_weight=0.0)
    current = frames
    drifts: list[float] = []
    for _ in range(4):
        fitted, _report = fit_frames(current, config, lengths=lengths, signs=signs)
        if current is not frames:
            drifts.append(
                max(
                    float(
                        np.linalg.norm(
                            [a.x - b.x, a.y - b.y, a.z - b.z]
                        )
                    )
                    for prev, new in zip(current, fitted, strict=True)
                    for a, b in zip(prev.world_landmarks, new.world_landmarks, strict=True)
                )
            )
        current = fitted

    assert len(drifts) == 3
    assert drifts[-1] <= drifts[0] + 1e-6, drifts
    assert max(drifts) < 0.05, drifts


def test_direction_prior_rollback_matches_legacy_output():
    """权重全为 0 时，带方向的符号图与两元组符号图必须给出逐位相同的输出。"""
    frames = synth_pose.make_frames(4, phase_swing=0.1)
    lengths = calibrate_bone_lengths(frames)
    positions = node_positions(frames[0].world_landmarks)
    plain = {}
    with_direction = {}
    for index in range(len(frames)):
        plain[index] = {}
        with_direction[index] = {}
        for key in ("left_knee", "left_ankle"):
            node = node_map()[key]
            vector = positions[key] - positions[node.parent]
            unit = vector / np.linalg.norm(vector)
            sign = 1 if unit[2] >= 0 else -1
            plain[index][key] = (sign, 0.9)
            with_direction[index][key] = (sign, 0.9, unit)

    config = FitConfig()
    left, _ = fit_frames(frames, config, lengths=lengths, signs=plain)
    right, _ = fit_frames(frames, config, lengths=lengths, signs=with_direction)
    for a, b in zip(left, right, strict=True):
        for point_a, point_b in zip(a.world_landmarks, b.world_landmarks, strict=True):
            assert point_a.x == pytest.approx(point_b.x, abs=1e-12)
            assert point_a.y == pytest.approx(point_b.y, abs=1e-12)
            assert point_a.z == pytest.approx(point_b.z, abs=1e-12)
