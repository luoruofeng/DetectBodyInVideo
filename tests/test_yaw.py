"""身体 yaw 估计（任务 3.4 的测试骨骼）。

覆盖范围：
1. 合成旋转 ±20°/±45°/±60° → 恢复误差 < 10°；
2. 正侧面（±90°）退化：yaw 落在 ±90° 附近、置信度显著低于中间角度；
3. 髋线被遮挡时退回肩线单侧估计并降权；
4. 与 ``world_landmarks`` 的 z 推算结果对比（对比表写入 ``docs/benchmark.md``）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import synth_pose

try:
    from app.backend.skeleton import orientation
except ImportError:  # pragma: no cover - 第 3 节前的过渡态
    orientation = None  # type: ignore[assignment]

requires_yaw = pytest.mark.skipif(orientation is None, reason="skeleton.orientation 尚未实现")

ANGLES = (20.0, -20.0, 45.0, -45.0, 60.0, -60.0)


@requires_yaw
@pytest.mark.parametrize("degrees", ANGLES)
def test_estimate_yaw_recovers_rotation(degrees: float):
    frame = synth_pose.make_frames(1, degrees=degrees)[0]
    result = orientation.estimate_yaw(frame)
    assert math.degrees(result.yaw_rad) == pytest.approx(degrees, abs=10.0)
    assert result.confidence >= 0.3


@requires_yaw
def test_side_view_is_degenerate_but_bounded():
    frame = synth_pose.make_frames(1, degrees=90.0)[0]
    result = orientation.estimate_yaw(frame)
    assert abs(math.degrees(result.yaw_rad)) == pytest.approx(90.0, abs=15.0)
    assert result.confidence < 0.4


@requires_yaw
def test_hip_occlusion_falls_back_to_shoulders():
    frame = synth_pose.make_frames(1, degrees=45.0)[0]
    for index in (23, 24):
        frame.landmarks[index].visibility = 0.0
        frame.world_landmarks[index].visibility = 0.0
    result = orientation.estimate_yaw(frame)
    assert result.source in {"shoulders", "degenerate"}
    assert math.degrees(result.yaw_rad) == pytest.approx(45.0, abs=12.0)


# --------------------------------------------------------------------------- #
# 3.4：新旧算法对比（宽度比 vs world z 推算）
# --------------------------------------------------------------------------- #
def _jitter_deg(values: list[float]) -> float:
    """帧间抖动：相邻帧差值的标准差（度）。"""
    if len(values) < 3:
        return 0.0
    return float(np.std(np.diff(np.asarray(values, dtype=np.float64))))


@requires_yaw
def test_width_ratio_yaw_is_much_steadier_than_depth_yaw():
    """静止转身姿态 + 深度噪声：宽度比方法的帧间抖动显著低于 z 推算。

    ``add_depth_noise`` 只动 ``world_landmarks``、不动归一化 2D，因此宽度比方法
    几乎不受影响；而 z 推算方法直接吃深度噪声 —— 这正是两者差距的来源。

    注意必须用**整段标定**的 ``estimate_yaws``（生产链路就是这么做的）：
    单帧 ``estimate_yaw`` 会拿该帧自己的世界坐标标定参考宽度，等于把深度噪声
    又引回来了（实测 20° 附近抖到 10° 以上）。
    """
    clean = synth_pose.make_frames(24, degrees=40.0)
    noisy = synth_pose.add_depth_noise(clean, z_sigma=0.03, xy_sigma=0.004)

    width = [math.degrees(item.yaw_rad) for item in orientation.estimate_yaws(noisy)]
    depth = [
        math.degrees(value)
        for value in (orientation.yaw_from_world_z(frame) for frame in noisy)
        if value is not None
    ]
    assert depth, "应当能算出 z 推算的 yaw"
    assert _jitter_deg(width) < _jitter_deg(depth) * 0.5


@requires_yaw
@pytest.mark.parametrize("degrees", ANGLES)
def test_width_ratio_error_is_below_depth_error(degrees: float):
    """带噪声时宽度比方法的平均误差也小于 z 推算（±60° 正侧面退化除外）。"""
    noisy = synth_pose.add_depth_noise(
        synth_pose.make_frames(12, degrees=degrees), z_sigma=0.03, xy_sigma=0.004
    )
    width = float(np.mean([math.degrees(item.yaw_rad) for item in orientation.estimate_yaws(noisy)]))
    depth = float(
        np.mean(
            [
                math.degrees(value)
                for value in (orientation.yaw_from_world_z(frame) for frame in noisy)
                if value is not None
            ]
        )
    )
    assert abs(width - degrees) < abs(depth - degrees)
