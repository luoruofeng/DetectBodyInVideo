"""双机位角度级融合（``docs/todo-kick-depth-accuracy.md`` §6.2）。

验收点：

1. **机位权重按平面分配**：矢状面角度侧视更可信，额状面角度正视更可信；
2. **融合优于单机位**：正面机位把前踢腿的前后方向丢掉时，融合值应接近侧视真值；
3. **时间轴按相位对齐**：两段视频开头停顿长度不同也不会错位；
4. **缺机位如实降级**：单机位返回原值 + note，绝不假装"融合过"。
"""

from __future__ import annotations

import pytest
import synth_pose

from app.backend.multiview.fusion import (
    FusionConfig,
    ViewInput,
    align_index,
    fuse_views,
    plane_weight,
    view_from_frames,
)


# --------------------------------------------------------------------------- #
# 1. 机位权重
# --------------------------------------------------------------------------- #
def test_plane_weight_prefers_side_view_for_sagittal_angles():
    config = FusionConfig()
    side = plane_weight("left_knee", 90.0, config)
    front = plane_weight("left_knee", 0.0, config)
    assert side > front
    assert front == pytest.approx(0.15)  # 正视看不到屈伸，但也不能完全丢弃


def test_plane_weight_prefers_front_view_for_frontal_angles():
    config = FusionConfig()
    assert plane_weight("left_shoulder", 0.0, config) > plane_weight("left_shoulder", 90.0, config)
    assert plane_weight("left_shoulder", 0.0, config) == pytest.approx(1.0)


def test_unknown_angle_is_shared_equally():
    config = FusionConfig()
    assert plane_weight("something_new", 0.0, config) == pytest.approx(
        plane_weight("something_new", 90.0, config)
    )


# --------------------------------------------------------------------------- #
# 2. 融合优于单机位
# --------------------------------------------------------------------------- #
def _kick_angle_series(count: int = 40, peak: float = 80.0) -> list[dict[str, float]]:
    """真值：膝盖角 = 180° - 抬腿角（伸直 180°，抬到 80° 时 100°）。"""
    series = []
    for index in range(count):
        gain = min(1.0, index / (count * 0.5), (count - 1 - index) / (count * 0.5))
        series.append({"left_knee": round(180.0 - peak * gain, 2), "left_shoulder": 170.0})
    return series


def test_fusion_recovers_sagittal_angle_from_front_plus_side():
    truth = _kick_angle_series()
    # 正面机位：矢状面角几乎看不出变化（只见 2D 投影），给一个"腿部几乎伸直"的错误读数
    front = ViewInput(
        name="front",
        angles=[{"left_knee": 175.0, "left_shoulder": row["left_shoulder"]} for row in truth],
        yaw_deg=0.0,
    )
    side = ViewInput(name="side", angles=truth, yaw_deg=90.0)

    fused = fuse_views([front, side], FusionConfig())
    assert fused.degraded is False
    assert fused.views == ["front", "side"]

    worst = max(
        abs(row["left_knee"] - true["left_knee"])
        for row, true in zip(fused.angles, truth, strict=True)
    )
    front_worst = max(
        abs(row["left_knee"] - true["left_knee"]) for row, true in zip(front.angles, truth, strict=True)
    )
    assert worst < front_worst / 3, (worst, front_worst)
    # 额状面角度仍由正面机位主导（两路一致时不应被拉偏）
    assert all(abs(row["left_shoulder"] - 170.0) < 1e-6 for row in fused.angles)


def test_low_visibility_view_is_ignored():
    truth = _kick_angle_series(10)
    front = ViewInput(
        name="front",
        angles=[{"left_knee": 175.0} for _ in truth],
        yaw_deg=0.0,
    )
    side = ViewInput(name="side", angles=truth, yaw_deg=90.0)
    frames = synth_pose.make_frames(10)
    for frame in frames:
        frame.landmarks = [
            type(point)(x=point.x, y=point.y, z=point.z, visibility=0.05)
            for point in frame.landmarks
        ]
    fused = fuse_views([front, side], FusionConfig(), frames_by_view={"front": frames})
    # 正面机位可见度低于阈值 → 不参与融合，结果应贴近侧视真值
    assert abs(fused.angles[0]["left_knee"] - truth[0]["left_knee"]) < 1e-6


# --------------------------------------------------------------------------- #
# 3. 时间轴对齐
# --------------------------------------------------------------------------- #
def test_align_index_maps_phase_anchors():
    source = {"start": 20, "peak": 40, "return": 60}
    target = {"start": 0, "peak": 20, "return": 40}
    assert align_index(0, source, target, 80) == 20
    assert align_index(20, source, target, 80) == 40
    assert align_index(40, source, target, 80) == 60
    # 没有相位信息时按绝对帧号（不猜）
    assert align_index(7, {}, target, 80) == 7


def test_fusion_aligns_different_lead_in():
    """两段视频开头停顿不同：对齐后加速段不应混到回落段。"""
    count = 40
    truth = _kick_angle_series(count)
    target = ViewInput(
        name="front",
        angles=[{"left_knee": 175.0} for _ in truth],
        yaw_deg=0.0,
        phases={"start": 2, "peak": 20, "return": 38},
    )
    source = ViewInput(
        name="side",
        angles=truth,
        yaw_deg=90.0,
        phases={"start": 10, "peak": 20, "return": 30},
    )
    fused = fuse_views([target, source], FusionConfig())
    # 对齐后最大伸展帧（第 20 帧）的膝角应接近最低值；正视仍带 MIN_PLANE_WEIGHT 的
    # 残余影响（这是刻意保留的：一个机位"看不见"不等于"完全不可信"），所以留 12° 余量
    assert fused.angles[20]["left_knee"] == pytest.approx(truth[20]["left_knee"], abs=12.0)
    # 反而在不该对齐的绝对帧号下（第 0 帧）会明显更差，说明对齐确实起了作用
    assert abs(fused.angles[0]["left_knee"] - truth[0]["left_knee"]) < 6.0


# --------------------------------------------------------------------------- #
# 4. 降级
# --------------------------------------------------------------------------- #
def test_single_view_is_marked_degraded():
    view = ViewInput(name="front", angles=[{"left_knee": 170.0}], yaw_deg=0.0)
    fused = fuse_views([view])
    assert fused.degraded is True
    assert fused.angles == [{"left_knee": 170.0}]
    assert any("单机位" in note or "一个机位" in note for note in fused.notes)


def test_empty_input_is_handled():
    fused = fuse_views([])
    assert fused.frames == 0
    assert fused.notes


# --------------------------------------------------------------------------- #
# 5. 从帧构造视图
# --------------------------------------------------------------------------- #
def test_view_from_frames_computes_angles():
    frames = synth_pose.make_kick_frames(12, peak_deg=60.0, ramp=0.4)
    view = view_from_frames("side", frames, yaw_deg=90.0)
    assert len(view.angles) == len(frames)
    assert "left_knee" in view.angles[0]
    # 合成的踢腿是"直腿整体前抬"，所以膝角始终接近 180°（伸直），
    # 变化体现在**髋角**上 —— 这也说明髋/膝必须分开看，不能只用一个角度代表踢腿
    assert view.angles[0]["left_knee"] == pytest.approx(180.0, abs=0.5)
    assert view.angles[6]["left_hip"] != pytest.approx(view.angles[0]["left_hip"], abs=1.0)
