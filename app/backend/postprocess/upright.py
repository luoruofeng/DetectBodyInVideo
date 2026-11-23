"""直立基准校准：修正"正对镜头站立却被解出躯干前倾"的单目假象。

单目正面画面看不到矢状面，MediaPipe world z 是基于人体先验的估计，直立的身体
常带稳定的 z 偏置。本模块只在开头若干帧**正对且静止**时估计一个偏置：

``chest.z - pelvis.z``：加性常数，从全片上半身（含两耳）的 z 里扣除。

**头部不在这里修**：``head_joint`` 的深度现在由被「固定骨长 + 2D 重投影 + 头部深度
先验」约束过的**鼻尖**反推（见 ``skeleton.mapping._head_reference``），不再读两耳那种
不受约束的 z，也就不需要"扣耳中点偏置"的补丁。早期版本曾在这里扣
``ear_mid.z - chest.z``，只对两耳生效；那条路已经退休，因为两耳的 z 本来就没有观测。

门控不满足时原样返回，不碰任何坐标。注意这是"没有证据就不掰"的折中：若视频开头
人物真的保持前倾静止，该段会被当成基准抹平。``max_bias_meters`` 用于拦截明显不可信
的大偏置。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
)
from app.backend.skeleton.orientation import YawEstimate
from app.backend.skeleton.topology import CHEST, NODES

logger = get_logger("postprocess.upright")

# 属于上半身、但不在 ``NODES`` 里的实关节：两耳不参与骨长拟合，却仍要跟着躯干基准
# 一起平移（rawLandmarks 与其它消费 world 点的下游都按"上半身整体"理解这次校正）。
HEAD_MARKER_INDICES: tuple[int, ...] = (LEFT_EAR, RIGHT_EAR)


class UprightBaselineConfig(BaseModel):
    enabled: bool = Field(
        True, description="开启直立基准校准；门控（正对 + 静止 + 偏置合理）不满足时自动跳过"
    )
    window: int = Field(60, ge=1, le=600, description="只看开头多少帧来估计基准")
    min_frames: int = Field(10, ge=3, le=300, description="至少多少帧满足正对静止才应用")
    frontal_ratio: float = Field(
        0.92, gt=0, le=1, description="yaw 观测宽度比高于该值视为正对（与 YawConfig 一致）"
    )
    max_motion: float = Field(
        0.02, ge=0, description="基准段相邻帧髋中点的最大 2D 位移（归一化画面高度）"
    )
    max_bias_meters: float = Field(
        0.25, ge=0, description="偏置绝对值上限（米）；超过说明可能是真前倾或估计异常，跳过"
    )


class UprightBaselineReport(BaseModel):
    applied: bool = False
    eligible_frames: int = 0
    baseline_meters: float = 0.0
    adjusted_frames: int = 0
    notes: list[str] = Field(default_factory=list)


_UPPER_BODY_INDICES: tuple[int, ...] = ()


def _upper_body_indices() -> tuple[int, ...]:
    """CHEST（左右肩中点）及其所有后代对应的实关节索引；髋/腿不属于上半身。

    两耳不在 ``NODES`` 里（它们不参与骨长约束拟合），但同样属于上半身，必须一起平移 ——
    否则躯干基准扣掉的常数会把胸/肩/鼻都挪回去，耳却留在原地。
    """
    global _UPPER_BODY_INDICES
    if _UPPER_BODY_INDICES:
        return _UPPER_BODY_INDICES
    keys = {CHEST}
    changed = True
    while changed:
        changed = False
        for node in NODES:
            if node.key in keys or node.parent not in keys:
                continue
            keys.add(node.key)
            changed = True
    indices = sorted(
        {
            node.index
            for node in NODES
            if node.key in keys and node.index is not None
        }
        | set(HEAD_MARKER_INDICES)
    )
    _UPPER_BODY_INDICES = tuple(indices)
    return _UPPER_BODY_INDICES


def _mid_world(
    frame: PoseFrame, first: int, second: int
) -> tuple[float, float, float] | None:
    points = frame.world_landmarks
    if len(points) < LANDMARK_COUNT:
        return None
    a, b = points[first], points[second]
    mid = ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0)
    return mid if all(math.isfinite(value) for value in mid) else None


def _hip_mid_2d(frame: PoseFrame) -> tuple[float, float] | None:
    points = frame.landmarks
    if len(points) < LANDMARK_COUNT:
        return None
    mid = (
        (points[LEFT_HIP].x + points[RIGHT_HIP].x) / 2.0,
        (points[LEFT_HIP].y + points[RIGHT_HIP].y) / 2.0,
    )
    return mid if all(math.isfinite(value) for value in mid) else None


def calibrate_upright_baseline(
    frames: Sequence[PoseFrame],
    yaws: Sequence[YawEstimate],
    config: UprightBaselineConfig | None = None,
) -> UprightBaselineReport:
    """估计并扣除直立时的躯干 z 偏置；原地修改 ``world_landmarks``，返回报告。

    只在 IK 之前调用：先扣偏置，再由固定骨长 + 2D 重投影把骨架拟合回自洽状态。
    归一化 ``landmarks`` 不动，避免影响 2D 观测与 yaw。

    偏置是 ``chest.z - pelvis.z`` 的加性常数，扣掉后整段上半身（含鼻尖、两耳、手）竖直。
    头相对躯干的前探**不在这里修**：``head_joint`` 的深度由鼻尖反推，见模块 docstring。
    """
    config = config or UprightBaselineConfig()
    report = UprightBaselineReport()
    if not config.enabled:
        report.notes.append("直立基准校准未启用")
        return report

    ordered = sorted(frames, key=lambda item: item.index)
    if len(ordered) < config.min_frames:
        report.notes.append(
            f"帧数不足（{len(ordered)} < {config.min_frames}），跳过直立基准校准"
        )
        return report

    yaw_by_index = {item.index: item for item in yaws}
    if not yaw_by_index:
        report.notes.append("没有 yaw 估计，无法确认正对，跳过直立基准校准")
        return report

    biases: list[float] = []
    previous: tuple[float, float] | None = None
    for frame in ordered[: config.window]:
        yaw = yaw_by_index.get(frame.index)
        mid_2d = _hip_mid_2d(frame)
        if (
            yaw is None
            or yaw.source == "none"
            or yaw.width_ratio < config.frontal_ratio
            or mid_2d is None
            or not frame.valid
        ):
            biases.clear()
            previous = None
            continue
        if previous is not None and math.hypot(
            mid_2d[0] - previous[0], mid_2d[1] - previous[1]
        ) > config.max_motion:
            # 基准必须来自一段连续静止的正对镜头片段，断掉就重新计数。
            biases.clear()
            previous = None
            continue
        previous = mid_2d
        chest = _mid_world(frame, LEFT_SHOULDER, RIGHT_SHOULDER)
        pelvis = _mid_world(frame, LEFT_HIP, RIGHT_HIP)
        if chest is None or pelvis is None:
            biases.clear()
            previous = None
            continue
        biases.append(chest[2] - pelvis[2])

    report.eligible_frames = len(biases)
    _apply_torso_baseline(report, ordered, biases, config)
    return report


def _apply_torso_baseline(
    report: UprightBaselineReport,
    ordered: Sequence[PoseFrame],
    biases: Sequence[float],
    config: UprightBaselineConfig,
) -> None:
    if len(biases) < config.min_frames:
        report.notes.append(
            f"正对静止帧不足（{len(biases)} < {config.min_frames}），跳过躯干直立基准校准"
        )
        return

    baseline = float(np.median(biases))
    if abs(baseline) > config.max_bias_meters:
        report.notes.append(
            f"躯干 z 偏置 {baseline:.3f} m 超过 {config.max_bias_meters} m，"
            "可能是真实前倾或估计异常，跳过躯干直立基准校准"
        )
        return

    indices = _upper_body_indices()
    for frame in ordered:
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        for index in indices:
            frame.world_landmarks[index].z -= baseline
        report.adjusted_frames += 1
    report.baseline_meters = baseline
    report.applied = True
    report.notes.append(
        f"扣除躯干直立基准：前 {config.window} 帧中 {len(biases)} 帧正对静止，"
        f"chest.z - pelvis.z 中位数 {baseline:.3f} m，已应用于 {report.adjusted_frames} 帧上半身"
    )
    logger.info(report.notes[-1])


__all__ = [
    "HEAD_MARKER_INDICES",
    "UprightBaselineConfig",
    "UprightBaselineReport",
    "calibrate_upright_baseline",
]
