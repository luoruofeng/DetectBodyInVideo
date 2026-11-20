"""身体 yaw 估计：用"肩线/髋线的投影宽度"而不是 z 值（任务 3）。

为什么不用 z：单目视频的 ``z`` 是模型基于人体先验猜出来的，yaw 恰好是 z 上最不可观测的
自由度 —— 微小的 z 误差会放大成几十度的朝向误差。而肩线在图像上的**投影宽度**
是 2D 直接可测的量，精度高一个数量级：

    cos(yaw) ≈ 观测宽度 / 参考宽度

三个关键细节：

1. **参考宽度必须是"与 yaw 无关"的量**。``chest = mid(左右肩)``，所以"胸中点→肩"这条
   锁骨骨长在绕竖直轴旋转时不变，把左右锁骨长度相加就得到正对时的肩宽；
   直接用某一帧的观测宽度当参考会在转身时把参考一起缩小。
2. **参考宽度要换算到归一化坐标**，需要每帧的尺度 ``k``（1/米）。这里用"躯干在图像上的
   竖直跨度 ÷ 躯干骨长"估 ``k``：绕竖直轴旋转不改变竖直跨度，所以它天然免疫 yaw，
   也天然免疫"人物走近导致画面变大"（3.3 的透视校正就是这么来的）。
3. **符号**只用左右肩 ``z`` 差的**符号**（左肩更近 → yaw 为正），量级一律不采信。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PoseFrame,
)

if TYPE_CHECKING:  # 仅类型提示：ik_fit 与 postprocess.pipeline 互相依赖，运行期必须延迟导入
    from app.backend.postprocess.ik_fit import BoneLengths

logger = get_logger("skeleton.orientation")


class YawConfig(BaseModel):
    """yaw 估计参数。"""

    min_visibility: float = Field(0.5, ge=0, lt=1, description="低于该可见度的点不参与估计")
    side_view_ratio: float = Field(
        0.25,
        gt=0,
        le=1,
        description="宽度比低于此值视为正侧面：yaw 退化为 ±90°，不再细化",
    )
    frontal_ratio: float = Field(
        0.92, gt=0, le=1, description="宽度比高于此值视为正对：yaw≈0，置信度自然下降"
    )
    shoulder_weight: float = Field(1.0, gt=0, description="肩线权重（肩更宽，通常更准）")
    hip_weight: float = Field(0.6, gt=0, description="髋线权重")
    sign_epsilon: float = Field(
        0.01, gt=0, description="左右肩 z 差小于该值（米）时符号不可辨，按正对处理"
    )


class YawEstimate(BaseModel):
    """一帧的 yaw 估计结果。角度为弧度，正角度表示左肩转向镜头。"""

    index: int = 0
    yaw_rad: float = 0.0
    confidence: float = 0.0
    source: Literal["shoulders", "hips", "both", "degenerate", "none"] = "none"
    width_ratio: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)


def _visibility(frame: PoseFrame, index: int) -> float:
    if len(frame.landmarks) <= index:
        return 0.0
    return float(frame.landmarks[index].visibility)


def _pair_ok(frame: PoseFrame, first: int, second: int, threshold: float) -> bool:
    return min(_visibility(frame, first), _visibility(frame, second)) >= threshold


def _torso_vertical_span(frame: PoseFrame) -> float | None:
    """躯干（两肩中点 → 两髋中点）在图像上的竖直跨度（归一化 y）。

    用竖直跨度估尺度的好处：绕竖直轴旋转不改变它，人物走近导致画面变大时它也跟着变大，
    因此"观测宽度 / 参考宽度"这个比值天然免疫 yaw 与透视缩放。
    """
    if len(frame.landmarks) < LANDMARK_COUNT:
        return None
    points = frame.landmarks
    shoulder_mid_y = (points[LEFT_SHOULDER].y + points[RIGHT_SHOULDER].y) / 2.0
    hip_mid_y = (points[LEFT_HIP].y + points[RIGHT_HIP].y) / 2.0
    return abs(shoulder_mid_y - hip_mid_y)


def estimate_yaw(
    frame: PoseFrame,
    reference: BoneLengths | None = None,
    config: YawConfig | None = None,
    *,
    aspect: float = 9 / 16.0,
) -> YawEstimate:
    """估计单帧的身体 yaw。

    ``reference`` 用整段视频标定出的骨长（见 ``ik_fit.calibrate_bone_lengths``）；
    不传则用该帧自身标定（单帧可用，但抗噪能力差一些）。
    """
    config = config or YawConfig()
    notes: list[str] = []

    if len(frame.landmarks) < LANDMARK_COUNT:
        return YawEstimate(source="none", notes=["缺少归一化关键点"])
    if reference is None:
        from app.backend.postprocess.ik_fit import calibrate_bone_lengths

        reference = calibrate_bone_lengths([frame])

    torso = reference.get("chest", 0.45)
    vertical_span = _torso_vertical_span(frame)
    if vertical_span is None or torso <= 1e-6:
        return YawEstimate(source="none", notes=["躯干尺度不可估"])
    if vertical_span <= 1e-6:
        return YawEstimate(source="none", notes=["躯干竖直跨度为 0（可能正侧面）"])
    scale = vertical_span / torso  # 归一化单位 / 米

    shoulders_ok = _pair_ok(frame, LEFT_SHOULDER, RIGHT_SHOULDER, config.min_visibility)
    hips_ok = _pair_ok(frame, LEFT_HIP, RIGHT_HIP, config.min_visibility)
    if not shoulders_ok and not hips_ok:
        return YawEstimate(source="none", notes=["肩线与髋线都不可见"])

    candidates: list[tuple[float, float, str, float, float]] = []
    # (yaw, confidence, source, weight, ratio)
    if shoulders_ok:
        observed = abs(frame.landmarks[LEFT_SHOULDER].x - frame.landmarks[RIGHT_SHOULDER].x) * aspect
        reference_width = max(0.1, reference.shoulder_width) * scale
        ratio = observed / reference_width if reference_width > 1e-6 else 0.0
        candidates.append(
            (*_yaw_from_ratio(ratio, config, "shoulders"), config.shoulder_weight, ratio)
        )
    if hips_ok:
        observed = abs(frame.landmarks[LEFT_HIP].x - frame.landmarks[RIGHT_HIP].x) * aspect
        reference_width = max(0.1, reference.hip_width) * scale
        ratio = observed / reference_width if reference_width > 1e-6 else 0.0
        candidates.append((*_yaw_from_ratio(ratio, config, "hips"), config.hip_weight, ratio))

    if len(candidates) == 1:
        yaw, confidence, source, _, ratio = candidates[0]
        if source == "shoulders" and not hips_ok:
            notes.append("髋线不可见，仅用肩线估计")
        if source == "hips" and not shoulders_ok:
            notes.append("肩线不可见，仅用髋线估计")
    else:
        # 加权平均（按 权重 × 置信度），角度直接线性平均即可：两者范围一致且差值不大
        total = sum(weight * confidence for _, confidence, _, weight, _ in candidates) or 1.0
        yaw = sum(item[0] * item[1] * item[3] for item in candidates) / total
        confidence = sum(item[1] * item[3] for item in candidates) / sum(
            item[3] for item in candidates
        )
        ratio = sum(item[4] * item[3] for item in candidates) / sum(item[3] for item in candidates)
        source = "both"

    # 符号：只看左右肩/髋 z 差的符号（量级不采信）
    sign = _sign_from_depth(frame, config)
    if sign == 0:
        notes.append("左右肩 z 差过小，朝向符号按正对处理")
        sign = 1
    yaw = abs(yaw) * sign

    if any(item[2] == "degenerate" for item in candidates):
        source = "degenerate"
        notes.append("宽度比过小（正侧面），yaw 退化为 ±90° 且不再细化")

    return YawEstimate(
        index=frame.index,
        yaw_rad=yaw,
        confidence=max(0.0, min(1.0, confidence)),
        source=source,  # type: ignore[arg-type]
        width_ratio=ratio,
        notes=notes,
    )


def _yaw_from_ratio(
    ratio: float, config: YawConfig, source: str
) -> tuple[float, float, str]:
    """宽度比 → (yaw, confidence, source)。

    置信度：中间区间最高。比值≈1（正对）时 ``d cos/dyaw = 0``，测量对 yaw 不敏感；
    比值≈0（正侧）时 yaw 退化为 ±90°。
    """
    clipped = max(0.0, min(1.0, ratio))
    if clipped < config.side_view_ratio:
        return math.pi / 2.0, 0.15, "degenerate"
    yaw = math.acos(clipped)
    # 峰值放在 0.7 附近（≈45°），两端线性下降
    distance = abs(clipped - 0.7) / 0.55
    confidence = max(0.0, min(1.0, 1.0 - distance))
    if clipped > config.frontal_ratio:
        confidence = min(confidence, 0.45)
    return yaw, confidence, source


def _sign_from_depth(frame: PoseFrame, config: YawConfig) -> int:
    """左肩更靠近镜头（z 更小）→ 正角度。只用符号。"""
    if len(frame.world_landmarks) >= LANDMARK_COUNT:
        delta = frame.world_landmarks[RIGHT_SHOULDER].z - frame.world_landmarks[LEFT_SHOULDER].z
        if abs(delta) < config.sign_epsilon:
            return 0
        return 1 if delta > 0 else -1
    # 没有世界坐标就没有可靠的"谁更近"，交给调用方按"正对"处理
    return 0


def estimate_yaws(
    frames: Sequence[PoseFrame],
    reference: BoneLengths | None = None,
    config: YawConfig | None = None,
    *,
    aspect: float = 9 / 16.0,
) -> list[YawEstimate]:
    """整段估计（逐帧独立，顺序与输入一致）。"""
    ordered = sorted(frames, key=lambda item: item.index)
    if reference is None:
        from app.backend.postprocess.ik_fit import calibrate_bone_lengths

        reference = calibrate_bone_lengths(ordered)
    return [estimate_yaw(frame, reference, config, aspect=aspect) for frame in ordered]


def as_prior(estimates: Sequence[YawEstimate]) -> dict[int, tuple[float, float]]:
    """转成 ``ik_fit.fit_frames(yaws=...)`` 需要的 ``{帧号: (yaw, 置信度)}``。"""
    return {estimate.index: (estimate.yaw_rad, estimate.confidence) for estimate in estimates}


def yaw_from_world_z(frame: PoseFrame) -> float | None:
    """**旧算法**：直接用 ``world_landmarks`` 的肩线 z 差算 yaw（``atan2(-dz, dx)``）。

    保留它只为"新旧对比"和排查用（``scripts/bench_depth.py`` 会同时打印两者的抖动）：
    单目视频里 z 是模型猜的，肩线在深度方向只有十几厘米，噪声却有几厘米，
    因此这个角度的帧间抖动远大于宽度比方法。**生产路径不要用它。**
    """
    if len(frame.world_landmarks) < LANDMARK_COUNT:
        return None
    left = frame.world_landmarks[LEFT_SHOULDER]
    right = frame.world_landmarks[RIGHT_SHOULDER]
    dx = left.x - right.x
    dz = left.z - right.z
    if math.hypot(dx, dz) < 1e-6:
        return None
    return math.atan2(-dz, dx)


__all__ = [
    "YawConfig",
    "YawEstimate",
    "as_prior",
    "estimate_yaw",
    "estimate_yaws",
    "yaw_from_world_z",
]
