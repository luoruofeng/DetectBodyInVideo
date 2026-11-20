"""脚部着地锚定（ground contact）。

深蹲时 MediaPipe world landmarks 的原点始终跟着髋走，髋的真实下沉被抹掉；
如果直接把「髋相对坐标」交给渲染器，膝一弯，脚踝反而向髋靠近，看起来就是
**双脚离地**。

本模块用一条保守的地面约束修正这件事：

1. 逐帧取髋到最低可见脚（踝/脚尖）的竖直跨度 ``span``；
2. 在整条人物轨迹上取较高分位数作为「站立参考腿长」``span_ref``；
3. 只把 ``root.y`` 调整为 ``span - span_ref``，使脚底保持在固定高度，
   髋随下蹲自然下沉。

只改 ``root.y``（竖直方向），不改髋相对关节坐标、x/z 位移或相机朝向：
- ``joints`` 仍是髋局部坐标，保持导出契约不变；
- 单目不可观测的水平位移/深度一律不伪造；
- 腾空帧（span 大于参考值）会表现为髋略高于参考位，而不是被强行按回地面。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.backend.skeleton.transform import ConvertedFrame, Vector3

#  ankle / foot_index 在 ARKit 命名下的脚部关节；取其中最低的可见点作为着地点。
FOOT_JOINTS: tuple[str, ...] = (
    "left_foot_joint",
    "right_foot_joint",
    "left_toes_joint",
    "right_toes_joint",
)

# 脚到髋的竖直跨度超出这个区间基本是误检（或人物只露出上半身），不参与锚定。
MIN_FOOT_SPAN_METERS = 0.25
MAX_FOOT_SPAN_METERS = 1.30


class GroundContactConfig(BaseModel):
    """地面锚定参数。默认开启，``enabled=False`` 时逐位透传输入。"""

    enabled: bool = True
    quantile: float = Field(
        0.85, ge=0.5, le=1.0, description="站立参考腿长的分位数；偏高可避开深蹲帧"
    )
    min_visibility: float = Field(0.3, ge=0.0, le=1.0)
    min_span_meters: float = Field(MIN_FOOT_SPAN_METERS, gt=0)
    max_span_meters: float = Field(MAX_FOOT_SPAN_METERS, gt=0)


class GroundReference(BaseModel):
    """单个人物轨迹的地面参考。"""

    track_id: int | None = None
    foot_span_meters: float = 0.0
    frames_used: int = 0


def foot_span(frame: ConvertedFrame, config: GroundContactConfig | None = None) -> float | None:
    """髋到最低可见脚的竖直距离（米）。``joints`` 是 y 向上的髋局部坐标。"""
    config = config or GroundContactConfig()
    if not frame.valid:
        return None
    spans: list[float] = []
    for name in FOOT_JOINTS:
        point = frame.joints.get(name)
        if point is None or point.visibility < config.min_visibility:
            continue
        span = -point.y
        if (
            math.isfinite(span)
            and config.min_span_meters <= span <= config.max_span_meters
        ):
            spans.append(span)
    return max(spans) if spans else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def estimate_ground_references(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig | None = None
) -> dict[int | None, GroundReference]:
    """按人物轨迹估计站立参考腿长。"""
    config = config or GroundContactConfig()
    spans_by_track: dict[int | None, list[float]] = {}
    for frame in frames:
        span = foot_span(frame, config)
        if span is not None:
            spans_by_track.setdefault(frame.track_id, []).append(span)

    references: dict[int | None, GroundReference] = {}
    for track_id, spans in spans_by_track.items():
        references[track_id] = GroundReference(
            track_id=track_id,
            # 保留全精度：参考值提前四舍五入会让恒定站姿产生 1e-6 的虚假位移。
            foot_span_meters=_percentile(spans, config.quantile),
            frames_used=len(spans),
        )
    return references


def anchor_feet_to_ground(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig | None = None
) -> tuple[list[ConvertedFrame], list[GroundReference]]:
    """把每帧 ``root.y`` 调整到「脚底固定、髋随下蹲下沉」，返回新帧与参考列表。"""
    config = config or GroundContactConfig()
    if not config.enabled:
        return list(frames), []

    references = estimate_ground_references(frames, config)
    outputs: list[ConvertedFrame] = []
    for frame in frames:
        reference = references.get(frame.track_id)
        span = foot_span(frame, config)
        if reference is None or span is None:
            outputs.append(frame)
            continue

        delta = round(span - reference.foot_span_meters, 6)
        outputs.append(
            frame.model_copy(update={"root": Vector3(x=0.0, y=delta, z=0.0)})
        )
    return outputs, list(references.values())


__all__ = [
    "FOOT_JOINTS",
    "GroundContactConfig",
    "GroundReference",
    "anchor_feet_to_ground",
    "estimate_ground_references",
    "foot_span",
]
