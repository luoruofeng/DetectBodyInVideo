"""双机位（正面 + 侧面）角度级融合（``docs/todo-kick-depth-accuracy.md`` §6.2）。

为什么**不做**三角化：

- 三角化需要两路严格同步的时间戳与相机内外参；手机拍的两段视频几乎没有同步，
  强迫三角化只会把"不同时刻的两个人"解出一个不存在的 3D；
- 而消费方要的是**关节角**（左右对称性、ROM）。角度是旋转量，本身就在
  "该机位看得清的平面"里，两个机位的可靠角度直接加权就够，且误差不会互相放大。

所以这里的融合只做三件事：

1. **按机位分配角度权重**：矢状面角度（膝/髋屈伸、躯干前倾）在侧视里准，
   额状面角度（肩外展、肘屈伸）在正视里准；
2. **按可见度折算**：被遮挡帧的角度不可信，权重乘可见度；
3. **按动作相位对齐时间轴**：两段视频的开头/停顿长度不同，用"起脚 → 最大伸展 →
   回落"三段线性映射对齐，而不是按绝对帧号硬凑。

降级：只有一个机位时原样返回该机位，并写一条 note（不静默）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import PoseFrame

logger = get_logger("multiview.fusion")

# 矢状面（前后方向）角度：侧视准；额状面（左右方向）角度：正视准
SAGITTAL_ANGLES: frozenset[str] = frozenset(
    {"left_knee", "right_knee", "left_hip", "right_hip", "torso_lean"}
)
FRONTAL_ANGLES: frozenset[str] = frozenset(
    {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}
)

# 有符号俯仰角的正负依赖机位朝向，跨机位直接平均会互相抵消；只作单机位诊断量。
DIAGNOSTIC_ANGLES: frozenset[str] = frozenset({"torso_pitch_signed_deg"})

# 机位看得"背对"该平面时，权重不会归零（否则一个机位缺失该角度就整帧丢失）
MIN_PLANE_WEIGHT = 0.15


class ViewInput(BaseModel):
    """一个机位的输入：已后处理的角度序列 + 机位朝向 + 相位（可选）。"""

    name: str = Field(..., description="机位名：front / side / 任意可读名字")
    angles: list[dict[str, float]] = Field(default_factory=list, description="逐帧关节角（度）")
    yaw_deg: float = Field(
        0.0, description="身体相对该机位的 yaw（度）：0 = 正对镜头，90 = 正侧面"
    )
    phases: dict[str, int] = Field(
        default_factory=dict, description="起脚/最大伸展/回落的帧号（用于时间轴对齐）"
    )
    fps: float = Field(30.0, gt=0)


class FusionConfig(BaseModel):
    """融合参数。默认值按"手机单目 + 三脚架"的典型素材定。"""

    min_visibility: float = Field(
        0.3, ge=0.0, le=1.0, description="低于该可见度的角度不参与融合"
    )
    side_view_bonus: float = Field(
        1.0, ge=0.0, le=3.0, description="侧视对矢状面角度的额外加权（主视角，通常 > 1）"
    )
    front_view_bonus: float = Field(
        1.0, ge=0.0, le=3.0, description="正视对额状面角度的额外加权"
    )
    align_by_phase: bool = Field(True, description="按动作相位对齐时间轴；关闭则按绝对帧号")


class FusedAngleSeries(BaseModel):
    frames: int = 0
    angles: list[dict[str, float]] = Field(default_factory=list)
    views: list[str] = Field(default_factory=list)
    degraded: bool = Field(False, description="True = 只有单机位（未真正融合）")
    alignment: dict[str, int] = Field(default_factory=dict, description="对齐后的重合帧数")
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 角度权重
# --------------------------------------------------------------------------- #
def plane_weight(angle: str, yaw_deg: float, config: FusionConfig) -> float:
    """该机位对该角度的可信度（0~1+）：由"角度所在平面"与机位朝向决定。

    ``yaw`` 定义与 ``skeleton.orientation`` 一致：0 = 正对镜头，90 = 正侧面。
    - 矢状面角度随 ``|sin(yaw)|`` 变好（侧视）；
    - 额状面角度随 ``|cos(yaw)|`` 变好（正视）。
    """
    yaw = np.radians(float(yaw_deg))
    if angle in SAGITTAL_ANGLES:
        weight = abs(float(np.sin(yaw))) * config.side_view_bonus
    elif angle in FRONTAL_ANGLES:
        weight = abs(float(np.cos(yaw))) * config.front_view_bonus
    else:  # 未知角度：不假设平面，两机位等权
        weight = 1.0
    return max(MIN_PLANE_WEIGHT, weight)


# --------------------------------------------------------------------------- #
# 时间轴对齐
# --------------------------------------------------------------------------- #
_PHASE_KEYS = ("start", "peak", "return")


def _phase_anchors(phases: dict[str, int], count: int) -> list[int] | None:
    """把相位转成 ``[起脚, 最大伸展, 回落]`` 三个帧号；缺失或顺序不合理时返回 None。"""
    if not phases:
        return None
    values: list[int] = []
    for key in _PHASE_KEYS:
        value = phases.get(key)
        if not isinstance(value, int):
            return None
        values.append(value)
    start, peak, ret = values
    if not (0 <= start < peak < ret < max(1, count)):
        return None
    return values


def align_index(index: int, source: dict[str, int], target: dict[str, int], count: int) -> int:
    """``index`` 是 **target** 时间轴上的帧号 → 返回它在 **source** 时间轴上的对应帧号。

    参照点就是动作本身：起脚、最大伸展、回落（再补上首尾）。这样"两段视频开头
    停留时长不同"不会让融合把加速段和回落段的样本混在一起。
    """
    source_anchors = _phase_anchors(source, count)
    target_anchors = _phase_anchors(target, count)
    if source_anchors is None or target_anchors is None:
        return index

    source_start, source_peak, source_return = source_anchors
    target_start, target_peak, target_return = target_anchors
    last = max(1, count - 1)

    # 起脚之前：按"开头停留长度"线性压缩（起脚帧本身映射到起脚帧）
    if index <= target_start:
        ratio = index / target_start if target_start > 0 else 1.0
        return int(round(ratio * source_start))
    # 起脚 → 最大伸展
    if index <= target_peak:
        span = max(1, target_peak - target_start)
        ratio = (index - target_start) / span
        return int(round(source_start + ratio * (source_peak - source_start)))
    # 最大伸展 → 回落
    if index <= target_return:
        span = max(1, target_return - target_peak)
        ratio = (index - target_peak) / span
        return int(round(source_peak + ratio * (source_return - source_peak)))
    # 回落之后
    span = max(1, last - target_return)
    ratio = (index - target_return) / span
    return int(round(source_return + ratio * (last - source_return)))


# --------------------------------------------------------------------------- #
# 融合
# --------------------------------------------------------------------------- #
def _visibility_of(frames: Sequence[PoseFrame] | None, index: int) -> float:
    if not frames or index < 0 or index >= len(frames):
        return 1.0
    frame = frames[index]
    if not frame.valid:
        return 0.0
    return float(np.clip(frame.mean_visibility(), 0.0, 1.0))


def fuse_views(
    views: Sequence[ViewInput],
    config: FusionConfig | None = None,
    *,
    frames_by_view: dict[str, Sequence[PoseFrame]] | None = None,
) -> FusedAngleSeries:
    """按机位可靠度加权融合角度序列；单机位时原样返回并标记 ``degraded``。"""
    config = config or FusionConfig()
    usable = [view for view in views if view.angles]
    if not usable:
        return FusedAngleSeries(notes=["没有任何机位提供角度序列"])

    target = max(usable, key=lambda view: len(view.angles))
    notes: list[str] = []
    if len(usable) == 1:
        notes.append(
            f"只有 {target.name} 一个机位：未做融合，矢状面角度（踢腿幅度）不可验证；"
            "建议补拍正侧面"
        )
        return FusedAngleSeries(
            frames=len(target.angles),
            angles=[dict(item) for item in target.angles],
            views=[target.name],
            degraded=True,
            notes=notes,
        )

    merged: list[dict[str, float]] = []
    contributions: dict[str, float] = {}
    total_frames = len(target.angles)

    for position in range(total_frames):
        row: dict[str, float] = {}
        keys: set[str] = set()
        for view in usable:
            keys.update(view.angles[position].keys() if position < len(view.angles) else set())
        for key in keys:
            if key in DIAGNOSTIC_ANGLES:
                continue
            weighted = 0.0
            weight_sum = 0.0
            for view in usable:
                source_index = position
                if view is not target:
                    source_index = (
                        align_index(position, view.phases, target.phases, len(target.angles))
                        if config.align_by_phase
                        else position
                    )
                if source_index < 0 or source_index >= len(view.angles):
                    continue
                value = view.angles[source_index].get(key)
                if value is None:
                    continue
                visibility = _visibility_of((frames_by_view or {}).get(view.name), source_index)
                if visibility < config.min_visibility:
                    continue
                weight = plane_weight(key, view.yaw_deg, config) * visibility
                weighted += float(value) * weight
                weight_sum += weight
            if weight_sum > 0:
                row[key] = round(weighted / weight_sum, 3)
                contributions[key] = contributions.get(key, 0.0) + weight_sum
        merged.append(row)

    view_names = [view.name for view in usable]
    notes.append(f"双机位融合：{view_names[0]} + {view_names[1]}，基准时间轴 {target.name}")
    for key in sorted(contributions):
        notes.append(
            f"  {key}：按机位可信度加权（{'矢状面' if key in SAGITTAL_ANGLES else '额状面'}）"
        )
    return FusedAngleSeries(
        frames=total_frames,
        angles=merged,
        views=view_names,
        degraded=False,
        notes=notes,
    )


def view_from_frames(
    name: str,
    frames: Sequence[PoseFrame],
    *,
    yaw_deg: float,
    phases: dict[str, int] | None = None,
    fps: float = 30.0,
) -> ViewInput:
    """便利构造：直接从已后处理的帧取角度（与导出用的同一套 ``compute_angles``）。"""
    from app.backend.postprocess.angles import compute_angles

    ordered = sorted(frames, key=lambda item: item.index)
    angles = [
        compute_angles(frame) if frame.valid else {}
        for frame in ordered
    ]
    return ViewInput(
        name=name,
        angles=angles,
        yaw_deg=float(yaw_deg),
        phases=phases or {},
        fps=fps,
    )


__all__ = [
    "DIAGNOSTIC_ANGLES",
    "FRONTAL_ANGLES",
    "FusedAngleSeries",
    "FusionConfig",
    "SAGITTAL_ANGLES",
    "ViewInput",
    "align_index",
    "fuse_views",
    "plane_weight",
    "view_from_frames",
]
