"""关节角度计算（任务 5.3）。

- 三点夹角（肩-肘-腕、髋-膝-踝等），输出角度制 0~180°。
- 退化输入（零长度向量、NaN、缺关键点）一律返回 ``None``，绝不产出 NaN，
  导出 JSON 时直接省略该字段。
- 躯干前倾单独用 atan2(水平位移, 竖直位移) 计算，比三点夹角更稳定。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
    PoseFrame,
)

logger = get_logger("postprocess.angles")

# 名称 → (起点, 顶点, 终点)，顶点处即被测量的关节
JOINT_ANGLE_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_shoulder": (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    "right_shoulder": (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
    "left_knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "left_hip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
}

ANGLE_NAMES: tuple[str, ...] = (*JOINT_ANGLE_TRIPLETS, "torso_lean")


class AngleConfig(BaseModel):
    use_world_landmarks: bool = Field(
        True, description="优先用米制 world landmarks（尺度无关，更稳）；缺失时自动退回归一化坐标"
    )
    min_visibility: float = Field(0.3, ge=0, lt=1)
    round_digits: int = Field(2, ge=0, le=4)


def _vector(origin: Landmark, target: Landmark) -> np.ndarray:
    return np.array(
        [target.x - origin.x, target.y - origin.y, target.z - origin.z], dtype=np.float64
    )


def angle_deg(
    start: Landmark,
    vertex: Landmark,
    end: Landmark,
    *,
    min_visibility: float = 0.0,
) -> float | None:
    """三点夹角；任一向量长度≈0、坐标非有限、置信度不足 → ``None``。"""
    if min_visibility > 0 and (
        start.visibility < min_visibility
        or vertex.visibility < min_visibility
        or end.visibility < min_visibility
    ):
        return None
    a = _vector(vertex, start)
    b = _vector(vertex, end)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return None
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def midpoint(a: Landmark, b: Landmark) -> Landmark:
    return Landmark(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def torso_lean(shoulder_mid: Landmark, hip_mid: Landmark) -> float | None:
    """躯干相对竖直方向的前倾角（0°=直立，90°=水平）。"""
    dx = shoulder_mid.x - hip_mid.x
    dy = hip_mid.y - shoulder_mid.y  # y 向下，取向上为正
    dz = shoulder_mid.z - hip_mid.z
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return None
    horizontal = math.hypot(dx, dz)
    if horizontal < 1e-8 and abs(dy) < 1e-8:
        return None
    return math.degrees(math.atan2(horizontal, dy))


def _source(frame: PoseFrame, use_world: bool) -> list[Landmark]:
    if use_world and len(frame.world_landmarks) >= LANDMARK_COUNT:
        return frame.world_landmarks
    return frame.landmarks


def compute_angles(
    frame: PoseFrame, config: AngleConfig | None = None
) -> dict[str, float]:
    """返回该帧所有可计算角度；不可计算的关节直接不出现在结果里。"""
    config = config or AngleConfig()
    points = _source(frame, config.use_world_landmarks)
    if len(points) < LANDMARK_COUNT:
        return {}

    angles: dict[str, float] = {}
    for name, (first, vertex, last) in JOINT_ANGLE_TRIPLETS.items():
        value = angle_deg(
            points[first], points[vertex], points[last], min_visibility=config.min_visibility
        )
        if value is not None and math.isfinite(value):
            angles[name] = round(value, config.round_digits)

    shoulders = midpoint(points[LEFT_SHOULDER], points[RIGHT_SHOULDER])
    hips = midpoint(points[LEFT_HIP], points[RIGHT_HIP])
    lean = torso_lean(shoulders, hips)
    if lean is not None and math.isfinite(lean):
        angles["torso_lean"] = round(lean, config.round_digits)

    return angles


def attach_angles(
    frames: Sequence[PoseFrame], config: AngleConfig | None = None
) -> list[dict[str, float]]:
    """按帧顺序输出角度序列；无效帧返回空字典。"""
    config = config or AngleConfig()
    return [
        compute_angles(frame, config) if frame.valid else {}
        for frame in sorted(frames, key=lambda item: item.index)
    ]


def angle_series(frames: Sequence[PoseFrame], name: str, config: AngleConfig | None = None) -> list[float | None]:
    """取单一关节的角度时间序列（画曲线 / 做阈值判断用）。"""
    series: list[float | None] = []
    for angles in attach_angles(frames, config):
        series.append(angles.get(name))
    return series
