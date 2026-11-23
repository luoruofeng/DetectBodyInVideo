"""关节角度计算（任务 5.3）。

- 三点夹角（肩-肘-腕、髋-膝-踝等），输出角度制 0~180°。
- 退化输入（零长度向量、NaN、缺关键点）一律返回 ``None``，绝不产出 NaN，
  导出 JSON 时直接省略该字段。
- 躯干前倾单独用 atan2(水平位移, 竖直位移) 计算，比三点夹角更稳定；
  另输出有符号躯干俯仰角 ``torso_pitch_signed_deg``（正 = 向镜头前倾，负 = 后仰）。
- 头前倾 ``head_lean``（0°=颅骨中心正落在肩中点上方）：只看矢状面分量（不含左右），
  终点取 ``mapping.head_reference_point``（即 head_joint 的来源）。它是"头前倾"唯一
  可量化的读数 —— 没有它就只能靠肉眼在 3D 预览里判断修得对不对。
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

# NOTE: ``skeleton.mapping`` 不能在这里做模块级导入 —— ``skeleton/__init__`` 会导入
# ``skeleton.live``，而后者又导入本模块，形成环。函数内导入由 ``sys.modules`` 缓存，
# 相对每帧几十毫秒的拟合开销可以忽略。
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

ANGLE_NAMES: tuple[str, ...] = (
    *JOINT_ANGLE_TRIPLETS,
    "torso_lean",
    "torso_pitch_signed_deg",
    "head_lean",
)


class AngleConfig(BaseModel):
    use_world_landmarks: bool = Field(
        True, description="优先用米制 world landmarks（尺度无关，更稳）；缺失时自动退回归一化坐标"
    )
    min_visibility: float = Field(0.3, ge=0, lt=1)
    hold_occluded_frames: int = Field(
        15,
        ge=0,
        le=120,
        description="关节被遮挡导致角度不可计算时，最多沿用多少帧上一次有效角度；"
        "只为短时遮挡提供连续读数，超过上限继续返回缺失（0 = 关闭）",
    )
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


def torso_pitch_signed(shoulder_mid: Landmark, hip_mid: Landmark) -> float | None:
    """有符号躯干俯仰角（度）：正 = 肩比髋更靠近镜头（前倾），负 = 后仰。

    world landmarks 与归一化 z 都沿用 MediaPipe 约定 ``-z = 靠近镜头``（见
    ``postprocess/swing.py`` 的 ``forward = -thigh.z``）；侧倾（x 分量）不计入俯仰，
    因此本值可与无符号的 ``torso_lean`` 对照使用。
    """
    dx = shoulder_mid.x - hip_mid.x
    dy = hip_mid.y - shoulder_mid.y  # y 向下，取向上为正
    dz = shoulder_mid.z - hip_mid.z
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return None
    if dy <= 1e-8:
        return None
    return math.degrees(math.atan2(-dz, dy))


def head_lean(shoulder_mid: Landmark, head_center: Landmark) -> float | None:
    """头的**矢状面**前倾角（0°=颅骨中心正落在肩中点上方，90°=完全探出去）。

    只看 ``Δz``（沿视线方向）与竖直位移，**不含左右分量**：横向偏移是"歪头/侧移"，
    属于另一件事，混进来会把"头往右偏 6 cm"读成"前倾 24°"。

    正面机位下这个角度的绝对前后位置不可观测，所以只要它稳定在几度以内，就说明
    "头前倾"没有残留；一旦稳定偏大，说明头骨段长度或头部深度先验又被深度噪声带偏了
    （见 ``ik_fit.HeadDepthConfig`` 与 ``ik_fit.calibrate_bone_lengths`` 的头骨段封顶）。
    """
    dy = shoulder_mid.y - head_center.y  # y 向下，取向上为正
    dz = head_center.z - shoulder_mid.z
    if not (math.isfinite(dy) and math.isfinite(dz)):
        return None
    if abs(dy) < 1e-8 and abs(dz) < 1e-8:
        return None
    return math.degrees(math.atan2(abs(dz), dy))


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
    pitch = torso_pitch_signed(shoulders, hips)
    if pitch is not None and math.isfinite(pitch):
        angles["torso_pitch_signed_deg"] = round(pitch, config.round_digits)

    from app.backend.skeleton.mapping import head_reference_point

    head = head_reference_point(frame, use_world=config.use_world_landmarks)
    if head is not None:
        lean = head_lean(shoulders, Landmark(x=head.x, y=head.y, z=head.z))
        if lean is not None and math.isfinite(lean):
            angles["head_lean"] = round(lean, config.round_digits)

    return angles


def attach_angles(
    frames: Sequence[PoseFrame], config: AngleConfig | None = None
) -> list[dict[str, float]]:
    """按帧顺序输出角度序列；无效帧返回空字典。

    短时遮挡（关键点可见度不足）造成角度缺失时沿用上一有效读数，避免曲线突然
    断成空洞；连续遮挡超过 ``hold_occluded_frames`` 后仍然返回缺失，不长期伪造。
    """
    config = config or AngleConfig()
    results: list[dict[str, float]] = []
    last: dict[str, tuple[float, int]] = {}
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid:
            results.append({})
            continue
        current = compute_angles(frame, config)
        held = dict(current)
        updated: dict[str, tuple[float, int]] = {}
        for name in set(last) | set(current):
            if name in current:
                updated[name] = (current[name], 0)
                continue
            value, age = last[name]
            if config.hold_occluded_frames and age < config.hold_occluded_frames:
                held[name] = value
                updated[name] = (value, age + 1)
        last = updated
        results.append(held)
    return results


def angle_series(frames: Sequence[PoseFrame], name: str, config: AngleConfig | None = None) -> list[float | None]:
    """取单一关节的角度时间序列（画曲线 / 做阈值判断用）。"""
    series: list[float | None] = []
    for angles in attach_angles(frames, config):
        series.append(angles.get(name))
    return series
