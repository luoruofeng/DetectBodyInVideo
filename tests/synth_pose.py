"""合成姿态工具（深度拟合 / yaw 测试与基准脚本共用）。

为什么自造数据：真实视频需要模型权重与素材，CI 上跑不了，也无法给出"真值"。
这里的做法是**先造真值骨架（米制世界坐标），再用已知的弱透视相机投影出 2D**，
于是每一项指标都能和真值对比：

- 骨长误差：拟合后的骨段长度 vs 真值骨长；
- 重投影误差：拟合后的 3D 反投影 vs 投影像素；
- yaw 恢复误差：估计角度 vs 实际旋转角度。

坐标约定与 MediaPipe 一致：``x`` 向图像右、``y`` 向下、``z`` 向相机后方为负
（即 ``-z`` 是"靠近相机"），原点≈两髋中点，单位米。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    Landmark,
    PoseFrame,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

ASPECT = 9 / 16  # 竖屏 1080×1920
SCALE = 0.45  # 弱透视尺度 k（1/米），人物 1.7 m 约占画面高度 76%
VISIBILITY = 0.95

# 站立姿态的真值骨架（米制，原点=两髋中点，y 向下）
# 朝向约定：人物**面向镜头**，因此人物左侧关键点落在图像右侧（x > 0），
# 这与 MediaPipe "x 向图像右" 的定义一致，yaw 的符号判定也依赖它。
_STANDING: dict[int, tuple[float, float, float]] = {
    0: (0.00, -0.70, -0.06),  # nose
    1: (0.02, -0.74, -0.08),
    2: (0.03, -0.75, -0.08),
    3: (0.04, -0.74, -0.08),
    4: (-0.02, -0.74, -0.08),
    5: (-0.03, -0.75, -0.08),
    6: (-0.04, -0.74, -0.08),
    7: (0.07, -0.74, -0.02),  # left ear
    8: (-0.07, -0.74, -0.02),  # right ear
    9: (0.02, -0.68, -0.08),
    10: (-0.02, -0.68, -0.08),
    11: (0.19, -0.45, 0.00),  # left shoulder
    12: (-0.19, -0.45, 0.00),  # right shoulder
    13: (0.20, -0.72, 0.00),  # left elbow
    14: (-0.20, -0.72, 0.00),
    15: (0.21, -0.98, 0.00),  # left wrist
    16: (-0.21, -0.98, 0.00),
    17: (0.22, -1.06, -0.01),
    18: (-0.22, -1.06, -0.01),
    19: (0.21, -1.06, -0.01),
    20: (-0.21, -1.06, -0.01),
    21: (0.23, -1.03, -0.02),
    22: (-0.23, -1.03, -0.02),
    23: (0.09, 0.00, 0.00),  # left hip
    24: (-0.09, 0.00, 0.00),  # right hip
    25: (0.09, -0.42, 0.00),  # left knee
    26: (-0.09, -0.42, 0.00),
    27: (0.09, -0.84, 0.00),  # left ankle
    28: (-0.09, -0.84, 0.00),
    29: (0.09, -0.88, 0.06),
    30: (-0.09, -0.88, 0.06),
    31: (0.09, -0.90, -0.12),  # left foot index
    32: (-0.09, -0.90, -0.12),
}

# 真值骨长（米），键为 ARKit 关节名，用于核对拟合结果
TRUE_BONE_LENGTHS: dict[str, float] = {
    "spine_7_joint": 0.45,
    "head_joint": math.dist((0.0, -0.45, 0.0), _STANDING[0]),
    "left_shoulder_1_joint": 0.19,
    "left_arm_joint": math.dist(_STANDING[11], _STANDING[13]),
    "left_hand_joint": math.dist(_STANDING[13], _STANDING[15]),
    "left_upLeg_joint": 0.09,
    "left_leg_joint": 0.42,
    "left_foot_joint": 0.42,
    "left_toes_joint": math.dist(_STANDING[27], _STANDING[31]),
}

# 右侧与左侧镜像等长，避免手写两遍
for _left, _right in (
    ("left_shoulder_1_joint", "right_shoulder_1_joint"),
    ("left_arm_joint", "right_arm_joint"),
    ("left_hand_joint", "right_hand_joint"),
    ("left_upLeg_joint", "right_upLeg_joint"),
    ("left_leg_joint", "right_leg_joint"),
    ("left_foot_joint", "right_foot_joint"),
    ("left_toes_joint", "right_toes_joint"),
):
    TRUE_BONE_LENGTHS[_right] = TRUE_BONE_LENGTHS[_left]


def standing_points() -> dict[int, np.ndarray]:
    return {index: np.array(value, dtype=np.float64) for index, value in _STANDING.items()}


def rotate_about_vertical(
    points: dict[int, np.ndarray], degrees: float
) -> dict[int, np.ndarray]:
    """绕竖直轴（y，向下）旋转，正角度表示人物左肩转向镜头（yaw > 0）。

    与 ``postprocess.ik_fit`` / ``skeleton.orientation`` 的符号约定一致：
    左肩偏移 ``(cosθ, 0, -sinθ) * 半肩宽``。
    """
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated: dict[int, np.ndarray] = {}
    for index, point in points.items():
        x, y, z = point
        rotated[index] = np.array([x * cos_t + z * sin_t, y, -x * sin_t + z * cos_t])
    return rotated


def project_to_normalized(
    points: dict[int, np.ndarray],
    *,
    aspect: float = ASPECT,
    scale: float = SCALE,
    center: tuple[float, float] = (0.5, 0.5),
) -> dict[int, tuple[float, float, float]]:
    """弱透视投影：``u = k*X + u0``、``v = k*Y + v0``，返回归一化坐标。"""
    u0, v0 = center[0] * aspect, center[1]
    projected: dict[int, tuple[float, float, float]] = {}
    for index, point in points.items():
        u = scale * float(point[0]) + u0
        v = scale * float(point[1]) + v0
        projected[index] = (u / aspect, v, scale * float(point[2]))
    return projected


def make_frames(
    count: int = 12,
    *,
    degrees: float = 0.0,
    fps: float = 30.0,
    visibility: float = VISIBILITY,
    phase_swing: float = 0.0,
) -> list[PoseFrame]:
    """生成 ``count`` 帧：可整体绕竖直轴旋转，可按正弦摆动双臂（制造真实运动）。"""
    base = standing_points()
    frames: list[PoseFrame] = []
    for index in range(count):
        phase = 2.0 * math.pi * index / max(1, count - 1)
        points = {key: value.copy() for key, value in base.items()}
        if phase_swing:
            dy = phase_swing * math.sin(phase)
            for group in ((LEFT_WRIST, (17, 19, 21)), (RIGHT_WRIST, (18, 20, 22))):
                wrist, fingers = group
                points[wrist] = points[wrist] + np.array([0.0, dy, 0.0])
                for finger in fingers:
                    points[finger] = points[finger] + np.array([0.0, dy, 0.0])
        points = rotate_about_vertical(points, degrees)
        normalized = project_to_normalized(points)
        frames.append(
            PoseFrame(
                index=index,
                timestamp_ms=round(index * 1000.0 / fps, 4),
                landmarks=[
                    Landmark(
                        x=normalized[i][0],
                        y=normalized[i][1],
                        z=normalized[i][2],
                        visibility=visibility,
                    )
                    for i in range(LANDMARK_COUNT)
                ],
                world_landmarks=[
                    Landmark(
                        x=float(points[i][0]),
                        y=float(points[i][1]),
                        z=float(points[i][2]),
                        visibility=visibility,
                    )
                    for i in range(LANDMARK_COUNT)
                ],
                score=visibility,
                valid=True,
                track_id=1,
            )
        )
    return frames


def add_depth_noise(
    frames: Sequence[PoseFrame],
    *,
    z_sigma: float = 0.04,
    xy_sigma: float = 0.012,
    seed: int = 7,
) -> list[PoseFrame]:
    """给 ``world_landmarks`` 加噪声（z 大、x/y 小），模拟单目深度误差。

    归一化 ``landmarks`` 保持"真值 2D"不动 —— 这才是优化的依据：2D 可信、3D 不可信。
    """
    rng = np.random.default_rng(seed)
    noisy: list[PoseFrame] = []
    for frame in frames:
        copy = frame.model_copy(deep=True)
        for point in copy.world_landmarks:
            point.x += float(rng.normal(0.0, xy_sigma))
            point.y += float(rng.normal(0.0, xy_sigma))
            point.z += float(rng.normal(0.0, z_sigma))
        noisy.append(copy)
    return noisy


def bone_lengths(frame: PoseFrame) -> dict[str, float]:
    """按真值拓扑量一帧的骨段长度（虚节点 pelvis/chest 用中点）。"""
    from app.backend.skeleton.topology import CHEST, PELVIS, node_map

    positions: dict[str, np.ndarray] = {}
    for key, node in node_map().items():
        if node.index is not None:
            point = frame.world_landmarks[node.index]
            positions[key] = np.array([point.x, point.y, point.z], dtype=np.float64)
    positions[PELVIS] = (positions["left_hip"] + positions["right_hip"]) / 2.0
    positions[CHEST] = (positions["left_shoulder"] + positions["right_shoulder"]) / 2.0

    lengths: dict[str, float] = {}
    for key, node in node_map().items():
        if node.parent is None:
            continue
        lengths[key] = float(np.linalg.norm(positions[key] - positions[node.parent]))
    return lengths


def bone_length_errors(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """|实测骨长 - 真值| / 真值，键为节点 key（真值缺失的段跳过）。"""
    from app.backend.skeleton.topology import node_map

    names = {node.key: node.joint for node in node_map().values()}
    result: dict[str, float] = {}
    for frame in frames:
        for key, length in bone_lengths(frame).items():
            joint = names.get(key)
            truth = TRUE_BONE_LENGTHS.get(joint or "")
            if truth is None or truth <= 1e-9:
                continue
            error = abs(length - truth) / truth
            result[key] = max(result.get(key, 0.0), error)
    return result


__all__ = [
    "ASPECT",
    "SCALE",
    "TRUE_BONE_LENGTHS",
    "add_depth_noise",
    "bone_length_errors",
    "bone_lengths",
    "make_frames",
    "project_to_normalized",
    "rotate_about_vertical",
    "standing_points",
]
