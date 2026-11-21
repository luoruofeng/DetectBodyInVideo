"""摆动腿识别与踢腿相位（``docs/todo-kick-depth-accuracy.md`` §5.2 / §5.3）。

为什么需要这一层：深度方向先验（``postprocess.depth_sign``）只回答"这根骨段朝前还是
朝后"。但一段踢腿视频里两条腿的证据强度完全不同：

- **摆动腿**真的在沿矢状面运动，它的 Δz 符号有明确的物理含义，值得用更强的先验；
- **支撑腿**只是站着，Δz≈0，"往哪边"基本是噪声，强行施加方向先验只会把站直的腿掰弯。

相位同理：动作本身可预测（起脚 → 加速 → 最大伸展 → 回落），最大伸展附近是唯一
"前后方向真的在动"的时刻，此时方向先验最有价值；静止段则应收紧到默认值，避免
拿先验去对抗 2D 观测。

本模块只读 2D 轨迹与骨长，不做任何 3D 反解，因此不会引入新的深度假设。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame
from app.backend.skeleton.topology import node_map

logger = get_logger("postprocess.swing")

Side = Literal["left", "right"]

# 每侧腿的关键点（BlazePose 33 点下标），来源见 engine/base.py
LEG_INDICES: dict[str, tuple[int, int, int]] = {
    "left": (23, 25, 27),  # hip, knee, ankle
    "right": (24, 26, 28),
}

# 矢状面角度的角度约束：抬起角不会超过这个范围（超过说明检测点已经不可信）
MAX_ELEVATION_DEG = 180.0


class SwingConfig(BaseModel):
    """摆动腿 / 相位检测参数。默认开启，失败时只记 note，不影响主链。"""

    enabled: bool = Field(True, description="总开关")
    min_frames: int = Field(8, ge=3, description="少于该帧数不做判断（证据不足）")
    min_energy_ratio: float = Field(
        1.5,
        gt=1.0,
        description="摆动腿与支撑腿的 2D 速度能量比下限；低于它视为「双腿对称摆动」，退回对称处理",
    )
    phase_threshold: float = Field(
        0.15,
        gt=0,
        le=1.0,
        description="相位起止判据：2D 踝部速度超过峰值速度的该比例算「在动」；"
        "再小会被 2D 抖动触发（速度对噪声敏感，所以不能像位移阈值那样取 5%）",
    )
    peak_displacement_ratio: float = Field(
        0.95,
        gt=0,
        le=1.0,
        description="最大伸展判据：位移**首次**达到最大值的该比例即算峰值。"
        "平台期（角度保持最大）里取 argmax 会被微小漂移拖到平台末尾，取首次到达才稳",
    )


class SwingLegReport(BaseModel):
    """摆动腿识别结果（进任务摘要与质检 notes）。"""

    applied: bool = False
    swing_leg: Literal["left", "right", "none"] = "none"
    confidence: float = Field(0.0, ge=0, le=1, description="0 = 双腿对称无法区分")
    energy: dict[str, float] = Field(default_factory=dict, description="每侧 2D 速度能量（相对值）")
    frames: int = 0
    notes: list[str] = Field(default_factory=list)


class KickPhaseReport(BaseModel):
    """踢腿相位：帧号（与分析帧号一致）。检测不到时 ``detected=False``。"""

    applied: bool = False
    detected: bool = False
    side: Literal["left", "right", "none"] = "none"
    phases: dict[str, int] = Field(default_factory=dict, description="start/acceleration/peak/return")
    peak_speed: float = 0.0
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 几何：矢状面角度（与坐标 y 方向无关）
# --------------------------------------------------------------------------- #
def _world_points(frame: PoseFrame) -> dict[str, np.ndarray]:
    """世界坐标节点位置（虚节点用中点补齐），缺世界坐标时返回空。"""
    if len(frame.world_landmarks) < LANDMARK_COUNT:
        return {}
    points: dict[str, np.ndarray] = {}
    for node in node_map().values():
        if node.index is None:
            continue
        item = frame.world_landmarks[node.index]
        points[node.key] = np.array([item.x, item.y, item.z], dtype=np.float64)
    for key, parents in (
        ("pelvis", ("left_hip", "right_hip")),
        ("chest", ("left_shoulder", "right_shoulder")),
    ):
        first, second = (points.get(name) for name in parents)
        if first is not None and second is not None:
            points[key] = (first + second) / 2.0
    return points


def _down_direction(points: dict[str, np.ndarray]) -> np.ndarray:
    """身体"向下"的单位向量：由骨盆指向胸腔的反方向，再用双腿方向定符号。

    为什么不用固定的 ``±y``：不同引擎/夹具的 y 轴朝向不一致（MediaPipe 世界坐标
    ``y`` 向下，而 ``tests/synth_pose.py`` 为了直观用了相反的一侧）。躯干轴给出了
    "竖直"这条**线**，但它有两个朝向；用"两条大腿的平均方向"消歧 —— 站立/踢腿时
    支撑腿总是朝下，因此这个符号判定在真实素材和合成夹具上都成立。
    """
    pelvis = points.get("pelvis")
    chest = points.get("chest")
    down = np.array([0.0, 1.0, 0.0])
    if pelvis is not None and chest is not None:
        axis = pelvis - chest
        norm = float(np.linalg.norm(axis))
        if norm > 1e-6:
            down = axis / norm
    mean_thigh = np.zeros(3)
    for hip_key, knee_key in (
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
    ):
        hip = points.get(hip_key)
        knee = points.get(knee_key)
        if hip is None or knee is None:
            continue
        vector = knee - hip
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            mean_thigh += vector / norm
    if float(np.dot(down, mean_thigh)) < 0.0:
        down = -down
    return down


def sagittal_angles(frame: PoseFrame, side: str) -> tuple[float, float] | None:
    """返回 ``(大腿抬起角, 小腿屈曲角)``，度。

    - 抬起角：矢状面内 relative to 身体"竖直向下"的抬角，站立 0°，水平前抬 90°
      （``-z`` 约定为"靠近镜头"，与 ``engine/base.py`` 一致）；
    - 屈曲角：小腿相对**大腿延长线**的夹角，伸直 0°，屈膝为正值。

    两者都只依赖骨段方向，因此与坐标系的 y 正负、人物朝向无关。
    """
    keys = LEG_INDICES.get(side)
    if keys is None:
        return None
    points = _world_points(frame)
    if not points:
        return None
    hip_index, knee_index, ankle_index = keys
    names = {node.index: node.key for node in node_map().values() if node.index is not None}
    hip = points.get(names.get(hip_index, ""))
    knee = points.get(names.get(knee_index, ""))
    ankle = points.get(names.get(ankle_index, ""))
    if hip is None or knee is None or ankle is None:
        return None

    thigh = knee - hip
    shank = ankle - knee
    thigh_norm = float(np.linalg.norm(thigh))
    shank_norm = float(np.linalg.norm(shank))
    if thigh_norm <= 1e-9 or shank_norm <= 1e-9:
        return None

    down = _down_direction(points)
    downward = float(np.dot(thigh, down))
    forward = -float(thigh[2])  # -z = 靠近镜头
    elevation = math.degrees(math.atan2(forward, downward))
    elevation = max(-MAX_ELEVATION_DEG, min(MAX_ELEVATION_DEG, elevation))

    cosine = float(np.dot(thigh, shank) / (thigh_norm * shank_norm))
    flexion = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return elevation, flexion


# --------------------------------------------------------------------------- #
# 摆动腿识别
# --------------------------------------------------------------------------- #
def _ankle_series(frames: Sequence[PoseFrame], side: str) -> list[tuple[int, float, float]]:
    """``(帧号, u, v)`` 序列；``u = x*aspect`` 的口径无关紧要，这里只比较相对速度。"""
    indices = LEG_INDICES[side]
    ankle_index, knee_index = indices[2], indices[1]
    series: list[tuple[int, float, float]] = []
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid or len(frame.landmarks) < LANDMARK_COUNT:
            continue
        ankle = frame.landmarks[ankle_index]
        knee = frame.landmarks[knee_index]
        if ankle.visibility <= 0 or knee.visibility <= 0:
            continue
        series.append((frame.index, float(ankle.x), float(ankle.y)))
    return series


def leg_energy(frames: Sequence[PoseFrame], side: str) -> float:
    """单侧腿的 2D 运动速度能量（均方速度，单位：归一化坐标/帧）。

    只比较相对大小：摆动腿的踝/膝在画面里移动得更快，这是**不依赖深度**的判据。
    """
    series = _ankle_series(frames, side)
    if len(series) < 2:
        return 0.0
    squared: list[float] = []
    for (_, u0, v0), (_, u1, v1) in zip(series, series[1:], strict=False):
        squared.append((u1 - u0) ** 2 + (v1 - v0) ** 2)
    return float(np.mean(squared)) ** 0.5 if squared else 0.0


def identify_swing_leg(
    frames: Sequence[PoseFrame], config: SwingConfig | None = None
) -> SwingLegReport:
    """用左右踝的 2D 速度能量区分摆动腿与支撑腿；无法区分时如实返回 ``none``。"""
    config = config or SwingConfig()
    report = SwingLegReport(frames=len(frames))
    if not config.enabled:
        report.notes.append("摆动腿识别已关闭")
        return report
    if len(frames) < config.min_frames:
        report.notes.append(f"帧数不足（{len(frames)} < {config.min_frames}），不做摆动腿判断")
        return report

    energies = {side: round(leg_energy(frames, side), 6) for side in ("left", "right")}
    report.energy = energies
    left, right = energies["left"], energies["right"]
    peak = max(left, right)
    floor = min(left, right)
    if peak <= 1e-9:
        report.notes.append("双腿几乎没有 2D 运动，无法判断摆动腿")
        return report

    ratio = peak / floor if floor > 1e-9 else float("inf")
    report.swing_leg = "left" if left >= right else "right"
    report.confidence = round(min(1.0, max(0.0, 1.0 - 1.0 / max(1.0, ratio))), 4)
    if ratio < config.min_energy_ratio:
        # 双腿交替摆动 / 原地踏步：任何单侧判定都会制造单侧伪影，退回对称处理
        report.swing_leg = "none"
        report.applied = False
        report.notes.append(
            f"左右腿 2D 速度能量接近（比 {ratio:.2f} < {config.min_energy_ratio}），"
            "判定为双腿交替运动，方向先验按对称处理"
        )
        return report
    report.applied = True
    report.notes.append(
        f"摆动腿判定为 {report.swing_leg}（速度能量 左 {left:.5f} / 右 {right:.5f}，"
        f"比 {ratio:.2f}，置信度 {report.confidence:.2f}）"
    )
    return report


# --------------------------------------------------------------------------- #
# 踢腿相位
# --------------------------------------------------------------------------- #
def ankle_speed_series(
    frames: Sequence[PoseFrame], side: str
) -> tuple[list[int], np.ndarray]:
    """摆动腿踝部的逐帧 2D 速度（归一化坐标/帧）与对应的帧号。"""
    series = _ankle_series(frames, side)
    if len(series) < 2:
        return [], np.zeros(0)
    indices = [item[0] for item in series]
    speeds = [0.0]
    for (_, u0, v0), (_, u1, v1) in zip(series, series[1:], strict=False):
        speeds.append(math.hypot(u1 - u0, v1 - v0))
    return indices, np.asarray(speeds, dtype=np.float64)


def detect_kick_phases(
    frames: Sequence[PoseFrame],
    side: str | None = None,
    config: SwingConfig | None = None,
) -> KickPhaseReport:
    """从 2D 踝部轨迹检测 ``起脚 / 加速 / 最大伸展 / 回落`` 四个相位帧号。

    判据只用 2D 速度与 2D 位移，**不假设任何深度**：

    - ``start``：速度首次超过峰值 ``phase_threshold`` 的帧；
    - ``acceleration``：速度首次超过峰值 50% 的帧；
    - ``peak``：相对 ``start`` 位置 2D 位移最大的帧（踢腿伸展到最远处）；
    - ``return``：速度最后一次超过 ``phase_threshold`` 的帧。

    检测失败（帧数不足 / 没有明显运动）时返回 ``detected=False`` 并写 note，
    调用方应忽略它而不是中断分析。
    """
    config = config or SwingConfig()
    report = KickPhaseReport()
    if not config.enabled:
        report.notes.append("相位检测已关闭")
        return report

    if side is None:
        swing = identify_swing_leg(frames, config)
        if swing.swing_leg == "none":
            report.notes.append("无法确定摆动腿，跳过相位检测")
            return report
        side = swing.swing_leg
    if len(frames) < config.min_frames:
        report.notes.append(f"帧数不足（{len(frames)} < {config.min_frames}），跳过相位检测")
        return report

    indices, speeds = ankle_speed_series(frames, side)
    if len(indices) < config.min_frames:
        report.notes.append("踝部轨迹点不足，跳过相位检测")
        return report
    peak_speed = float(speeds.max())
    if peak_speed <= 1e-9:
        report.notes.append("踝部没有明显运动，跳过相位检测")
        return report

    series = _ankle_series(frames, side)
    # 原点取开头**最早几帧**的中位位置（拍摄要求开头有静止站立）：取太长的窗口会
    # 把窗口内的运动也平均进原点，导致"刚起脚"那几帧的位移被压缩、start 判晚。
    settle = min(max(2, len(series) // 20), len(series))
    origin = np.median(
        np.asarray([[item[1], item[2]] for item in series[:settle]], dtype=np.float64), axis=0
    )
    displacement = [
        float(np.hypot(item[1] - origin[0], item[2] - origin[1])) for item in series
    ]
    max_displacement = max(displacement)
    if max_displacement <= 1e-9:
        report.notes.append("踝部没有明显运动，跳过相位检测")
        return report

    # 起止用**速度**判据：正面机位下前踢的 2D 位移 ∝ 角度的平方（前后方向在画面里
    # 不可见，只有竖直分量的 (1−cosθ) 变化），用位移阈值会让起止晚 5~9 帧。
    moving = [
        position for position, value in enumerate(speeds) if value >= peak_speed * config.phase_threshold
    ]
    if len(moving) < 2:
        report.notes.append("运动持续时间过短，跳过相位检测")
        return report
    start = moving[0]
    end = moving[-1]
    half = [position for position in moving if speeds[position] >= peak_speed * 0.5]
    acceleration = half[0] if half else start

    plateau = [
        position
        for position in range(start, end + 1)
        if displacement[position] >= max_displacement * config.peak_displacement_ratio
    ]
    peak_position = plateau[0] if plateau else start + int(np.argmax(displacement[start : end + 1]))

    report.applied = True
    report.detected = True
    report.side = side  # type: ignore[assignment]
    report.peak_speed = round(peak_speed, 6)
    report.phases = {
        "start": int(indices[start]),
        "acceleration": int(indices[acceleration]),
        "peak": int(indices[peak_position]),
        "return": int(indices[end]),
    }
    report.notes.append(
        "踢腿相位：起脚 #{start} → 加速 #{acceleration} → 最大伸展 #{peak} → 回落 #{end}".format(
            start=report.phases["start"],
            acceleration=report.phases["acceleration"],
            peak=report.phases["peak"],
            end=report.phases["return"],
        )
    )
    return report


def direction_weight_scale(
    frame_index: int, phases: KickPhaseReport, *, peak_gain: float = 1.5, idle_scale: float = 0.6
) -> float:
    """按相位给方向先验的权重缩放：最大伸展附近放宽，静止段收紧。

    返回 1.0 表示用默认权重；检测失败时也返回 1.0（不改变既有行为）。
    """
    if not phases.detected:
        return 1.0
    peak = phases.phases.get("peak")
    start = phases.phases.get("start")
    end = phases.phases.get("return")
    if peak is None or start is None or end is None:
        return 1.0
    span = max(1, end - start)
    distance = abs(frame_index - peak) / span
    if distance <= 0.25:
        return peak_gain
    if frame_index < start or frame_index > end:
        return idle_scale
    return 1.0


__all__ = [
    "KickPhaseReport",
    "LEG_INDICES",
    "SwingConfig",
    "SwingLegReport",
    "ankle_speed_series",
    "detect_kick_phases",
    "direction_weight_scale",
    "identify_swing_leg",
    "leg_energy",
    "sagittal_angles",
]
