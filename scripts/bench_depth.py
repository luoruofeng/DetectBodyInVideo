"""深度（Z 轴）精度基准（任务 1.3）。

为什么要单独一套脚本：``scripts/bench.py`` 只测"推理链路吞吐"，
而这一份测的是**输出姿态的几何质量**。四项指标对应 `docs/todo-depth-z-accuracy.md` 的 1.2：

1. ``z_jitter``       相邻帧 ``world_landmarks.z`` 的 RMS（米），按 躯干/四肢/末端 分组；
2. ``bone_length_cv`` 每条骨段长度的变异系数（std/mean），基线预期 > 3%；
3. ``reproj_error_px`` 把 ``world_landmarks`` 反投影回像素后与 ``landmarks`` 的平均误差；
4. ``angle_drift_deg`` 膝/肘/髋角的帧间标准差（"静止站立"片段里越小越好）；
5. ``depth_sign_flips`` 每条骨段 ``sign(Δz)`` 的帧间翻转次数 —— 单目只剩"一个比特"
   的歧义指标，"前踢腿被解成后踢腿"就长在这个数上；
6. ``sagittal_motion_share`` 四肢顺着镜头方向伸展的帧占比，是上面那个歧义能否被
   观测到的**拍摄期**判据（越高说明越该补拍侧视机位）。

直立姿态专项（"正对镜头站立却解出躯干/头部前倾"）：

9. ``head_forward_abs_meters`` / ``torso_pitch_signed_deg`` —— ``head_joint`` 相对颈轴的
   z 偏移与躯干俯仰角。正面直立素材上两者中位数都应当 ≈ 0；头那一项一旦稳定偏离 0，
   说明 ``ik_fit.calibrate_bone_lengths`` 的头骨段解剖封顶或 ``ik_fit.head_depth`` 的
   深度目标失效了（``head_joint`` 的深度由鼻尖反推，见 ``skeleton.mapping``）。
   导出的逐帧读数同名指标是 ``angles.head_lean``（矢状面角度，个位数度数为正常）。

踢腿专项（``docs/todo-kick-depth-accuracy.md`` §1.3）：

7. ``sign_accuracy`` 与人工标注（``--labels``）比对每根骨段 ``sign(Δz)`` 的正确率，
   分别统计大腿 / 小腿 / 摆动腿 / 支撑腿；
8. ``kick_sagittal_error_deg`` 与人工标注比对大腿抬起角、小腿屈曲角的 mean / p95 误差。

用法：

    # 真实视频（走完整分析链路，指标在"后处理之后"的帧上统计）
    python scripts/bench_depth.py samples/front_full.mp4 --json out/front_full.json

    # 无模型 / 无素材时用确定性合成序列（CI 可跑，结果可重复）
    python scripts/bench_depth.py --synthetic

    # 与上一次结果对比（回归检测）
    python scripts/bench_depth.py --synthetic --baseline out/front_full.json

    # A/B：关掉本次新增的处理，看指标是否真的变化
    python scripts/bench_depth.py --synthetic --no-ik-fit --no-z-smoothing
    python scripts/bench_depth.py --synthetic --no-head-depth   # 头前探指标应当劣化

    # 踢腿验收：三档深度后端 + 人工标注
    python scripts/bench_depth.py samples/benchmark/BM-05-front-kick.mp4 \\
        --depth-sign --depth-sign-mode temporal \\
        --labels samples/benchmark/kick-labels.json --json out/bm05-temporal.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.engine.base import LANDMARK_COUNT, PoseFrame  # noqa: E402
from app.backend.postprocess.angles import JOINT_ANGLE_TRIPLETS, compute_angles  # noqa: E402
from app.backend.skeleton.topology import (  # noqa: E402
    BONE_SEGMENTS,
    CHEST,
    CORE_INDICES,
    JITTER_GROUPS,
    PELVIS,
    node_map,
)

# 回归判定的参考阈值：超过算"劣化"（只用于 --baseline 输出提示，不阻断退出码）
REGRESSION_LIMITS: dict[str, float] = {
    "z_jitter_torso": 0.25,
    "z_jitter_limb": 0.25,
    "bone_length_cv_max": 0.25,
    "reproj_error_px": 0.10,
    "angle_drift_deg": 0.15,
    "depth_sign_flips": 0.30,
    "sagittal_motion_share": 0.30,
    "sign_accuracy": 0.10,
    "kick_sagittal_error_thigh_mean_deg": 0.20,
    "kick_sagittal_error_shank_mean_deg": 0.20,
    # 理想值是 0，"越大越差"；只看绝对值，符号由 head_forward_meters 单独看
    "head_forward_abs_meters": 0.50,
    "head_forward_abs_p90_meters": 0.50,
}

# 这些指标"越大越好"，判定方向要反过来（否则符号准确率上升会被标成劣化）
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"sign_accuracy", "sign_accuracy_thigh", "sign_accuracy_shank"}
)


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def node_positions(frame: PoseFrame) -> dict[str, np.ndarray]:
    """取出所有节点位置（虚节点 pelvis/chest 用中点补齐），缺 world 时返回空。"""
    points = frame.world_landmarks
    if len(points) < LANDMARK_COUNT:
        return {}
    index_of = {node.key: node.index for node in node_map().values()}
    positions: dict[str, np.ndarray] = {}
    for key, index in index_of.items():
        if index is None:
            continue
        positions[key] = np.array([points[index].x, points[index].y, points[index].z], dtype=np.float64)
    if "left_hip" in positions and "right_hip" in positions:
        positions[PELVIS] = (positions["left_hip"] + positions["right_hip"]) / 2.0
    if "left_shoulder" in positions and "right_shoulder" in positions:
        positions[CHEST] = (positions["left_shoulder"] + positions["right_shoulder"]) / 2.0
    return positions


def estimate_image_scale(
    frame: PoseFrame, aspect: float
) -> tuple[float, float, float]:
    """弱透视投影参数 ``(k, u0, v0)``：``u = k*X + u0``、``v = k*Y + v0``。

    直接复用 ``postprocess.ik_fit`` 的实现，保证基准与生产环境用同一套相机近似。
    """
    from app.backend.postprocess.ik_fit import estimate_image_scale as _scale

    return _scale(frame, aspect)


def reprojection_error_px(
    frame: PoseFrame, *, aspect: float, width: float, height: float
) -> float | None:
    """``world_landmarks`` 反投影回图像后与 2D 关键点的平均像素误差。"""
    scale, u0, v0 = estimate_image_scale(frame, aspect)
    if scale <= 1e-9:
        return None
    errors: list[float] = []
    for index in CORE_INDICES:
        if len(frame.landmarks) < LANDMARK_COUNT or len(frame.world_landmarks) < LANDMARK_COUNT:
            return None
        world = frame.world_landmarks[index]
        norm = frame.landmarks[index]
        if norm.visibility <= 0:
            continue
        predicted_x = (scale * world.x + u0) / aspect if aspect else 0.0
        predicted_y = scale * world.y + v0
        errors.append(math.hypot((predicted_x - norm.x) * width, (predicted_y - norm.y) * height))
    if not errors:
        return None
    return float(statistics.fmean(errors))


# --------------------------------------------------------------------------- #
# 四项指标
# --------------------------------------------------------------------------- #
def z_jitter(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """相邻帧 z 的 RMS（米）。只统计前后帧都有效的点，跳帧超过 2 帧不算。"""
    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    metrics: dict[str, float] = {}
    for group, indices in JITTER_GROUPS.items():
        squared: list[float] = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.index - previous.index != 1:
                continue
            for index in indices:
                a = previous.world_landmarks[index] if len(previous.world_landmarks) > index else None
                b = current.world_landmarks[index] if len(current.world_landmarks) > index else None
                if a is None or b is None or a.visibility <= 0 or b.visibility <= 0:
                    continue
                squared.append((b.z - a.z) ** 2)
        metrics[group] = math.sqrt(statistics.fmean(squared)) if squared else 0.0
    metrics["overall"] = (
        math.sqrt(statistics.fmean([value**2 for key, value in metrics.items() if key != "overall"]))
        if metrics
        else 0.0
    )
    return {key: round(value, 6) for key, value in metrics.items()}


def xy_jitter(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """相邻帧 ``x/y`` 的 RMS（米）。

    存在的意义是任务 4.5 的对照组：z 加强平滑后这个指标**不应变差**
    （变化 < 5%），否则说明平滑把水平面精度也吃掉了。
    """
    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    metrics: dict[str, float] = {}
    for group, indices in JITTER_GROUPS.items():
        squared: list[float] = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.index - previous.index != 1:
                continue
            for index in indices:
                a = previous.world_landmarks[index] if len(previous.world_landmarks) > index else None
                b = current.world_landmarks[index] if len(current.world_landmarks) > index else None
                if a is None or b is None or a.visibility <= 0 or b.visibility <= 0:
                    continue
                squared.append((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        metrics[group] = math.sqrt(statistics.fmean(squared)) if squared else 0.0
    metrics["overall"] = (
        math.sqrt(statistics.fmean([value**2 for key, value in metrics.items() if key != "overall"]))
        if metrics
        else 0.0
    )
    return {key: round(value, 6) for key, value in metrics.items()}


def bone_lengths_per_frame(frames: Sequence[PoseFrame]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {node.key: [] for node in BONE_SEGMENTS}
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid:
            continue
        positions = node_positions(frame)
        if not positions:
            continue
        for node in BONE_SEGMENTS:
            parent = positions.get(node.parent or "")
            child = positions.get(node.key)
            if parent is None or child is None:
                continue
            series[node.key].append(float(np.linalg.norm(child - parent)))
    return series


def bone_length_cv(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """每条骨段的变异系数（std/mean）；越小说明骨长越恒定。"""
    result: dict[str, float] = {}
    for key, values in bone_lengths_per_frame(frames).items():
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if mean <= 1e-9:
            continue
        result[key] = float(np.std(values) / mean)
    return {key: round(value, 6) for key, value in result.items()}


def angle_drift(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """膝/肘/髋角的帧间标准差（度）。静止片段里它反映 Z 抖动对角度的影响。"""
    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    series: dict[str, list[float]] = {name: [] for name in JOINT_ANGLE_TRIPLETS}
    for frame in ordered:
        angles = compute_angles(frame)
        for name in series:
            if name in angles:
                series[name].append(angles[name])
    result: dict[str, float] = {}
    for name, values in series.items():
        if len(values) < 3:
            continue
        deltas = np.diff(np.asarray(values, dtype=np.float64))
        result[name] = float(np.std(deltas))
    if result:
        result["mean"] = float(np.mean(list(result.values())))
    return {key: round(value, 4) for key, value in result.items()}


def yaw_comparison(frames: Sequence[PoseFrame], *, truth_deg: float | None = None) -> dict[str, float]:
    """新旧 yaw 算法对比（任务 3.4）。

    新算法 = 肩宽/髋宽投影宽度比（``skeleton.orientation``）；
    旧算法 = ``world_landmarks`` 肩线 z 差（``orientation.yaw_from_world_z``）。
    输出两者的均值与帧间抖动（std of diff，度），以及相对真值的平均误差。
    """
    from app.backend.skeleton.orientation import estimate_yaws, yaw_from_world_z

    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    # 与生产链路一致：骨长参考取整段时间中位数，而不是逐帧自标定
    # （逐帧自标定会把该帧的深度噪声再引回参考宽度，指标会虚高）
    width = [item.yaw_rad for item in estimate_yaws(ordered)]
    depth = [value for value in (yaw_from_world_z(frame) for frame in ordered) if value is not None]
    if not width and not depth:
        return {}

    def _deg(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        degrees = [math.degrees(value) for value in values]
        jitter = float(np.std(np.diff(degrees))) if len(degrees) >= 3 else 0.0
        mean = float(np.mean(degrees))
        error = abs(mean - truth_deg) if truth_deg is not None else 0.0
        return round(mean, 3), round(jitter, 3), round(error, 3)

    width_mean, width_jitter, width_error = _deg(width)
    depth_mean, depth_jitter, depth_error = _deg(depth)
    return {
        "truth_deg": round(float(truth_deg), 3) if truth_deg is not None else 0.0,
        "width_mean_deg": width_mean,
        "width_jitter_deg": width_jitter,
        "width_error_deg": width_error,
        "depth_mean_deg": depth_mean,
        "depth_jitter_deg": depth_jitter,
        "depth_error_deg": depth_error,
    }


def depth_sign_flips(frames: Sequence[PoseFrame]) -> dict[str, Any]:
    """每条骨段 ``sign(Δz)`` 的帧间翻转次数（单目 1-bit 歧义的直接指标）。

    为什么单列这一项：弱透视下骨长与 2D 投影长度把 ``|Δz|`` 定死，**只剩符号**。
    正确还原的片段里，除真正的方向反转外翻转次数应接近 0；"腿朝反方向踢"这类
    故障的表现就是同一骨段在一段动作内反复翻面。
    """
    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    per_key: dict[str, int] = {}
    for node in BONE_SEGMENTS:
        signs: list[int] = []
        for frame in ordered:
            positions = node_positions(frame)
            if not positions:
                continue
            parent = positions.get(node.parent or "")
            child = positions.get(node.key)
            if parent is None or child is None:
                continue
            dz = float(child[2] - parent[2])
            if abs(dz) < 1e-3:
                continue
            signs.append(1 if dz > 0 else -1)
        if not signs:
            continue
        per_key[node.key] = sum(
            1 for a, b in zip(signs, signs[1:], strict=False) if a != b
        )
    return {
        "total": int(sum(per_key.values())),
        "active_segments": int(sum(1 for value in per_key.values() if value)),
        "worst": int(max(per_key.values())) if per_key else 0,
        "per_segment": per_key,
    }


def sagittal_motion_share(
    frames: Sequence[PoseFrame], *, aspect: float, threshold: float = 0.5
) -> float:
    """四肢骨段"顺着镜头方向"的帧占比（0~1）。复用质检同口径实现，保证数字可比。"""
    from app.backend.postprocess.quality import sagittal_motion_share as _share

    return round(_share(frames, aspect=aspect, threshold=threshold), 4)


# --------------------------------------------------------------------------- #
# 踢腿专项：人工标注真值（§1.2 / §1.3）
# --------------------------------------------------------------------------- #
# 标注文件契约（``samples/benchmark/kick-labels.json``）：
#
#   {
#     "video": "BM-06-side-full.mp4",
#     "fps": 30,
#     "swingLeg": "left",
#     "frames": [
#       {"index": 0, "thighDeg": 8.0, "shankDeg": 3.0,
#        "signs": {"left_knee": -1, "left_ankle": -1},
#        "phase": "start"},
#       ...
#     ],
#     "phases": {"start": 0, "peak": 18, "return": 34}
#   }
#
# - ``thighDeg``：大腿在矢状面内相对"竖直向下"的抬起角（0 = 站立，90 = 水平前抬）；
# - ``shankDeg``：小腿相对大腿延长线的屈曲角（0 = 伸直，正值 = 屈膝）；
# - ``signs``：该帧每根骨段 ``sign(Δz)`` 的真值，**-1 = 子关节更靠近镜头**（踢向前方）；
# - ``phases``：踢腿相位帧号（供 §5.3 的相位检测对比）。

LEG_KEYS: dict[str, dict[str, str]] = {
    "left": {"hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle"},
    "right": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle"},
}


def load_labels(path: Path | str) -> dict[str, Any]:
    """读标注文件；缺文件时抛 ``SystemExit``（基准脚本宁可报错也不静默跳过真值）。"""
    label_path = Path(path)
    if not label_path.is_file():
        raise SystemExit(f"找不到标注文件：{label_path}")
    return json.loads(label_path.read_text(encoding="utf-8"))


def sagittal_angles(frame: PoseFrame, side: str) -> tuple[float, float] | None:
    """返回 ``(大腿抬起角, 小腿屈曲角)``，度。委托生产实现，保证基准与链路同口径。"""
    from app.backend.postprocess.swing import sagittal_angles as _angles

    return _angles(frame, side)


def bone_signs(frame: PoseFrame, side: str) -> dict[str, int]:
    """该侧大腿（膝）与小腿（踝）的 ``sign(Δz)``，键与标注文件的 ``signs`` 一致。"""
    keys = LEG_KEYS.get(side)
    if keys is None:
        return {}
    positions = node_positions(frame)
    result: dict[str, int] = {}
    for child, parent in ((keys["knee"], keys["hip"]), (keys["ankle"], keys["knee"])):
        child_point = positions.get(child)
        parent_point = positions.get(parent)
        if child_point is None or parent_point is None:
            continue
        dz = float(child_point[2] - parent_point[2])
        if abs(dz) < 1e-4:
            continue
        result[child] = 1 if dz > 0 else -1
    return result


def _label_frames(labels: dict[str, Any]) -> list[dict[str, Any]]:
    frames = labels.get("frames")
    return [item for item in frames if isinstance(item, dict)] if isinstance(frames, list) else []


def sign_accuracy(
    frames: Sequence[PoseFrame], labels: dict[str, Any]
) -> dict[str, Any]:
    """``sign(Δz)`` 与标注一致的样本占比，分别统计大腿 / 小腿 / 摆动腿 / 支撑腿。

    统计对象是**后处理之后、即将导出的帧**（而不是中间产物），因为消费方看到的
    就是这份结果：符号判错的直接后果就是"腿朝反方向踢"。
    """
    swing = str(labels.get("swingLeg") or "left").lower()
    if swing not in LEG_KEYS:
        swing = "left"
    by_index = {frame.index: frame for frame in frames if frame.valid}

    groups: dict[str, list[int]] = {
        "thigh": [0, 0],
        "shank": [0, 0],
        "swingLeg": [0, 0],
        "supportLeg": [0, 0],
        "overall": [0, 0],
    }
    per_segment: dict[str, list[int]] = {}

    for item in _label_frames(labels):
        index = item.get("index")
        truth = item.get("signs")
        if not isinstance(index, int) or not isinstance(truth, dict):
            continue
        frame = by_index.get(index)
        if frame is None:
            continue
        for side in LEG_KEYS:
            observed = bone_signs(frame, side)
            for key, expected in truth.items():
                if key not in observed or not isinstance(expected, int) or expected == 0:
                    continue
                hit = 1 if observed[key] == expected else 0
                group = "thigh" if key.endswith("knee") else "shank"
                groups[group][0] += hit
                groups[group][1] += 1
                groups["swingLeg" if side == swing else "supportLeg"][0] += hit
                groups["swingLeg" if side == swing else "supportLeg"][1] += 1
                groups["overall"][0] += hit
                groups["overall"][1] += 1
                bucket = per_segment.setdefault(key, [0, 0])
                bucket[0] += hit
                bucket[1] += 1

    def _ratio(pair: list[int]) -> float | None:
        return round(pair[0] / pair[1], 4) if pair[1] else None

    result: dict[str, Any] = {
        "samples": groups["overall"][1],
        "overall": _ratio(groups["overall"]),
        "byGroup": {
            key: _ratio(value) for key, value in groups.items() if key != "overall"
        },
        "bySegment": {key: _ratio(value) for key, value in sorted(per_segment.items())},
        "swingLeg": swing,
    }
    return result if groups["overall"][1] else {}


def kick_sagittal_error_deg(
    frames: Sequence[PoseFrame], labels: dict[str, Any]
) -> dict[str, Any]:
    """大腿抬起角 / 小腿屈曲角与标注的绝对误差（mean / p95，度）。"""
    swing = str(labels.get("swingLeg") or "left").lower()
    if swing not in LEG_KEYS:
        swing = "left"
    by_index = {frame.index: frame for frame in frames if frame.valid}
    errors: dict[str, list[float]] = {"thigh": [], "shank": []}

    for item in _label_frames(labels):
        index = item.get("index")
        if not isinstance(index, int):
            continue
        frame = by_index.get(index)
        if frame is None:
            continue
        angles = sagittal_angles(frame, swing)
        if angles is None:
            continue
        for name, estimate, key in (
            ("thigh", angles[0], "thighDeg"),
            ("shank", angles[1], "shankDeg"),
        ):
            truth = item.get(key)
            if not isinstance(truth, (int, float)):
                continue
            errors[name].append(abs(float(estimate) - float(truth)))

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p95": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": round(float(array.mean()), 3),
            "p95": round(float(np.percentile(array, 95)), 3),
        }

    if not errors["thigh"] and not errors["shank"]:
        return {}
    return {
        "swingLeg": swing,
        "frames": max(len(errors["thigh"]), len(errors["shank"])),
        "thigh": _summary(errors["thigh"]),
        "shank": _summary(errors["shank"]),
    }


def kick_phase_error(
    frames: Sequence[PoseFrame],
    labels: dict[str, Any],
    *,
    detected_phases: dict[str, int] | None = None,
) -> dict[str, Any]:
    """踢腿相位检测（§5.3）与标注相位的帧差；检测不可用或失败时返回空。

    ``detected_phases`` 优先用链路自己给出的相位（基于**平滑前**的 2D 轨迹）。
    在输出帧上重算会把平滑带来的 4~5 帧滞后记到检测头上，指标就失去意义了。
    """
    truth = labels.get("phases")
    if not isinstance(truth, dict) or not truth:
        return {}
    if detected_phases:
        detected_phases = {key: int(value) for key, value in detected_phases.items()}
    else:
        try:
            from app.backend.postprocess.swing import detect_kick_phases

            detected_phases = detect_kick_phases(frames).phases
        except Exception:  # pragma: no cover - 模块缺失/检测失败都不影响基准
            return {}
    result: dict[str, Any] = {}
    for key, frame_index in truth.items():
        estimated = detected_phases.get(key)
        if not isinstance(frame_index, int) or estimated is None:
            continue
        result[key] = abs(int(estimated) - int(frame_index))
    if not result:
        return {}
    return {"frameError": result, "maxFrameError": max(result.values())}


def head_torso_alignment(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """直立姿态检查：头相对颈轴的深度偏移、躯干俯仰角。

    单列这一项的原因：正面机位下**头的前后位置完全没有观测**，它只能靠先验落在
    颈正上方；这个先验由「头骨段解剖封顶（``ik_fit.calibrate_bone_lengths``）+
    ``ik_fit.head_depth.target_forward_meters`` + ``mapping`` 里由鼻尖反推 head_joint 深度」
    三层共同表达，任何一层失效这里都会稳定偏离 0，而不是像其它指标那样只是抖动变大。

    - ``head_forward_meters``：``head_joint.z - neck_1_joint.z`` 的中位数（负 = 头朝镜头探）；
    - ``head_forward_abs_meters``：同上取绝对值后的中位数，正面直立素材应当 < 0.02；
    - ``head_forward_abs_p90_meters``：绝对值 p90，用来看"偶尔几帧突然前探"；
    - ``torso_pitch_signed_deg``：躯干俯仰中位数（正 = 肩比髋更靠镜头）。

    取的是 ``skeleton.mapping`` 的输出而不是 world_landmarks，因为 3D 预览/导出的
    ``head_joint`` 就是那一层算出来的 —— 指标必须和用户看到的东西同源。
    """
    from app.backend.skeleton.mapping import map_frame

    forward: list[float] = []
    pitch: list[float] = []
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        skeleton = map_frame(frame, use_world=True)
        head = skeleton.joints.get("head_joint")
        neck = skeleton.joints.get("neck_1_joint")
        if head is not None and neck is not None and math.isfinite(head.z - neck.z):
            forward.append(head.z - neck.z)
        value = compute_angles(frame).get("torso_pitch_signed_deg")
        if value is not None and math.isfinite(value):
            pitch.append(float(value))
    if not forward:
        return {
            "head_forward_meters": 0.0,
            "head_forward_abs_meters": 0.0,
            "head_forward_abs_p90_meters": 0.0,
            "torso_pitch_signed_deg": 0.0,
            "frames": 0,
        }
    values = np.asarray(forward, dtype=np.float64)
    return {
        "head_forward_meters": round(float(np.median(values)), 6),
        "head_forward_abs_meters": round(float(np.median(np.abs(values))), 6),
        "head_forward_abs_p90_meters": round(
            float(np.percentile(np.abs(values), 90)), 6
        ),
        "torso_pitch_signed_deg": (
            round(float(np.median(pitch)), 4) if pitch else 0.0
        ),
        "frames": len(values),
    }


def reprojection_summary(
    frames: Sequence[PoseFrame], *, aspect: float, width: float, height: float
) -> dict[str, float]:
    per_frame = [
        value
        for value in (
            reprojection_error_px(frame, aspect=aspect, width=width, height=height)
            for frame in sorted(frames, key=lambda item: item.index)
            if frame.valid
        )
        if value is not None
    ]
    if not per_frame:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    values = np.asarray(per_frame, dtype=np.float64)
    return {
        "mean": round(float(values.mean()), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(values.max()), 4),
    }


# --------------------------------------------------------------------------- #
# 跑链路
# --------------------------------------------------------------------------- #
def _synthetic_frames(
    z_noise: float = 0.0, degrees: float = 0.0
) -> tuple[list[PoseFrame], int, int, float]:
    """确定性合成序列（几何自洽：3D 真值骨架 + 已知弱透视相机投影出 2D）。

    这里用 ``tests/synth_pose.py`` 而不是 ``export/samples.py`` 的黄金样例：
    后者是"形状回归"用的固定夹具，它的 2D 与 3D 不满足同一个相机模型，
    拿它算重投影误差没有物理意义。
    """
    tests_dir = PROJECT_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import synth_pose

    fps = 30.0
    frames = synth_pose.make_frames(30, fps=fps, phase_swing=0.12, degrees=degrees)
    if z_noise > 0:
        # 深度噪声比横向噪声大一个量级：单目深度本来就差得多
        frames = synth_pose.add_depth_noise(frames, z_sigma=z_noise, xy_sigma=z_noise * 0.3)
    return frames, 1080, 1920, fps


def _synthetic_kick_frames(
    peak_deg: float = 80.0, z_noise: float = 0.0, corrupt: bool = False
) -> tuple[list[PoseFrame], int, int, float, dict[str, Any]]:
    """合成"前踢腿"序列 + 同源真值标注（没有真实素材时的替代基线）。

    真值来自**投影前的 3D 骨架**，因此 ``sign_accuracy`` / 矢状面角误差在这里
    是物理上有意义的数字，而不是"自己和自己比"。``corrupt=True`` 会按真实故障形态
    把部分帧的腿整体镜像（2D 与骨长都不变、只有 Δz 反号），用来验证指标能否抓到
    "腿翻面"。
    """
    tests_dir = PROJECT_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import synth_pose

    fps = 30.0
    # 三角速度剖面（没有平台段）：相位唯一，才能用"帧差"验收 §5.3 的检测精度
    frames = synth_pose.make_kick_frames(60, peak_deg=peak_deg, fps=fps, ramp=0.5)
    labels = synth_pose.kick_labels(frames)
    if z_noise > 0:
        frames = synth_pose.add_depth_noise(frames, z_sigma=z_noise, xy_sigma=z_noise * 0.3)
    if corrupt:
        from app.backend.engine.base import (
            LEFT_ANKLE,
            LEFT_FOOT_INDEX,
            LEFT_HEEL,
            LEFT_HIP,
            LEFT_KNEE,
        )

        for position, frame in enumerate(frames):
            if position % 7 != 3:
                continue
            hip_z = frame.world_landmarks[LEFT_HIP].z
            for index in (LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX):
                point = frame.world_landmarks[index]
                point.z = 2.0 * hip_z - point.z
    return frames, 1080, 1920, fps, labels


def _analyze(
    path: Path, options: Any
) -> tuple[list[PoseFrame], int, int, float, dict[str, Any]]:
    from app.backend.pipeline import AnalysisOptions, analyze_video

    result = analyze_video(path, options=AnalysisOptions(postprocess=options))
    plan = result.sampling
    # 采样计划里没有画幅，指标又要按像素算（重投影误差），所以直接读一遍容器。
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1080
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 1920
    capture.release()
    fit = getattr(result.postprocess, "fit", None)
    extra = {
        "engine": result.selection.engine,
        "engineDisplayName": result.selection.display_name,
        "verdict": result.postprocess.quality.verdict,
        "fitMsPerFrame": getattr(fit, "milliseconds_per_frame", None),
        "depthSignApplied": result.postprocess.depth_sign.applied,
        "depthSignMode": result.postprocess.depth_sign.mode,
        "swingLeg": getattr(result.postprocess, "swing", None)
        and result.postprocess.swing.swing_leg,
        "kickPhases": dict(getattr(result.postprocess.kick_phases, "phases", {}) or {}),
    }
    return (
        result.postprocess.frames,
        int(width),
        int(height),
        float(getattr(plan, "effective_fps", 0.0) or 30.0),
        extra,
    )


def build_postprocess_config(args: argparse.Namespace, *, stride: int, fps: float) -> Any:
    """按 A/B 开关装配后处理配置。

    对"还没有 ik_fit / 逐轴 z 参数"的旧代码保持兼容（``hasattr`` 判断），
    这样同一份脚本既能采基线，也能采优化后的结果。
    """
    from app.backend.postprocess.pipeline import PostProcessConfig

    postprocess = PostProcessConfig(stride=stride, fps=fps)
    if hasattr(postprocess, "ik_fit"):
        fit_update: dict[str, Any] = {"enabled": not args.no_ik_fit}
        # §4 的权重扫描：0 = 只有符号（hinge），>0 才启用方向向量先验
        direction_weight = getattr(args, "direction_weight", 0.0)
        if direction_weight and hasattr(postprocess.ik_fit, "depth_direction_weight"):
            fit_update["depth_direction_weight"] = float(direction_weight)
        sparsity = getattr(args, "direction_sparsity", 0.0)
        if sparsity and hasattr(postprocess.ik_fit, "depth_sparsity_weight"):
            fit_update["depth_sparsity_weight"] = float(sparsity)
        # 二阶（加速度）时域约束的 A/B：默认值随 FitConfig 走，显式传入才覆盖
        accel = getattr(args, "accel_weight", None)
        if accel is not None and hasattr(postprocess.ik_fit, "acceleration_weight"):
            fit_update["acceleration_weight"] = float(accel)
        # 头部专项 A/B：关掉后 head_forward_* 应当明显劣化（回到"直立也有几厘米前探"）
        if getattr(args, "no_head_depth", False) and hasattr(postprocess.ik_fit, "head_depth"):
            fit_update["head_depth"] = postprocess.ik_fit.head_depth.model_copy(
                update={"enabled": False}
            )
        postprocess = postprocess.model_copy(
            update={"ik_fit": postprocess.ik_fit.model_copy(update=fit_update)}
        )

    smoothing = postprocess.smoothing
    one_euro = smoothing.one_euro
    world_one_euro = getattr(smoothing, "world_one_euro", None)
    if args.no_z_smoothing and hasattr(one_euro, "min_cutoff_z"):
        disabled = {"min_cutoff_z": None, "beta_z": None, "d_cutoff_z": None}
        updates: dict[str, Any] = {"one_euro": one_euro.model_copy(update=disabled)}
        # world_landmarks 那一趟也要一起还原，否则 z 抖动指标看不出差别（指标就长在世界点上）
        if world_one_euro is not None and hasattr(world_one_euro, "min_cutoff_z"):
            updates["world_one_euro"] = world_one_euro.model_copy(update=disabled)
        smoothing = smoothing.model_copy(update=updates)
    if args.no_smoothing:
        smoothing = smoothing.model_copy(update={"method": "none"})
    # 零相位（前向+后向）世界点平滑的 A/B
    zero_phase = getattr(args, "zero_phase", None)
    if zero_phase is not None and hasattr(smoothing, "zero_phase_world"):
        smoothing = smoothing.model_copy(update={"zero_phase_world": bool(zero_phase)})
    mode = getattr(args, "depth_sign_mode", None)
    if getattr(args, "depth_sign", False) and hasattr(postprocess, "depth_sign"):
        update: dict[str, Any] = {"enabled": True}
        # §1.4 的三档基线必须能选 ``model``，否则"temporal vs model"根本采不出来。
        # 这里显式写 mode（即使等于默认值）会被 ``pipeline.py`` 的合并规则覆盖，
        # 所以顺带把服务级配置也钉住（见 ``_pin_settings_for_run``）。
        if mode:
            update["mode"] = mode
        postprocess = postprocess.model_copy(
            update={"depth_sign": postprocess.depth_sign.model_copy(update=update)}
        )
    elif mode and hasattr(postprocess, "depth_sign"):
        postprocess = postprocess.model_copy(
            update={"depth_sign": postprocess.depth_sign.model_copy(update={"mode": mode})}
        )
    return postprocess.model_copy(update={"smoothing": smoothing})


def _pin_settings_for_run(args: argparse.Namespace) -> None:
    """把 ``--depth-sign-mode`` / ``--lifter`` 钉到服务级配置上。

    ``pipeline.py`` 的合并规则是"前端传的 mode 等于默认值 temporal 时用 settings 覆盖"，
    基准脚本直接构造配置，等于前端传默认值 —— 不钉住 settings 就选不到 ``model``。
    """
    import os

    changed = False
    mode = getattr(args, "depth_sign_mode", None)
    if mode:
        os.environ["DETECT_BODY_DEPTH_SIGN_MODE"] = mode
        changed = True
    lifter = getattr(args, "lifter", None)
    if lifter:
        os.environ["DETECT_BODY_LIFTER"] = lifter
        changed = True
    if getattr(args, "depth_sign", False):
        os.environ["DETECT_BODY_DEPTH_SIGN_ENABLED"] = "1"
        changed = True
    if changed:
        from app.backend.core.settings import reset_settings_cache

        reset_settings_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    generated_labels: dict[str, Any] | None = None
    if getattr(args, "synthetic_kick", False):
        frames, width, height, fps, generated_labels = _synthetic_kick_frames(
            peak_deg=getattr(args, "kick_peak", 80.0),
            z_noise=getattr(args, "kick_noise", 0.0),
            corrupt=getattr(args, "corrupt_signs", False),
        )
        from app.backend.postprocess.pipeline import run_postprocess

        started = time.perf_counter()
        postprocess = run_postprocess(
            frames, build_postprocess_config(args, stride=1, fps=fps)
        )
        elapsed = time.perf_counter() - started
        frames = postprocess.frames
        fit = getattr(postprocess, "fit", None)
        extra = {
            "engine": "synthetic-kick",
            "engineDisplayName": "确定性合成前踢腿（tests/synth_pose.py）",
            "verdict": postprocess.quality.verdict,
            "fitMsPerFrame": getattr(fit, "milliseconds_per_frame", None),
            "depthSignApplied": postprocess.depth_sign.applied,
            "depthSignMode": postprocess.depth_sign.mode,
            "swingLeg": getattr(postprocess, "swing", None)
            and postprocess.swing.swing_leg,
            # 相位由链路自己给出（基于平滑前的 2D 轨迹），不要在输出帧上重算：
            # 平滑会把动作起点抹开 4~5 帧，重算等于把平滑滞后记到检测头上
            "kickPhases": dict(getattr(postprocess.kick_phases, "phases", {}) or {}),
        }
        timing = {
            "postprocessSeconds": round(elapsed, 4),
            "framesPerSecond": round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0,
        }
    elif args.synthetic:
        frames, width, height, fps = _synthetic_frames(args.z_noise, args.rotation)
        from app.backend.postprocess.pipeline import run_postprocess

        started = time.perf_counter()
        postprocess = run_postprocess(
            frames, build_postprocess_config(args, stride=1, fps=fps)
        )
        elapsed = time.perf_counter() - started
        frames = postprocess.frames
        fit = getattr(postprocess, "fit", None)
        extra = {
            "engine": "synthetic",
            "engineDisplayName": "确定性合成序列（tests/synth_pose.py）",
            "verdict": postprocess.quality.verdict,
            "fitMsPerFrame": getattr(fit, "milliseconds_per_frame", None),
            "depthSignApplied": postprocess.depth_sign.applied,
        }
        timing = {
            "postprocessSeconds": round(elapsed, 4),
            "framesPerSecond": round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0,
        }
    else:
        if not args.video:
            raise SystemExit("请给出视频路径，或加 --synthetic")
        path = Path(args.video)
        if not path.is_file():
            raise SystemExit(f"找不到视频：{path}")
        started = time.perf_counter()
        postprocess_config = build_postprocess_config(
            args, stride=max(1, int(getattr(args, "every", 1) or 1)), fps=30.0
        )
        frames, width, height, fps, extra = _analyze(path, options=postprocess_config)
        elapsed = time.perf_counter() - started
        timing = {
            "analysisSeconds": round(elapsed, 4),
            "framesPerSecond": round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0,
        }

    aspect = width / height if height else 9 / 16
    cv = bone_length_cv(frames)
    labels = load_labels(args.labels) if getattr(args, "labels", "") else generated_labels
    metrics = {
        "label": args.label
        or (
            "synthetic-kick"
            if getattr(args, "synthetic_kick", False)
            else ("synthetic" if args.synthetic else Path(args.video).stem)
        ),
        "source": (
            "synthetic-kick"
            if getattr(args, "synthetic_kick", False)
            else ("synthetic" if args.synthetic else str(args.video))
        ),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 3),
        "frames": len(frames),
        "depth_sign_mode": getattr(args, "depth_sign_mode", None) or "off",
        "z_jitter": z_jitter(frames),
        "xy_jitter": xy_jitter(frames),
        "bone_length_cv": cv,
        "bone_length_cv_max": round(max(cv.values()), 6) if cv else 0.0,
        "bone_length_cv_mean": round(float(np.mean(list(cv.values()))), 6) if cv else 0.0,
        "reproj_error_px": reprojection_summary(
            frames, aspect=aspect, width=width, height=height
        ),
        "angle_drift_deg": angle_drift(frames),
        "depth_sign_flips": depth_sign_flips(frames),
        "sagittal_motion_share": sagittal_motion_share(frames, aspect=aspect),
        "head_torso": head_torso_alignment(frames),
        "timing": timing,
        "env": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
        },
        "ik_fit_enabled": not args.no_ik_fit,
        "yaw": yaw_comparison(frames, truth_deg=args.rotation if args.synthetic else None),
    }
    if labels is not None:
        metrics["labels"] = str(args.labels) if getattr(args, "labels", "") else "generated:synthetic-kick"
        metrics["sign_accuracy"] = sign_accuracy(frames, labels)
        metrics["kick_sagittal_error_deg"] = kick_sagittal_error_deg(frames, labels)
        metrics["kick_phase_error"] = kick_phase_error(
            frames, labels, detected_phases=extra.get("kickPhases")
        )
    metrics.update({f"info_{key}": value for key, value in extra.items()})
    if getattr(args, "write_labels", ""):
        draft = _label_draft(
            frames,
            labels=labels,
            video="synthetic-kick" if getattr(args, "synthetic_kick", False) else Path(args.video).name,
        )
        target = Path(args.write_labels)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n标注草稿已写入 {target}（status={draft.get('status')}，需人工修正后使用）")
    return metrics


def _label_draft(
    frames: Sequence[PoseFrame], *, labels: dict[str, Any] | None, video: str
) -> dict[str, Any]:
    """生成标注草稿：先跑一遍链路，再人工修正（§1.2 允许的半自动流程）。

    角度与符号都来自**本次输出**，所以它是"待修正的草稿"而不是真值 ——
    ``status`` 字段写明这一点，避免后来的人把它当真值用。
    """
    if labels is not None and labels.get("frames"):
        draft = dict(labels)
        draft.setdefault("status", "generated")
        draft["video"] = video
        return draft

    tests_dir = PROJECT_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import synth_pose

    from app.backend.postprocess.swing import identify_swing_leg

    side = identify_swing_leg(frames).swing_leg
    draft = synth_pose.kick_labels(frames, side="right" if side == "right" else "left")
    draft["video"] = video
    draft["status"] = "auto-draft"
    draft["note"] = (
        "自动草稿：角度与符号来自本次分析输出，逐帧人工核对后把 status 改为 annotated 再当真值用"
    )
    return draft


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _flat(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for group, value in (metrics.get("z_jitter") or {}).items():
        flat[f"z_jitter_{group}"] = float(value)
    xy = metrics.get("xy_jitter") or {}
    if xy.get("overall") is not None:
        flat["xy_jitter_overall"] = float(xy["overall"])
    for key in ("bone_length_cv_max", "bone_length_cv_mean"):
        flat[key] = float(metrics.get(key) or 0.0)
    flat["reproj_error_px"] = float((metrics.get("reproj_error_px") or {}).get("mean", 0.0))
    flat["angle_drift_deg"] = float((metrics.get("angle_drift_deg") or {}).get("mean", 0.0))
    flat["depth_sign_flips"] = float((metrics.get("depth_sign_flips") or {}).get("total", 0))
    flat["sagittal_motion_share"] = float(metrics.get("sagittal_motion_share") or 0.0)
    head = metrics.get("head_torso") or {}
    for key in (
        "head_forward_meters",
        "head_forward_abs_meters",
        "head_forward_abs_p90_meters",
        "torso_pitch_signed_deg",
    ):
        if head.get(key) is not None:
            flat[key] = float(head[key])
    yaw = metrics.get("yaw") or {}
    for key, source in (
        ("yaw_error_width_deg", "width_error_deg"),
        ("yaw_jitter_width_deg", "width_jitter_deg"),
        ("yaw_jitter_depth_deg", "depth_jitter_deg"),
    ):
        if yaw.get(source) is not None:
            flat[key] = float(yaw[source])
    # 踢腿专项：只有提供了 --labels 才有数字（没标注就没有真值，宁可缺项也不填 0）
    accuracy = metrics.get("sign_accuracy") or {}
    if accuracy.get("overall") is not None:
        flat["sign_accuracy"] = float(accuracy["overall"])
    for group in ("thigh", "shank", "swingLeg", "supportLeg"):
        value = (accuracy.get("byGroup") or {}).get(group)
        if value is not None:
            flat[f"sign_accuracy_{group}"] = float(value)
    sagittal = metrics.get("kick_sagittal_error_deg") or {}
    for joint in ("thigh", "shank"):
        block = sagittal.get(joint) or {}
        if block:
            flat[f"kick_sagittal_error_{joint}_mean_deg"] = float(block.get("mean", 0.0))
            flat[f"kick_sagittal_error_{joint}_p95_deg"] = float(block.get("p95", 0.0))
    phase = metrics.get("kick_phase_error") or {}
    if phase.get("maxFrameError") is not None:
        flat["kick_phase_max_frame_error"] = float(phase["maxFrameError"])
    return flat


def print_table(metrics: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    flat = _flat(metrics)
    base = _flat(baseline) if baseline else {}
    print(f"\n=== 深度精度指标：{metrics['label']} ===")
    print(f"来源 {metrics['source']} · {metrics['resolution']} · {metrics['fps']} fps · {metrics['frames']} 帧")
    header = f"{'指标':<26}{'本次':>12}"
    if base:
        header += f"{'基线':>12}{'变化':>12}{'判定':>8}"
    print(header)
    print("-" * (len(header) + 8))
    for key, value in flat.items():
        row = f"{key:<26}{value:>12.5f}"
        if base:
            previous = base.get(key, 0.0)
            delta = value - previous
            ratio = (delta / previous * 100.0) if previous > 1e-12 else 0.0
            limit = REGRESSION_LIMITS.get(key)
            verdict = "—"
            if limit is not None and previous > 1e-12:
                worse = ratio < -limit * 100 if key in HIGHER_IS_BETTER else ratio > limit * 100
                verdict = "劣化" if worse else "ok"
            row += f"{previous:>12.5f}{ratio:>11.1f}%{verdict:>8}"
        print(row)
    if metrics.get("timing"):
        print("\n耗时：" + " · ".join(f"{key}={value}" for key, value in metrics["timing"].items()))
    info = {key[5:]: value for key, value in metrics.items() if key.startswith("info_")}
    if info:
        print("链路：" + " · ".join(f"{key}={value}" for key, value in info.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="深度（Z 轴）精度基准")
    parser.add_argument("video", nargs="?", help="待测视频；配合 --synthetic 时可省略")
    parser.add_argument("--synthetic", action="store_true", help="用确定性合成姿态序列，不需要模型")
    parser.add_argument(
        "--synthetic-kick",
        action="store_true",
        help="用确定性合成的「前踢腿」序列（含同源真值标注），可在无素材时验证符号准确率与矢状面角误差",
    )
    parser.add_argument(
        "--kick-peak",
        type=float,
        default=80.0,
        help="合成踢腿的最大抬腿角（度），默认 80",
    )
    parser.add_argument(
        "--kick-noise",
        type=float,
        default=0.0,
        help="合成踢腿模式下给 world z 叠加的高斯噪声（米），默认 0（纯几何真值）",
    )
    parser.add_argument(
        "--corrupt-signs",
        action="store_true",
        help="合成踢腿模式下按真实故障形态镜像部分帧（2D 与骨长不变、只有 Δz 反号），"
        "用来确认指标能抓到「腿翻面」",
    )
    parser.add_argument(
        "--z-noise",
        type=float,
        default=0.02,
        help="合成模式下给 world z 叠加的高斯噪声标准差（米），默认 0.02",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="合成模式下整体绕竖直轴旋转的角度（度），正值为左肩转向镜头；默认 0",
    )
    parser.add_argument("--engine", default="auto", choices=["auto", "mediapipe", "onnx"])
    parser.add_argument("--label", default="", help="结果标签（默认用文件名）")
    parser.add_argument("--json", default="", help="把本次结果写到该 JSON 路径")
    parser.add_argument("--baseline", default="", help="上一次结果的 JSON，用于 diff 回归检测")
    parser.add_argument("--no-ik-fit", action="store_true", help="关闭骨长约束拟合（A/B）")
    parser.add_argument(
        "--no-head-depth",
        action="store_true",
        help="关闭头部深度专项（A/B）：head_forward_abs_meters 应当明显变大",
    )
    parser.add_argument("--no-z-smoothing", action="store_true", help="关闭 z 轴强平滑（A/B）")
    parser.add_argument("--no-smoothing", action="store_true", help="完全关闭平滑（A/B）")
    parser.add_argument(
        "--depth-sign",
        action="store_true",
        help="开启深度符号先验（时序一致性），对比 depth_sign_flips 前后的变化",
    )
    parser.add_argument(
        "--depth-sign-mode",
        default="",
        choices=["", "temporal", "model"],
        help="符号先验后端（§1.3 前置）：temporal 零依赖 / model 需 ONNX 权重；"
        "给出时自动开启 --depth-sign",
    )
    parser.add_argument(
        "--lifter",
        default="",
        choices=["", "videopose3d", "motionbert", "agformer"],
        help="抬升器后端（§3.3 的 A/B）：写入 DETECT_BODY_LIFTER，决定搜哪个 ONNX 权重",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="人工标注 JSON（samples/benchmark/kick-labels.json），用于 sign_accuracy 与矢状面角误差",
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=0.0,
        help="§4 方向向量先验权重（0 = 只有符号 hinge）；用于小权重扫描",
    )
    parser.add_argument(
        "--direction-sparsity",
        type=float,
        default=0.0,
        help="§4 深度稀疏项权重（0 = 关闭）；默认关闭，只在扫描时打开",
    )
    parser.add_argument(
        "--accel-weight",
        type=float,
        default=None,
        help="二阶（加速度）时域约束权重；不传则用 FitConfig 默认值。0 = 关闭",
    )
    parser.add_argument(
        "--zero-phase",
        dest="zero_phase",
        action="store_true",
        default=None,
        help="world 世界点那趟平滑改成零相位（前向+后向）",
    )
    parser.add_argument(
        "--no-zero-phase",
        dest="zero_phase",
        action="store_false",
        help="强制关闭零相位平滑",
    )
    parser.add_argument(
        "--write-labels",
        default="",
        help="把本次结果的矢状面角与符号写成标注草稿（§1.2 的半自动起点：先跑一遍再人工修正）",
    )
    args = parser.parse_args(argv)
    if args.depth_sign_mode:
        args.depth_sign = True
    _pin_settings_for_run(args)

    metrics = run(args)
    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            raise SystemExit(f"找不到基线文件：{baseline_path}")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print_table(metrics, baseline)

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n结果已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
