"""深度（Z 轴）精度基准（任务 1.3）。

为什么要单独一套脚本：``scripts/bench.py`` 只测"推理链路吞吐"，
而这一份测的是**输出姿态的几何质量**。四项指标对应 `docs/todo-depth-z-accuracy.md` 的 1.2：

1. ``z_jitter``       相邻帧 ``world_landmarks.z`` 的 RMS（米），按 躯干/四肢/末端 分组；
2. ``bone_length_cv`` 每条骨段长度的变异系数（std/mean），基线预期 > 3%；
3. ``reproj_error_px`` 把 ``world_landmarks`` 反投影回像素后与 ``landmarks`` 的平均误差；
4. ``angle_drift_deg`` 膝/肘/髋角的帧间标准差（"静止站立"片段里越小越好）。

用法：

    # 真实视频（走完整分析链路，指标在"后处理之后"的帧上统计）
    python scripts/bench_depth.py samples/front_full.mp4 --json out/front_full.json

    # 无模型 / 无素材时用确定性合成序列（CI 可跑，结果可重复）
    python scripts/bench_depth.py --synthetic

    # 与上一次结果对比（回归检测）
    python scripts/bench_depth.py --synthetic --baseline out/front_full.json

    # A/B：关掉本次新增的处理，看指标是否真的变化
    python scripts/bench_depth.py --synthetic --no-ik-fit --no-z-smoothing
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
}


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


def _analyze(
    path: Path, options: Any
) -> tuple[list[PoseFrame], int, int, float, dict[str, Any]]:
    from app.backend.pipeline import AnalysisOptions, analyze_video

    result = analyze_video(path, options=AnalysisOptions(postprocess=options))
    probe = result.sampling
    width = getattr(probe, "display_width", 0) or 1080
    height = getattr(probe, "display_height", 0) or 1920
    fit = getattr(result.postprocess, "fit", None)
    extra = {
        "engine": result.selection.engine,
        "engineDisplayName": result.selection.display_name,
        "verdict": result.postprocess.quality.verdict,
        "fitMsPerFrame": getattr(fit, "milliseconds_per_frame", None),
    }
    return (
        result.postprocess.frames,
        int(width),
        int(height),
        float(result.sampling.fps or 30.0),
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
        postprocess = postprocess.model_copy(
            update={"ik_fit": postprocess.ik_fit.model_copy(update={"enabled": not args.no_ik_fit})}
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
    return postprocess.model_copy(update={"smoothing": smoothing})


def run(args: argparse.Namespace) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    if args.synthetic:
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
    metrics = {
        "label": args.label or ("synthetic" if args.synthetic else Path(args.video).stem),
        "source": "synthetic" if args.synthetic else str(args.video),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 3),
        "frames": len(frames),
        "z_jitter": z_jitter(frames),
        "xy_jitter": xy_jitter(frames),
        "bone_length_cv": cv,
        "bone_length_cv_max": round(max(cv.values()), 6) if cv else 0.0,
        "bone_length_cv_mean": round(float(np.mean(list(cv.values()))), 6) if cv else 0.0,
        "reproj_error_px": reprojection_summary(
            frames, aspect=aspect, width=width, height=height
        ),
        "angle_drift_deg": angle_drift(frames),
        "timing": timing,
        "env": {
            "platform": f"{platform.system()} / {platform.machine()}",
            "python": platform.python_version(),
        },
        "ik_fit_enabled": not args.no_ik_fit,
        "yaw": yaw_comparison(frames, truth_deg=args.rotation if args.synthetic else None),
    }
    metrics.update({f"info_{key}": value for key, value in extra.items()})
    return metrics


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
    yaw = metrics.get("yaw") or {}
    for key, source in (
        ("yaw_error_width_deg", "width_error_deg"),
        ("yaw_jitter_width_deg", "width_jitter_deg"),
        ("yaw_jitter_depth_deg", "depth_jitter_deg"),
    ):
        if yaw.get(source) is not None:
            flat[key] = float(yaw[source])
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
                verdict = "劣化" if ratio > limit * 100 else "ok"
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
    parser.add_argument("--no-z-smoothing", action="store_true", help="关闭 z 轴强平滑（A/B）")
    parser.add_argument("--no-smoothing", action="store_true", help="完全关闭平滑（A/B）")
    args = parser.parse_args(argv)

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
