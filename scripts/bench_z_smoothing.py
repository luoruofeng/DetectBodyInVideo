"""Z 轴强平滑的调参基准（任务 4.4）。

目的：在"Z 抖动下降"和"动作延迟"之间选一组默认值。两组指标都在确定性合成序列上测，
不需要真实素材与模型权重，CI 可跑、结果可复现：

1. ``z_jitter``：静止姿态 + 高斯 Z 噪声，相邻帧 RMS（米）——越小越稳；
2. ``step_lag``：人物整体在 Z 上走一个阶跃（0 → -0.15 m），输出到达 50% 处的延迟帧数
   ——越大说明滤波越"拖"。

用法：

    python scripts/bench_z_smoothing.py
    python scripts/bench_z_smoothing.py --json out/z_params.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import synth_pose  # noqa: E402
from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame  # noqa: E402
from app.backend.postprocess.one_euro import (  # noqa: E402
    OneEuroParams,
    default_world_one_euro_params,
    smooth_frames,
)

FPS = 30.0
JITTER_LANDMARKS = (11, 12, 23, 24, 27, 28)
STEP_METERS = -0.15


def param_groups() -> dict[str, OneEuroParams]:
    """三组候选参数（只改 z，x/y 固定为米制默认值 1.0 / 1.0）。"""
    strong = OneEuroParams(min_cutoff=1.0, beta=1.0, min_cutoff_z=0.15, beta_z=0.15)
    return {
        # 弱 = 与 x/y 完全一致（改造前的行为）
        "weak": OneEuroParams(min_cutoff=1.0, beta=1.0),
        "default": default_world_one_euro_params(),
        "strong": strong,
    }


def _static_frames(count: int = 60) -> list[PoseFrame]:
    return synth_pose.add_depth_noise(
        synth_pose.make_frames(count, fps=FPS), z_sigma=0.02, xy_sigma=0.004
    )


def _step_frames(count: int = 60, at: int = 20) -> list[PoseFrame]:
    """第 ``at`` 帧起人物整体在 Z 上前移 ``STEP_METERS``（无噪声）。"""
    points = synth_pose.standing_points()
    frames: list[PoseFrame] = []
    for index in range(count):
        offset = STEP_METERS if index >= at else 0.0
        shifted = {key: value + np.array([0.0, 0.0, offset]) for key, value in points.items()}
        normalized = synth_pose.project_to_normalized(shifted)
        frames.append(
            PoseFrame(
                index=index,
                timestamp_ms=index * 1000.0 / FPS,
                landmarks=[
                    Landmark(x=normalized[i][0], y=normalized[i][1], z=normalized[i][2], visibility=0.95)
                    for i in range(LANDMARK_COUNT)
                ],
                world_landmarks=[
                    Landmark(x=float(shifted[i][0]), y=float(shifted[i][1]), z=float(shifted[i][2]), visibility=0.95)
                    for i in range(LANDMARK_COUNT)
                ],
                score=0.95,
                valid=True,
            )
        )
    return frames


def z_jitter(frames: list[PoseFrame]) -> float:
    squared: list[float] = []
    for previous, current in zip(frames, frames[1:], strict=False):
        for index in JITTER_LANDMARKS:
            a = previous.world_landmarks[index].z
            b = current.world_landmarks[index].z
            squared.append((b - a) ** 2)
    return float(np.sqrt(np.mean(squared))) if squared else 0.0


def step_lag(smoothed: list[PoseFrame], at: int, reference: list[PoseFrame]) -> float:
    """输出到达阶跃 50% 处的延迟（帧，含小数；线性插值）。"""
    def signal(frames: list[PoseFrame], index: int) -> float:
        return float(np.mean([frames[index].world_landmarks[i].z for i in JITTER_LANDMARKS]))

    start = signal(reference, at - 1)
    target = signal(reference, -1)
    half = start + 0.5 * (target - start)
    for offset in range(0, 30):
        index = at + offset
        if index >= len(smoothed):
            break
        value = signal(smoothed, index)
        if value <= half:
            previous = signal(smoothed, index - 1)
            span = previous - value
            fraction = (previous - half) / span if span > 1e-12 else 0.0
            return float(offset - 1 + fraction)
    return float("nan")


def run() -> dict[str, Any]:
    static = _static_frames()
    step = _step_frames()
    raw_jitter = z_jitter(static)
    rows: list[dict[str, Any]] = []
    for name, params in param_groups().items():
        smoothed_static = smooth_frames(static, params, use_world=True)
        smoothed_step = smooth_frames(step, params, use_world=True)
        jitter = z_jitter(smoothed_static)
        lag = step_lag(smoothed_step, 20, step)
        rows.append(
            {
                "group": name,
                "min_cutoff_z": params.z_params().min_cutoff,
                "beta_z": params.z_params().beta,
                "z_jitter": round(jitter, 6),
                "z_jitter_reduction_pct": round((1.0 - jitter / raw_jitter) * 100.0, 2)
                if raw_jitter > 1e-12
                else 0.0,
                "step_lag_frames": round(lag, 2) if math.isfinite(lag) else None,
                "step_lag_ms": round(lag * 1000.0 / FPS, 1) if math.isfinite(lag) else None,
            }
        )
    return {"raw_z_jitter": round(raw_jitter, 6), "fps": FPS, "rows": rows}


def print_table(payload: dict[str, Any]) -> None:
    print(f"\n=== Z 强平滑调参（{payload['fps']} fps，原始 z 抖动 {payload['raw_z_jitter']} m）===")
    header = f"{'参数组':<10}{'min_cutoff_z':>14}{'beta_z':>9}{'z 抖动(m)':>13}{'下降':>9}{'阶跃延迟':>12}"
    print(header)
    print("-" * 72)
    for row in payload["rows"]:
        print(
            f"{row['group']:<10}{row['min_cutoff_z']:>14}{row['beta_z']:>9}"
            f"{row['z_jitter']:>13.5f}{row['z_jitter_reduction_pct']:>8.1f}%"
            f"{str(row['step_lag_frames']) + ' 帧':>12}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Z 轴强平滑调参基准")
    parser.add_argument("--json", default="", help="把结果写到该 JSON 路径")
    args = parser.parse_args(argv)

    payload = run()
    print_table(payload)
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n结果已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
