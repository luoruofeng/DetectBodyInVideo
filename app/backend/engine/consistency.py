"""双引擎一致性对比（任务 4.3 验收项）。

同一段视频、同一批帧分别用 MediaPipe 与 RTMPose 推理，按归一化坐标算
逐点欧氏距离：均值 / P95 小于阈值即认为两者等价（默认 0.05，约画面 5% 宽）。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseEngine, PoseFrame

logger = get_logger("engine.consistency")


class ConsistencyReport(BaseModel):
    frames_compared: int = 0
    landmarks_compared: int = 0
    mean_error: float = 0.0
    p95_error: float = 0.0
    max_error: float = 0.0
    threshold: float = 0.05
    equivalent: bool = False
    per_landmark: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def _distances(
    left: PoseFrame, right: PoseFrame, *, min_visibility: float = 0.3
) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    limit = min(len(left.landmarks), len(right.landmarks), LANDMARK_COUNT)
    for index in range(limit):
        a, b = left.landmarks[index], right.landmarks[index]
        if a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        pairs.append((index, float(np.hypot(a.x - b.x, a.y - b.y))))
    return pairs


def compare_frames(
    reference: list[PoseFrame],
    candidate: list[PoseFrame],
    *,
    threshold: float = 0.05,
    min_visibility: float = 0.3,
) -> ConsistencyReport:
    """``reference`` 为基准（通常 MediaPipe），``candidate`` 为待验证引擎结果。"""
    by_index: dict[int, PoseFrame] = {}
    for frame in candidate:
        by_index.setdefault(frame.index, frame)

    all_distances: list[float] = []
    per_landmark: dict[int, list[float]] = {}
    compared = 0
    for base in reference:
        other = by_index.get(base.index)
        if other is None:
            continue
        pairs = _distances(base, other, min_visibility=min_visibility)
        if not pairs:
            continue
        compared += 1
        for index, distance in pairs:
            all_distances.append(distance)
            per_landmark.setdefault(index, []).append(distance)

    if not all_distances:
        return ConsistencyReport(
            threshold=threshold,
            equivalent=False,
            notes=["没有可对比的帧（两侧都缺检测或置信度不足）"],
        )

    array = np.asarray(all_distances, dtype=np.float32)
    mean_error = float(array.mean())
    p95 = float(np.percentile(array, 95))
    return ConsistencyReport(
        frames_compared=compared,
        landmarks_compared=int(array.size),
        mean_error=round(mean_error, 5),
        p95_error=round(p95, 5),
        max_error=round(float(array.max()), 5),
        threshold=threshold,
        equivalent=bool(p95 <= threshold),
        per_landmark={
            f"mp_{index}": round(float(np.mean(values)), 5)
            for index, values in sorted(per_landmark.items())
        },
        notes=[
            f"对比 {compared} 帧 / {array.size} 个点",
            f"P95 误差 {p95:.4f} {'≤' if p95 <= threshold else '>'} 阈值 {threshold}",
        ],
    )


def compare_engines_on_video(
    path: Path,
    reference: PoseEngine,
    candidate: PoseEngine,
    *,
    max_frames: int = 60,
    stride: int = 5,
    threshold: float = 0.05,
) -> ConsistencyReport:
    """在同一视频上跑两个引擎并比较；引擎需已 ``prepare()``。"""
    from app.backend.media.reader import VideoFrameReader  # 避免循环导入

    reference_frames: list[PoseFrame] = []
    candidate_frames: list[PoseFrame] = []
    with VideoFrameReader(path) as reader:
        for packet in _take(reader.frames(stride=stride), max_frames):
            ref = reference.infer(packet.image, index=packet.index, timestamp_ms=packet.timestamp_ms)
            cand = candidate.infer(packet.image, index=packet.index, timestamp_ms=packet.timestamp_ms)
            if ref:
                reference_frames.append(ref[0])
            if cand:
                candidate_frames.append(cand[0])

    report = compare_frames(reference_frames, candidate_frames, threshold=threshold)
    logger.info(
        "引擎一致性：%s vs %s → mean=%.4f p95=%.4f 等价=%s",
        reference.name,
        candidate.name,
        report.mean_error,
        report.p95_error,
        report.equivalent,
    )
    return report


def _take(items: Iterable[PoseFrame] | Iterable[object], limit: int) -> list:
    result: list = []
    for item in items:
        result.append(item)
        if len(result) >= limit:
            break
    return result
