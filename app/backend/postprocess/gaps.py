"""缺失帧处理（任务 5.2）。

原则：**只补短缺口，不伪造长时间缺失**。
- 置信度不足 / 关键点太少 / 未检测到人的帧一律 ``valid=False``。
- 缺口 ≤ ``max_gap_frames`` 且两侧都有有效观测时线性插值，并标记
  ``interpolated=True``（导出端可据此降低权重）。
- 长缺口保持 ``valid=False`` 且不写坐标，避免 iOS 端模型凭空乱抖。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame

logger = get_logger("postprocess.gaps")


class GapFillConfig(BaseModel):
    visibility_threshold: float = Field(0.5, gt=0, lt=1, description="单点可见性阈值")
    min_visible_landmarks: int = Field(6, ge=3, le=LANDMARK_COUNT)
    min_score: float = Field(0.4, ge=0, le=1, description="整帧平均置信度阈值")
    max_gap_frames: int = Field(5, ge=0, le=60, description="≤N 帧的缺口才插值")
    interpolate: bool = True


class GapSpan(BaseModel):
    start_index: int
    end_index: int
    length: int
    filled: bool


class GapFillReport(BaseModel):
    frames_total: int = 0
    frames_invalid: int = 0
    frames_filled: int = 0
    frames_missing: int = 0
    gaps: list[GapSpan] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def count_visible(frame: PoseFrame, threshold: float = 0.5) -> int:
    return sum(1 for item in frame.landmarks if item.visibility >= threshold)


def is_frame_valid(frame: PoseFrame, config: GapFillConfig) -> bool:
    if len(frame.landmarks) < config.min_visible_landmarks:
        return False
    if frame.score and frame.score < config.min_score:
        return False
    return count_visible(frame, config.visibility_threshold) >= config.min_visible_landmarks


def mark_invalid(frames: Sequence[PoseFrame], config: GapFillConfig) -> int:
    """标记低质量帧（原地修改），返回被标为无效的帧数。"""
    invalid = 0
    for frame in frames:
        valid = is_frame_valid(frame, config)
        frame.valid = valid
        if not valid:
            invalid += 1
    return invalid


def densify(frames: Sequence[PoseFrame], *, stride: int = 1, fps: float = 30.0) -> list[PoseFrame]:
    """把稀疏采样（每 N 帧）补成连续帧列表，缺的位置插入空占位帧。"""
    ordered = sorted(frames, key=lambda item: item.index)
    if not ordered:
        return []
    stride = max(1, int(stride))
    if stride == 1:
        return list(ordered)

    # 优先用已有帧反推"每帧毫秒数"，避免 fps 与时间戳不同步导致占位帧时间漂移
    interval_ms = 1000.0 / fps if fps > 0 else 1000.0 / 30.0
    inferred: list[float] = []
    for before, after in zip(ordered, ordered[1:], strict=False):
        if after.index > before.index and after.timestamp_ms > before.timestamp_ms:
            inferred.append(
                (after.timestamp_ms - before.timestamp_ms) / (after.index - before.index)
            )
    if inferred:
        interval_ms = float(sorted(inferred)[len(inferred) // 2])

    by_index = {frame.index: frame for frame in ordered}
    result: list[PoseFrame] = []
    cursor = ordered[0].index
    last_index = ordered[-1].index
    while cursor <= last_index:
        existing = by_index.get(cursor)
        if existing is not None:
            result.append(existing)
        else:
            timestamp = ordered[0].timestamp_ms + (cursor - ordered[0].index) * interval_ms
            result.append(
                PoseFrame(
                    index=cursor,
                    timestamp_ms=timestamp,
                    landmarks=[],
                    valid=False,
                )
            )
        cursor += stride
    return result


def fill_short_gaps(frames: Sequence[PoseFrame], config: GapFillConfig) -> GapFillReport:
    """线性插值补齐短缺口（原地修改 ``frames`` 中对应帧）。"""
    ordered = sorted(frames, key=lambda item: item.index)
    report = GapFillReport(frames_total=len(ordered))
    if not ordered:
        return report

    valid_flags = [is_frame_valid(frame, config) for frame in ordered]
    report.frames_invalid = sum(1 for flag in valid_flags if not flag)
    if not config.interpolate or config.max_gap_frames <= 0:
        report.notes.append("未启用插值，仅标记无效帧")
        return report

    index = 0
    while index < len(ordered):
        if valid_flags[index]:
            index += 1
            continue
        start = index
        while index < len(ordered) and not valid_flags[index]:
            index += 1
        end = index  # 第一个有效帧
        length = end - start
        has_both_sides = start > 0 and end < len(ordered)
        fillable = (
            has_both_sides
            and length <= config.max_gap_frames
            and bool(ordered[start - 1].landmarks)
            and bool(ordered[end].landmarks)
        )
        report.gaps.append(
            GapSpan(
                start_index=ordered[start].index,
                end_index=ordered[end - 1].index,
                length=length,
                filled=fillable,
            )
        )
        if fillable:
            _interpolate(ordered, start, end)
            for offset in range(start, end):
                valid_flags[offset] = True
            report.frames_filled += length
        else:
            report.frames_missing += length
        # ``index == end`` 已是有效帧（或已到末尾），直接进入下一轮判断

    if report.frames_missing:
        report.notes.append(
            f"{report.frames_missing} 帧处于长缺口或片头片尾，保持 valid=false（不伪造数据）"
        )
    return report


def _interpolate(frames: list[PoseFrame], start: int, end: int) -> None:
    """在 ``frames[start:end]`` 上按时间戳线性插值，两端为有效帧。

    ``world_landmarks`` 必须一起插值：导出链路只认世界坐标（``skeleton/transform.py``），
    补出来的帧若缺世界坐标，3D 播放时就会突然掉一帧骨架。
    """
    before = frames[start - 1]
    after = frames[end]
    span = max(1e-6, after.timestamp_ms - before.timestamp_ms)

    for offset in range(start, end):
        target = frames[offset]
        ratio = (target.timestamp_ms - before.timestamp_ms) / span
        ratio = min(1.0, max(0.0, ratio))
        landmarks: list[Landmark] = []
        for index in range(LANDMARK_COUNT):
            a = before.landmark(index)
            b = after.landmark(index)
            if a is None or b is None:
                landmarks.append(Landmark())
                continue
            interpolated = a.lerp(b, ratio)
            # 插值点的置信度取两端较小值，避免"补出来的点"被当成高可信观测
            interpolated.visibility = min(a.visibility, b.visibility)
            landmarks.append(interpolated)
        target.landmarks = landmarks
        target.world_landmarks = _interpolate_world(before, after, ratio)
        target.score = min(before.score, after.score)
        target.valid = True
        target.interpolated = True
        if target.bbox is None:
            target.bbox = target.bbox_from_landmarks()


def _interpolate_world(before: PoseFrame, after: PoseFrame, ratio: float) -> list[Landmark]:
    """插值米制世界坐标；两端任一侧缺失就返回空列表（宁缺勿造）。"""
    if len(before.world_landmarks) < LANDMARK_COUNT or len(after.world_landmarks) < LANDMARK_COUNT:
        return []
    world: list[Landmark] = []
    for index in range(LANDMARK_COUNT):
        a = before.world_landmarks[index]
        b = after.world_landmarks[index]
        interpolated = a.lerp(b, ratio)
        interpolated.visibility = min(a.visibility, b.visibility)
        world.append(interpolated)
    return world


def process_gaps(
    frames: Sequence[PoseFrame], config: GapFillConfig | None = None
) -> tuple[list[PoseFrame], GapFillReport]:
    config = config or GapFillConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    mark_invalid(ordered, config)
    report = fill_short_gaps(ordered, config)
    return ordered, report


def interpolated_ratio(frames: Sequence[PoseFrame]) -> float:
    if not frames:
        return 0.0
    return sum(1 for frame in frames if frame.interpolated) / len(frames)
