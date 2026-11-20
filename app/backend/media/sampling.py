"""采样策略（任务 3.4）。

默认全帧分析；可选每 N 帧降采样、目标 fps、区间裁剪；
"自动裁掉无人物区间"用 OpenCV HOG 行人检测做轻量粗筛（无需模型权重）。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
from pydantic import BaseModel, Field

from app.backend.core.errors import ValidationFailedError
from app.backend.core.logging import get_logger
from app.backend.media.probe import DEFAULT_FPS, VideoProbe
from app.backend.media.reader import VideoFrameReader

logger = get_logger("media.sampling")


class TimeRange(BaseModel):
    start_s: float
    end_s: float


class SamplingConfig(BaseModel):
    every_n_frames: int = Field(1, ge=1, le=120, description="每 N 帧取一帧（target_fps 优先）")
    target_fps: float | None = Field(None, gt=0, le=240, description="重采样目标帧率")
    start_s: float | None = Field(None, ge=0, description="起点（秒）")
    end_s: float | None = Field(None, ge=0, description="终点（秒）")
    trim_empty: bool = Field(False, description="自动裁掉无人物区间（轻量粗筛）")


class AnalysisPlan(BaseModel):
    total_frames: int
    selected_frames: int
    stride: int
    effective_fps: float
    start_s: float
    end_s: float
    target_fps: float | None = None
    trimmed: bool = False
    active_ranges: list[TimeRange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def resolve_bounds(config: SamplingConfig, duration_s: float) -> tuple[float, float]:
    start = max(0.0, float(config.start_s or 0.0))
    end = float(config.end_s) if config.end_s else duration_s
    end = min(end, duration_s) if duration_s > 0 else end
    if end <= start:
        raise ValidationFailedError(
            f"结束时间 {end:.2f}s 必须大于开始时间 {start:.2f}s",
            details={"startS": start, "endS": end, "durationS": duration_s},
        )
    return start, end


def _stride_for(config: SamplingConfig, fps: float) -> int:
    if config.target_fps and config.target_fps > 0:
        # 目标帧率高于源帧率时不做帧复制，最多保持原帧率
        return max(1, int(round(fps / config.target_fps)))
    return max(1, config.every_n_frames)


def _count_window(start_s: float, end_s: float, fps: float, frame_count: int, stride: int) -> int:
    if end_s <= start_s or fps <= 0:
        return 0
    start_index = min(max(0, int(round(start_s * fps))), max(0, frame_count))
    end_index = min(max(0, int(round(end_s * fps))), frame_count)
    if end_index <= start_index:
        return 0
    return int(math.ceil((end_index - start_index) / stride))


def select_indices(config: SamplingConfig, probe: VideoProbe) -> list[int]:
    """返回要分析的帧号列表（内存占用与帧数线性，超长视频建议直接流式消费）。"""
    fps = probe.fps or DEFAULT_FPS
    frame_count = probe.frame_count
    stride = _stride_for(config, fps)
    start_s, end_s = resolve_bounds(config, probe.duration_s or frame_count / fps)
    start_index = max(0, int(round(start_s * fps)))
    end_index = min(frame_count, int(round(end_s * fps)))
    return list(range(start_index, end_index, stride))


def coarse_active_ranges(
    path: Path,
    probe: VideoProbe,
    *,
    sample_fps: float = 1.0,
    padding_s: float = 0.4,
    merge_gap_s: float = 1.5,
    max_scanned_frames: int = 400,
) -> list[TimeRange] | None:
    """HOG 行人检测粗筛人物活跃区间。

    返回 ``None`` 表示轻量检测不可用（调用方应退回全片段）；
    返回 ``[]`` 表示确实没检测到人物（可疑素材）。
    """
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:  # noqa: BLE001 - 精简版 OpenCV 可能没有 HOG
        logger.warning("当前 OpenCV 缺少 HOG 行人检测，跳过自动裁剪")
        return None

    fps = probe.fps or DEFAULT_FPS
    stride = max(1, int(round(fps / max(0.1, sample_fps))))
    hits: list[float] = []
    scanned = 0

    reader = VideoFrameReader(path, probe)
    try:
        for packet in reader.frames(stride=stride):
            scanned += 1
            if scanned > max_scanned_frames:
                break
            image = packet.image
            height, width = image.shape[:2]
            if width > 512:
                scale = 512.0 / width
                image = cv2.resize(image, (512, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                rects, weights = hog.detectMultiScale(
                    gray, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
            except cv2.error:
                logger.warning("HOG 检测失败，跳过自动裁剪")
                return None
            if len(rects) and float(weights.max()) > 0.35:
                hits.append(packet.timestamp_s)
    except Exception:  # noqa: BLE001 - 粗筛失败不影响主流程
        logger.warning("自动裁剪粗筛失败", exc_info=True)
        return None
    finally:
        reader.close()

    if not hits:
        return []

    duration = probe.duration_s or (probe.frame_count / fps if fps else 0.0)
    ranges: list[TimeRange] = []
    start = previous = hits[0]
    for timestamp in hits[1:]:
        if timestamp - previous > merge_gap_s:
            ranges.append(
                TimeRange(
                    start_s=max(0.0, round(start - padding_s, 3)),
                    end_s=round(min(duration, previous + padding_s), 3) if duration else round(previous + padding_s, 3),
                )
            )
            start = timestamp
        previous = timestamp
    ranges.append(
        TimeRange(
            start_s=max(0.0, round(start - padding_s, 3)),
            end_s=round(min(duration, previous + padding_s), 3) if duration else round(previous + padding_s, 3),
        )
    )
    return [item for item in ranges if item.end_s > item.start_s]


def build_analysis_plan(
    path: Path,
    probe: VideoProbe,
    config: SamplingConfig,
) -> AnalysisPlan:
    fps = probe.fps or DEFAULT_FPS
    frame_count = probe.frame_count
    duration = probe.duration_s or (frame_count / fps)
    warnings: list[str] = []
    stride = _stride_for(config, fps)
    start_s, end_s = resolve_bounds(config, duration)

    if config.target_fps and config.target_fps > fps:
        warnings.append(
            f"目标帧率 {config.target_fps:g} fps 高于源帧率 {fps:g} fps，将保持原帧率（不复制帧）"
        )
    if probe.is_variable_frame_rate:
        warnings.append("源视频为可变帧率，分析时按时间戳重采样，帧号可能不连续")

    ranges: list[TimeRange] = []
    trimmed = False
    if config.trim_empty:
        sample_fps = 1.0 if duration <= 300 else 0.5
        active = coarse_active_ranges(path, probe, sample_fps=sample_fps)
        if active is None:
            warnings.append("自动裁剪不可用，已按完整区间分析")
        elif not active:
            warnings.append("未检测到人物，自动裁剪已跳过，建议检查视频内容")
        else:
            ranges = [
                TimeRange(start_s=max(start_s, item.start_s), end_s=min(end_s, item.end_s))
                for item in active
            ]
            ranges = [item for item in ranges if item.end_s > item.start_s]
            trimmed = bool(ranges)
            if trimmed:
                start_s = ranges[0].start_s
                end_s = ranges[-1].end_s
                kept = sum(item.end_s - item.start_s for item in ranges)
                warnings.append(
                    f"自动裁剪后保留 {len(ranges)} 段，共 {kept:.1f}s（原 {duration:.1f}s）"
                )

    if trimmed:
        selected = sum(_count_window(item.start_s, item.end_s, fps, frame_count, stride) for item in ranges)
    else:
        selected = _count_window(start_s, end_s, fps, frame_count, stride)

    effective_fps = round(fps / stride, 3)
    return AnalysisPlan(
        total_frames=frame_count,
        selected_frames=max(0, selected),
        stride=stride,
        effective_fps=effective_fps,
        start_s=round(start_s, 3),
        end_s=round(end_s, 3),
        target_fps=config.target_fps,
        trimmed=trimmed,
        active_ranges=ranges,
        warnings=warnings,
    )
