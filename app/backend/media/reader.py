"""逐帧读取器（任务 3.3）。

- 流式读取：生成器逐帧产出，从不整段读入内存。
- 支持起止时间码、帧步长、任意帧号。
- VFR 视频可按时间戳重采样到固定 fps（只丢帧不复制，避免伪造数据）。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.backend.core.logging import get_logger
from app.backend.media.probe import DEFAULT_FPS, VideoProbe, apply_rotation, open_capture, probe_video

logger = get_logger("media.reader")


@dataclass(slots=True)
class FramePacket:
    index: int
    timestamp_ms: float
    image: np.ndarray

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0


class VideoFrameReader:
    """单实例对应一路视频；内部只有一个 VideoCapture，请顺序消费。"""

    def __init__(self, path: Path | str, probe: VideoProbe | None = None) -> None:
        self.path = Path(path)
        self.probe = probe or probe_video(self.path)
        self.fps = float(self.probe.fps or DEFAULT_FPS)
        self.frame_count = int(self.probe.frame_count or 0)
        self._capture: cv2.VideoCapture | None = None

    # ---- 生命周期 ----
    def open(self) -> None:
        if self._capture is None:
            self._capture = open_capture(self.path)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoFrameReader:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _capture_or_open(self) -> cv2.VideoCapture:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        return self._capture

    # ---- 读取 ----
    def read_frame(self, index: int) -> FramePacket | None:
        """精确读取指定帧号（会移动内部指针）。"""
        if index < 0 or (self.frame_count and index >= self.frame_count):
            return None
        capture = self._capture_or_open()
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            logger.debug("读取第 %s 帧失败", index)
            return None
        timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if timestamp <= 0:
            timestamp = index * 1000.0 / self.fps
        return FramePacket(
            index=index,
            timestamp_ms=timestamp,
            image=apply_rotation(frame, self.probe.rotation),
        )

    def frames(
        self,
        *,
        start_s: float = 0.0,
        end_s: float | None = None,
        stride: int = 1,
        target_fps: float | None = None,
    ) -> Iterator[FramePacket]:
        """按区间/步长流式产帧。

        ``target_fps`` 优先于 ``stride``：按时间戳挑最近的帧，实现 VFR → 固定 fps。
        """
        capture = self._capture_or_open()
        fps = self.fps
        stride = max(1, int(stride))
        start_s = max(0.0, float(start_s or 0.0))
        start_index = min(self.frame_count - 1, int(round(start_s * fps))) if self.frame_count else int(round(start_s * fps))
        start_index = max(0, start_index)

        if start_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

        interval_ms = 1000.0 / target_fps if target_fps and target_fps > 0 else None
        end_ms = end_s * 1000.0 if end_s is not None else None
        index = start_index
        next_emit_ms: float | None = None

        while True:
            if not capture.grab():
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if timestamp <= 0:
                timestamp = index * 1000.0 / fps
            if end_ms is not None and timestamp >= end_ms - 1e-6:
                break

            if interval_ms is None:
                should_emit = (index - start_index) % stride == 0
            else:
                if next_emit_ms is None:
                    next_emit_ms = timestamp
                # 5% 容差吸收 POS_MSEC 的浮点噪声，且远小于半个间隔避免帧率翻倍
                tolerance = max(0.5, interval_ms * 0.05)
                should_emit = timestamp + tolerance >= next_emit_ms
                if should_emit:
                    next_emit_ms = timestamp + interval_ms

            if should_emit:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    logger.debug("retrieve 第 %s 帧失败，跳过", index)
                else:
                    yield FramePacket(
                        index=index,
                        timestamp_ms=timestamp,
                        image=apply_rotation(frame, self.probe.rotation),
                    )
            index += 1

    def frames_at(self, indices: Iterable[int]) -> Iterator[FramePacket]:
        """按任意帧号序列取帧（用于按采样策略精确取帧）。"""
        for index in indices:
            packet = self.read_frame(index)
            if packet is not None:
                yield packet

    def iter_all(self) -> Iterator[FramePacket]:
        return self.frames()
