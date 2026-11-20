"""视频元信息解析与缩略图（任务 3.2）。

约定：
- 关闭 OpenCV 的自动旋转（``CAP_PROP_ORIENTATION_AUTO=0``），由我们按元数据手动旋转，
  保证三平台行为一致、后续关键点坐标与画面方向一致。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.errors import UnsupportedMediaError
from app.backend.core.logging import get_logger
from app.backend.media.validation import FOURCC_TO_CODEC, ensure_codec_allowed, probe_codec

logger = get_logger("media.probe")

DEFAULT_FPS = 30.0


class VideoProbe(BaseModel):
    container: str = ""
    codec: str | None = None
    width: int = 0
    height: int = 0
    rotation: int = 0
    display_width: int = 0
    display_height: int = 0
    fps: float = DEFAULT_FPS
    frame_count: int = 0
    duration_s: float = 0.0
    is_variable_frame_rate: bool = False
    has_audio: bool | None = None
    warnings: list[str] = Field(default_factory=list)


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise UnsupportedMediaError(
            "无法解码该视频，编码可能不受支持或文件已损坏；请转码为 H.264 / H.265 (MP4) 后重试",
            details={"filename": Path(path).name},
        )
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is not None:
        try:
            capture.set(orientation_auto, 0)  # 关闭自动旋转，统一由 apply_rotation 处理
        except Exception:  # noqa: BLE001 - 部分后端不支持
            logger.debug("无法关闭自动旋转", exc_info=True)
    return capture


def _safe_get(capture: cv2.VideoCapture, prop: int | None, default: float = 0.0) -> float:
    if prop is None:
        return default
    try:
        value = float(capture.get(prop))
    except Exception:  # noqa: BLE001
        return default
    return value if math.isfinite(value) else default


def read_rotation(capture: cv2.VideoCapture) -> int:
    rotation = _safe_get(capture, getattr(cv2, "CAP_PROP_ORIENTATION_META", None), 0.0)
    if not rotation:
        return 0
    return int(round(rotation / 90.0) * 90) % 360


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _fourcc_codec(capture: cv2.VideoCapture) -> str | None:
    try:
        raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    fourcc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower()
    return FOURCC_TO_CODEC.get(fourcc, fourcc or None)


def _timing_samples(capture: cv2.VideoCapture, fps: float, max_samples: int = 30) -> list[float]:
    """采样帧间隔（毫秒），用于识别可变帧率。"""
    deltas: list[float] = []
    previous: float | None = None
    for _ in range(max_samples):
        ok, _frame = capture.read()
        if not ok:
            break
        timestamp = _safe_get(capture, cv2.CAP_PROP_POS_MSEC, -1.0)
        if previous is not None and timestamp > previous:
            deltas.append(timestamp - previous)
        previous = timestamp
    return deltas


def _detect_vfr(deltas: list[float], fps: float) -> bool:
    if len(deltas) < 5 or fps <= 0:
        return False
    expected = 1000.0 / fps
    median = float(np.median(deltas))
    if median <= 0:
        return False
    if abs(median - expected) / expected > 0.2:
        return True
    low, high = np.percentile(deltas, [10, 90])
    return bool(low > 0 and (high - low) / median > 0.35)


def probe_video(path: Path, *, allowed_codecs: tuple[str, ...] | list[str] | None = None,
                rejected_codecs: tuple[str, ...] | list[str] | None = None) -> VideoProbe:
    """读取帧率/分辨率/帧数/时长/旋转/VFR，并验证视频确实可解码。"""
    warnings: list[str] = []
    capture = open_capture(path)
    try:
        raw_width = int(_safe_get(capture, cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(_safe_get(capture, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _safe_get(capture, cv2.CAP_PROP_FPS, DEFAULT_FPS)
        if fps <= 1 or fps > 480:
            warnings.append(f"帧率异常（{fps:.1f}），已按 {DEFAULT_FPS:g} fps 处理")
            fps = DEFAULT_FPS

        frame_count = int(_safe_get(capture, cv2.CAP_PROP_FRAME_COUNT))
        rotation = read_rotation(capture)

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise UnsupportedMediaError(
                "视频无法解码首帧，可能编码不受支持或文件损坏；请转码为 H.264 / H.265 (MP4)",
                details={"filename": path.name},
            )
        if not raw_width or not raw_height:
            raw_height, raw_width = first_frame.shape[:2]

        deltas = _timing_samples(capture, fps)
        is_vfr = _detect_vfr(deltas, fps)
        if is_vfr:
            warnings.append("检测到可变帧率（VFR），分析时将按时间戳重采样到固定 fps")

        if frame_count <= 0:
            duration_guess = _safe_get(capture, cv2.CAP_PROP_POS_MSEC, 0.0) / 1000.0
            frame_count = max(1, int(round(duration_guess * fps))) if duration_guess else 1
            warnings.append("容器未提供总帧数，已按时长估算")

        codec = probe_codec(path) or _fourcc_codec(capture)
        ensure_codec_allowed(codec, allowed=allowed_codecs or (), rejected=rejected_codecs or ())

        duration_s = frame_count / fps if fps > 0 else 0.0
        if rotation in (90, 270):
            display_width, display_height = raw_height, raw_width
        else:
            display_width, display_height = raw_width, raw_height

        if fps < 20:
            warnings.append(f"帧率仅 {fps:.1f} fps，快速动作可能出现抖动")

        return VideoProbe(
            container=path.suffix.lower().lstrip("."),
            codec=codec,
            width=raw_width,
            height=raw_height,
            rotation=rotation,
            display_width=display_width,
            display_height=display_height,
            fps=round(fps, 4),
            frame_count=frame_count,
            duration_s=round(duration_s, 3),
            is_variable_frame_rate=is_vfr,
            warnings=warnings,
        )
    finally:
        capture.release()


def generate_thumbnail(
    path: Path,
    destination: Path,
    *,
    probe: VideoProbe | None = None,
    max_width: int = 480,
    at_ratio: float = 0.1,
) -> bool:
    """抽一帧生成 JPEG 缩略图；失败返回 False（不阻断上传流程）。"""
    capture = None
    try:
        capture = open_capture(path)
        total = int(_safe_get(capture, cv2.CAP_PROP_FRAME_COUNT)) or 1
        target = max(0, min(total - 1, int(total * at_ratio)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            return False

        rotation = probe.rotation if probe else read_rotation(capture)
        frame = apply_rotation(frame, rotation)

        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, max(1, int(round(height * scale)))),
                               interpolation=cv2.INTER_AREA)

        destination.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(destination), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]))
    except Exception:  # noqa: BLE001 - 缩略图失败不影响主流程
        logger.warning("生成缩略图失败：%s", path.name, exc_info=True)
        return False
    finally:
        if capture is not None:
            capture.release()
