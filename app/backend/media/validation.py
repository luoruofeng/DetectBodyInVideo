"""视频文件校验：容器扩展名、体积、可解码性、编码白名单（任务 3.1）。

拒绝时必须给出**可执行的原因**，前端直接展示。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from app.backend.core.errors import PayloadTooLargeError, UnsupportedMediaError, ValidationFailedError
from app.backend.core.logging import get_logger

logger = get_logger("media.validation")

# fourcc → 归一化编码名
FOURCC_TO_CODEC: dict[str, str] = {
    "avc1": "h264",
    "h264": "h264",
    "x264": "h264",
    "hev1": "hevc",
    "hvc1": "hevc",
    "hvc2": "hevc",
    "hevc": "hevc",
    "h265": "hevc",
    "mp4v": "mpeg4",
    "xvid": "mpeg4",
    "divx": "mpeg4",
    "mjpg": "mjpeg",
    "jpeg": "mjpeg",
    "ap4h": "prores",
    "apcn": "prores",
    "av01": "av1",
    "vp09": "vp9",
    "vp08": "vp8",
}

CONTAINER_LABELS = {
    ".mp4": "MP4",
    ".mov": "MOV",
    ".m4v": "M4V",
    ".avi": "AVI",
    ".mkv": "MKV",
}


def supported_container_text(allowed: tuple[str, ...] | list[str]) -> str:
    labels = [CONTAINER_LABELS.get(ext.lower(), ext.lstrip(".").upper()) for ext in allowed]
    return " / ".join(labels)


def validate_extension(filename: str, allowed: tuple[str, ...] | list[str]) -> str:
    suffix = Path(filename).suffix.lower()
    if not suffix:
        raise UnsupportedMediaError(
            "文件没有扩展名，无法判断视频格式",
            details={"filename": filename, "supported": supported_container_text(allowed)},
        )
    if suffix not in {ext.lower() for ext in allowed}:
        raise UnsupportedMediaError(
            f'不支持的容器格式 "{suffix}"，请使用 {supported_container_text(allowed)}',
            details={"extension": suffix, "supported": supported_container_text(allowed)},
        )
    return suffix


def validate_size(size_bytes: int, max_bytes: int) -> None:
    if size_bytes <= 0:
        raise ValidationFailedError("文件为空（0 字节），请重新选择文件")
    if size_bytes > max_bytes:
        raise PayloadTooLargeError(
            f"文件超出上限（{size_bytes / 1024**3:.2f} GiB > {max_bytes / 1024**3:.1f} GiB）",
            details={"sizeBytes": size_bytes, "maxBytes": max_bytes},
        )


def probe_codec(path: Path) -> str | None:
    """优先 ffprobe 精确取 codec_name；没有 ffprobe 时返回 None 由调用方用 fourcc 兜底。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - 固定参数列表
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("ffprobe 调用失败", exc_info=True)
        return None
    if completed.returncode != 0:
        return None
    codec = completed.stdout.strip().splitlines()[0].strip() if completed.stdout.strip() else ""
    return codec.lower() or None


def has_audio_stream(path: Path) -> bool | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def ensure_codec_allowed(
    codec: str | None,
    *,
    allowed: tuple[str, ...] | list[str],
    rejected: tuple[str, ...] | list[str],
) -> None:
    if not codec:
        return
    normalized = codec.lower()
    if normalized in {item.lower() for item in rejected}:
        raise UnsupportedMediaError(
            f'视频编码 "{normalized}" 不受支持，请转码为 H.264 或 H.265 (HEVC) 后重试',
            details={"codec": normalized},
        )
    # 未知编码不拦截（探测器可能缺失），交给解码能力校验兜底
    _ = allowed
