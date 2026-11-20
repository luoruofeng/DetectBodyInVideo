"""模型权重定位与下载（任务 4.2 / 4.3 / 4.5 共用）。

模型文件统一放 ``settings.models_dir``：
- MediaPipe Tasks：``pose_landmarker_{lite,full,heavy}.task``、``hand_landmarker.task``、
  ``face_landmarker.task``，官方 URL 内置，可一键下载。
- ONNX：``rtmpose*.onnx`` / ``rtmo*.onnx`` 需用户从 MMPose 模型库自行下载，
  这里只负责定位并给出可执行的提示（避免内置不稳定第三方直链）。
"""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from app.backend.core.logging import get_logger
from app.backend.core.settings import RESOURCE_ROOT, Settings, get_settings
from app.backend.engine.base import EngineUnavailableError

logger = get_logger("engine.models")

ModelKind = Literal["pose", "hand", "face"]

MEDIAPIPE_BASE = "https://storage.googleapis.com/mediapipe-models"

MEDIAPIPE_MODEL_URLS: dict[str, str] = {
    "pose_landmarker_lite.task": f"{MEDIAPIPE_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{MEDIAPIPE_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{MEDIAPIPE_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "hand_landmarker.task": f"{MEDIAPIPE_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": f"{MEDIAPIPE_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

# MediaPipe 三档模型文件名
POSE_MODEL_FILENAMES: dict[str, str] = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

# ONNX 模型候选文件名（按档位排序，先匹配到的先用）
ONNX_POSE_FILENAMES: dict[str, tuple[str, ...]] = {
    "lite": ("rtmpose-s_256x192.onnx", "rtmpose_t_256x192.onnx", "rtmpose-s.onnx"),
    "full": (
        "rtmpose-m_256x192.onnx",
        "rtmpose_m_256x192.onnx",
        "rtmpose-m.onnx",
        "rtmo-s_640x640.onnx",
    ),
    "heavy": (
        "rtmpose-l_384x288.onnx",
        "rtmpose_l_384x288.onnx",
        "rtmpose-l.onnx",
        "rtmo-m_640x640.onnx",
    ),
}

RTMPOSE_HINT = (
    "请从 MMPose 模型库下载 RTMPose / RTMO 的 ONNX 权重，"
    "命名为 rtmpose-m_256x192.onnx（或 rtmo-s_640x640.onnx）后放入 models/ 目录；"
    "也可在设置 DETECT_BODY_ONNX_POSE_MODEL 中直接指定路径"
)


def models_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    settings.ensure_directories()
    return settings.models_dir


def model_path(filename: str, settings: Settings | None = None) -> Path:
    return models_dir(settings) / filename


def find_model(filename: str, settings: Settings | None = None) -> Path | None:
    settings = settings or get_settings()
    direct = settings.models_dir / filename
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    bundled = RESOURCE_ROOT / "models" / filename
    if bundled.is_file() and bundled.stat().st_size > 0 and bundled != direct:
        return bundled
    return None


def mirror_url(url: str, settings: Settings | None = None, *, use_mirror: bool | None = None) -> str:
    """把官方直链换成镜像前缀（任务 10.4）；镜像未配置时原样返回。"""
    settings = settings or get_settings()
    mirror = (settings.model_mirror_url or "").strip()
    if not mirror or use_mirror is False:
        return url
    if url.startswith(MEDIAPIPE_BASE):
        return mirror.rstrip("/") + url[len(MEDIAPIPE_BASE) :]
    return url


def download_model(
    url: str,
    destination: Path,
    *,
    timeout: float = 60.0,
    progress: Callable[[int, int], None] | None = None,
    settings: Settings | None = None,
    use_mirror: bool | None = None,
) -> Path:
    """下载到临时文件再原子替换，避免半截文件被当成可用模型；支持进度回调。"""
    settings = settings or get_settings()
    url = mirror_url(url, settings, use_mirror=use_mirror)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    logger.info("开始下载模型 %s → %s", url, destination)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - 固定 https 直链
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            if progress is not None:
                progress(downloaded, total)
            with temporary.open("wb") as sink:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    sink.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("模型下载完成：%s（%.1f MB）", destination.name, destination.stat().st_size / 1024**2)
    return destination


def resolve_mediapipe_model(
    kind: ModelKind,
    *,
    settings: Settings | None = None,
    tier: str = "full",
    explicit: str | Path | None = None,
    allow_download: bool | None = None,
) -> Path:
    """定位 MediaPipe 模型；缺失时可自动下载（受 ``auto_download_models`` 控制）。"""
    settings = settings or get_settings()
    if kind == "pose":
        filename = POSE_MODEL_FILENAMES.get(tier, POSE_MODEL_FILENAMES["full"])
    else:
        filename = f"{kind}_landmarker.task"

    candidate = Path(explicit).expanduser() if explicit else settings.models_dir / filename
    if explicit and candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    if not explicit:
        found = find_model(filename, settings)
        if found is not None:
            return found

    if allow_download is None:
        allow_download = settings.auto_download_models
    url = MEDIAPIPE_MODEL_URLS.get(filename)
    if allow_download and url:
        return download_model(url, candidate)

    # 硬件按显存推荐的档位可能高于用户已下载的档位（例如 heavy 推荐但只有 full）。
    # 有任意一档姿态模型就先跑起来，并让调用方把回退写进引擎说明。
    if not explicit and kind == "pose":
        fallback_order = [
            POSE_MODEL_FILENAMES[tier_name]
            for tier_name in ("full", "lite", "heavy")
            if POSE_MODEL_FILENAMES[tier_name] != filename
        ]
        for fallback_name in fallback_order:
            found = find_model(fallback_name, settings)
            if found is not None:
                logger.warning(
                    "缺少 %s 档姿态模型 %s，回退到 %s", tier, filename, found.name
                )
                return found

    raise EngineUnavailableError(
        f"缺少 MediaPipe 模型 {filename}",
        details={
            "expectedPath": str(candidate),
            "downloadUrl": url,
            "autoDownloadEnv": "DETECT_BODY_AUTO_DOWNLOAD_MODELS=1",
            "hint": "运行 scripts/bootstrap 或手动下载到 models/ 目录",
        },
    )


def resolve_onnx_model(
    *,
    settings: Settings | None = None,
    tier: str = "full",
    explicit: str | Path | None = None,
) -> Path:
    settings = settings or get_settings()
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise EngineUnavailableError(
            "指定的 ONNX 模型不存在",
            details={"path": str(candidate), "hint": RTMPOSE_HINT},
        )

    directories = [settings.models_dir, RESOURCE_ROOT / "models"]
    for directory in directories:
        for name in ONNX_POSE_FILENAMES.get(tier, ONNX_POSE_FILENAMES["full"]):
            candidate = directory / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    # 兜底：目录里任意 rtmpose*/rtmo* 权重都能用
    for directory in directories:
        for pattern in ("rtmpose*.onnx", "rtmo*.onnx"):
            for candidate in sorted(directory.glob(pattern)):
                if candidate.stat().st_size > 0:
                    return candidate

    raise EngineUnavailableError(
        "缺少 ONNX 姿态模型（RTMPose / RTMO）",
        details={"modelsDir": str(settings.models_dir), "hint": RTMPOSE_HINT},
    )


__all__ = [
    "MEDIAPIPE_MODEL_URLS",
    "ONNX_POSE_FILENAMES",
    "POSE_MODEL_FILENAMES",
    "RTMPOSE_HINT",
    "download_model",
    "find_model",
    "mirror_url",
    "model_path",
    "models_dir",
    "resolve_mediapipe_model",
    "resolve_onnx_model",
]
