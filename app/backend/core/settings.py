"""全局配置。

所有路径、服务参数集中在此，环境变量前缀 ``DETECT_BODY_``，支持根目录 ``.env``。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/app/backend/core/settings.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resource_root() -> Path:
    """打包后资源根目录；源码运行时就是仓库根（任务 10.1）。"""
    if getattr(sys, "frozen", False):
        bundle = getattr(sys, "_MEIPASS", None)
        return Path(bundle) if bundle else Path(sys.executable).resolve().parent
    return PROJECT_ROOT


RESOURCE_ROOT = _resource_root()


def _default_models_dir() -> Path:
    """开发时用仓库 models/；打包后写用户目录，避免 Program Files 只读。"""
    if getattr(sys, "frozen", False):
        return Path.home() / ".detect-body" / "models"
    return PROJECT_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DETECT_BODY_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 应用标识 ----
    app_name: str = "DetectBodyInVideo"
    version: str = "0.1.0"
    api_prefix: str = "/api"

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )

    # ---- 目录 ----
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".detect-body")
    models_dir: Path = Field(default_factory=_default_models_dir)
    static_dir: Path = Field(default_factory=lambda: RESOURCE_ROOT / "app" / "backend" / "static")
    # 国内网络可换成自建镜像（只需与官方 URL 的 host 对齐替换）
    model_mirror_url: str | None = Field(
        None, description="MediaPipe 模型镜像前缀，如 https://mirror.example.com/mediapipe-models"
    )

    # ---- 日志 ----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_max_bytes: int = 5 * 1024 * 1024
    log_backup_count: int = 3

    # ---- 上传 ----
    max_upload_bytes: int = 8 * 1024**3  # 8 GiB
    chunk_size: int = 8 * 1024 * 1024  # 8 MiB
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
    # 允许的编码（ffprobe / fourcc 归一化后的 codec_name）
    allowed_codecs: tuple[str, ...] = (
        "h264",
        "hevc",
        "h265",
        "avc1",
        "mp4v",
        "mjpeg",
        "mpeg4",
        "prores",
    )
    reject_codecs: tuple[str, ...] = ("av1", "vp9", "theora")

    # ---- 视频处理 ----
    thumbnail_max_width: int = 480
    upload_ttl_hours: int = 24

    # ---- 硬件 / 推理 ----
    max_inference_threads: int = 8
    prefer_model_tier: Literal["auto", "lite", "full", "heavy"] = "auto"

    # ---- 推理运行时（任务 9.1 / 9.3） ----
    runtime_queue_size: int = Field(64, ge=1, le=4096, description="生产者/消费者有界队列长度")
    runtime_batch_size: int = Field(4, ge=1, le=32, description="加速 EP 上的批大小")
    runtime_segment_frames: int = Field(0, ge=0, description="分段长度（帧），0 = 不分段")

    # ---- 深度（Z 轴）后处理 ----
    # 单目视频里 z 只是估计值：靠"固定骨长 + 2D 重投影"把它约束到与画面一致，
    # 再用更弱的 z 平滑压抖动。两项都可关（关掉后输出与旧版本逐位一致）。
    ik_fit_enabled: bool = Field(True, description="骨长约束拟合总开关")
    ik_fit_max_nfev: int = Field(60, ge=1, le=500, description="每帧求解的函数求值上限")
    ik_fit_fov_degrees: float = Field(
        60.0, ge=10, le=150, description="估计的相机水平视场角（仅用于报告与兜底，默认走数据自标定）"
    )
    z_smoothing_scale: float = Field(
        0.3,
        gt=0,
        le=1,
        description="z 轴 min_cutoff 相对 x/y 的系数（z 噪声比 x/y 大一个数量级，故显著更小）",
    )

    # ---- 姿态引擎（任务 4） ----
    pose_engine: Literal["auto", "mediapipe", "onnx"] = "auto"
    auto_download_models: bool = False
    max_poses: int = Field(1, ge=1, le=10, description="最多同时跟踪的人数")
    min_detection_confidence: float = Field(0.5, gt=0, lt=1)
    min_tracking_confidence: float = Field(0.5, gt=0, lt=1)
    smooth_landmarks: bool = True
    mediapipe_gpu_delegate: bool = False
    onnx_pose_model: str | None = Field(default=None, description="RTMPose/RTMO 权重路径（留空自动搜 models/）")
    track_iou_threshold: float = Field(0.3, gt=0, lt=1)
    track_max_age: int = Field(15, ge=0, le=120, description="track 丢失后保留帧数")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_dir(self) -> Path:
        """上传分块与临时文件目录。"""
        return self.data_dir / "temp"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uploads_dir(self) -> Path:
        return self.temp_dir / "uploads"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def library_dir(self) -> Path:
        """已入库视频（按 hash 去重）。"""
        return self.data_dir / "library"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exports_dir(self) -> Path:
        """导出的 JSON 落盘目录（任务 7.2）。"""
        return self.data_dir / "exports"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpoints_dir(self) -> Path:
        """长视频断点续算目录（任务 9.3）。"""
        return self.data_dir / "checkpoints"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.temp_dir,
            self.uploads_dir,
            self.library_dir,
            self.thumbnails_dir,
            self.logs_dir,
            self.exports_dir,
            self.checkpoints_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def is_dev(self) -> bool:
        return self.log_level == "DEBUG"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """测试用：清掉缓存的配置单例。"""
    get_settings.cache_clear()
