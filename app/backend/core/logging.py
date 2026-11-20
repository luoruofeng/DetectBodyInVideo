"""统一日志：控制台 + 轮转文件（~/.detect-body/logs/app.log）。"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from app.backend.core.settings import Settings, get_settings

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(process)d:%(threadName)s] %(name)s:%(lineno)d | %(message)s"
_CONFIGURED_FLAG = "_detect_body_logging_configured"


def setup_logging(settings: Settings | None = None, *, force: bool = False) -> logging.Logger:
    """初始化根 logger，重复调用安全（除非 ``force=True``）。"""
    settings = settings or get_settings()
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_FLAG, False) and not force:
        return logging.getLogger(settings.app_name)

    level = getattr(logging, settings.log_level, logging.INFO)
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # 打包成窗口程序（PyInstaller runw）后 sys.stdout 为 None，
    # 此时挂 StreamHandler 会让每条日志都走 logging 的异常兜底分支，直接跳过控制台。
    if sys.stdout is not None:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(console)

    log_path: Path = settings.logs_dir / "app.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
    except OSError:  # 只读磁盘 / 权限不足时降级为仅控制台
        root.warning("无法写入日志文件 %s，已降级为仅控制台输出", log_path)

    logging.captureWarnings(True)
    # 三方库太吵，压到 WARNING
    for noisy in ("uvicorn.access", "mediapipe", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setattr(root, _CONFIGURED_FLAG, True)
    logging.getLogger(settings.app_name).debug("日志系统就绪，文件输出：%s", log_path)
    return logging.getLogger(settings.app_name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"detect_body.{name}")
