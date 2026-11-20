"""首启模型检测与后台下载（任务 10.4）。

- 只内置 MediaPipe 官方直链（稳定），ONNX 权重仍由用户自行放入 models/；
- 下载线程带进度，前端轮询 ``status()``；支持 ``DETECT_BODY_MODEL_MIRROR_URL`` 镜像；
- 先写 ``.part`` 再原子替换，进度界面不会把半截文件当模型。
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.errors import ValidationFailedError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.models import (
    MEDIAPIPE_MODEL_URLS,
    POSE_MODEL_FILENAMES,
    download_model,
    find_model,
    mirror_url,
)

logger = get_logger("engine.model_downloads")

ModelKind = Literal["pose", "hand", "face"]
DownloadState = Literal["queued", "running", "done", "error"]


class ModelFileInfo(BaseModel):
    kind: ModelKind
    tier: str | None = None
    filename: str
    present: bool
    size_bytes: int = 0
    path: str
    url: str | None = None
    mirrorUrl: str | None = None


class DownloadTask(BaseModel):
    id: str
    filename: str
    state: DownloadState = "queued"
    downloaded: int = 0
    total: int = 0
    percent: float = 0.0
    error: str | None = None
    mirror: bool = False
    startedAt: float = Field(default_factory=time.time)
    finishedAt: float | None = None


class ModelsPayload(BaseModel):
    models: list[ModelFileInfo] = Field(default_factory=list)
    poseReady: bool = False
    mirrorConfigured: bool = False
    downloads: list[DownloadTask] = Field(default_factory=list)


def model_inventory(settings: Settings) -> list[ModelFileInfo]:
    entries: list[ModelFileInfo] = []
    items: list[tuple[ModelKind, str | None, str]] = [
        ("pose", tier, name) for tier, name in POSE_MODEL_FILENAMES.items()
    ]
    items.extend(
        [
            ("hand", None, "hand_landmarker.task"),
            ("face", None, "face_landmarker.task"),
        ]
    )
    for kind, tier, filename in items:
        found = find_model(filename, settings)
        url = MEDIAPIPE_MODEL_URLS.get(filename)
        entries.append(
            ModelFileInfo(
                kind=kind,
                tier=tier,
                filename=filename,
                present=found is not None,
                size_bytes=found.stat().st_size if found else 0,
                path=str(found or (settings.models_dir / filename)),
                url=url,
                mirrorUrl=mirror_url(url, settings) if url else None,
            )
        )
    return entries


class ModelDownloadManager:
    """进程内下载任务表；同一文件同时只跑一个任务。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tasks: dict[str, DownloadTask] = {}
        self._lock = threading.RLock()

    def status(self) -> ModelsPayload:
        with self._lock:
            tasks = [task.model_copy() for task in self._tasks.values()]
        models = model_inventory(self.settings)
        return ModelsPayload(
            models=models,
            poseReady=any(item.present for item in models if item.kind == "pose"),
            mirrorConfigured=bool((self.settings.model_mirror_url or "").strip()),
            downloads=sorted(tasks, key=lambda item: item.startedAt, reverse=True)[:6],
        )

    def _filename(self, kind: ModelKind, tier: str) -> str:
        if kind == "pose":
            filename = POSE_MODEL_FILENAMES.get(tier)
            if filename is None:
                raise ValidationFailedError(
                    "未知的模型档位", details={"tier": tier, "allowed": list(POSE_MODEL_FILENAMES)}
                )
            return filename
        if kind in {"hand", "face"}:
            return f"{kind}_landmarker.task"
        raise ValidationFailedError("未知的模型类型", details={"kind": kind})

    def start(
        self, *, kind: ModelKind = "pose", tier: str = "full", use_mirror: bool = False
    ) -> DownloadTask:
        filename = self._filename(kind, tier)
        url = MEDIAPIPE_MODEL_URLS.get(filename)
        if not url:
            raise ValidationFailedError("该模型没有内置下载地址", details={"filename": filename})
        target = self.settings.models_dir / filename
        with self._lock:
            for task in self._tasks.values():
                if task.filename == filename and task.state in {"queued", "running"}:
                    return task.model_copy()
            task = DownloadTask(
                id=uuid.uuid4().hex[:10],
                filename=filename,
                mirror=use_mirror,
            )
            self._tasks[task.id] = task
            self._prune()
        thread = threading.Thread(
            target=self._run, args=(task, url, target, use_mirror), name=f"download-{filename}", daemon=True
        )
        thread.start()
        return task.model_copy()

    def _run(self, task: DownloadTask, url: str, target: Path, use_mirror: bool) -> None:
        def progress(downloaded: int, total: int) -> None:
            with self._lock:
                task.downloaded = downloaded
                task.total = total
                task.percent = round(downloaded / total * 100, 1) if total > 0 else 0.0

        with self._lock:
            task.state = "running"
        try:
            download_model(
                url,
                target,
                progress=progress,
                settings=self.settings,
                use_mirror=use_mirror,
            )
        except Exception as exc:  # noqa: BLE001 - 后台线程记录错误给前端
            logger.warning("模型下载失败 %s：%s", task.filename, exc)
            with self._lock:
                task.state = "error"
                task.error = str(exc)
                task.finishedAt = time.time()
            return
        with self._lock:
            task.state = "done"
            task.percent = 100.0
            task.finishedAt = time.time()
        logger.info("模型下载完成：%s", task.filename)

    def _prune(self) -> None:
        finished = [task for task in self._tasks.values() if task.state in {"done", "error"}]
        finished.sort(key=lambda item: item.finishedAt or 0.0, reverse=True)
        for task in finished[6:]:
            self._tasks.pop(task.id, None)


__all__ = [
    "DownloadTask",
    "ModelDownloadManager",
    "ModelFileInfo",
    "ModelsPayload",
    "model_inventory",
]
