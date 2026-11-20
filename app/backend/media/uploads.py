"""分块上传会话（任务 3.1）：进度、断点续传、过期清理。

每个会话一个目录：``temp/uploads/<upload_id>/session.json`` + ``NNNNN.part``。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from app.backend.core.errors import NotFoundError, ValidationFailedError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.media.validation import validate_size

logger = get_logger("media.uploads")

MIN_CHUNK_SIZE = 256 * 1024
MAX_CHUNK_SIZE = 32 * 1024 * 1024
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class UploadSession(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    chunk_size: int
    total_chunks: int
    received_chunks: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def received_bytes(self) -> int:
        return len(self.received_chunks) * self.chunk_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress(self) -> float:
        if self.total_chunks <= 0:
            return 0.0
        return round(len(self.received_chunks) / self.total_chunks, 4)


class ChunkedUploadManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self.settings.ensure_directories()

    # ---- 路径 ----
    def _dir(self, upload_id: str) -> Path:
        if not _UPLOAD_ID_RE.match(upload_id):
            raise NotFoundError("上传会话不存在", details={"uploadId": upload_id})
        return self.settings.uploads_dir / upload_id

    def _session_file(self, upload_id: str) -> Path:
        return self._dir(upload_id) / "session.json"

    def _write_session(self, session: UploadSession) -> None:
        path = self._session_file(session.upload_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    # ---- 生命周期 ----
    def create(
        self,
        filename: str,
        size_bytes: int,
        chunk_size: int | None = None,
    ) -> UploadSession:
        safe_name = Path(filename).name.strip() or "video.mp4"
        validate_size(size_bytes, self.settings.max_upload_bytes)

        resolved = int(chunk_size or self.settings.chunk_size)
        resolved = max(MIN_CHUNK_SIZE, min(resolved, MAX_CHUNK_SIZE))
        total = max(1, math.ceil(size_bytes / resolved))

        session = UploadSession(
            upload_id=uuid.uuid4().hex,
            filename=safe_name,
            size_bytes=size_bytes,
            chunk_size=resolved,
            total_chunks=total,
        )
        with self._lock:
            self._dir(session.upload_id).mkdir(parents=True, exist_ok=True)
            self._write_session(session)
        logger.info("创建上传会话 %s：%s（%s 块）", session.upload_id, safe_name, total)
        return session

    def status(self, upload_id: str) -> UploadSession:
        path = self._session_file(upload_id)
        if not path.exists():
            raise NotFoundError("上传会话不存在或已过期", details={"uploadId": upload_id})
        try:
            return UploadSession.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValidationFailedError("上传会话数据损坏，请重新上传") from exc

    def append_chunk(self, upload_id: str, index: int, data: bytes) -> UploadSession:
        with self._lock:
            session = self.status(upload_id)
            if index < 0 or index >= session.total_chunks:
                raise ValidationFailedError(
                    f"分块序号越界：{index}（共 {session.total_chunks} 块）",
                    details={"index": index, "totalChunks": session.total_chunks},
                )

            expected = (
                session.chunk_size
                if index < session.total_chunks - 1
                else session.size_bytes - session.chunk_size * (session.total_chunks - 1)
            )
            if len(data) != expected:
                raise ValidationFailedError(
                    f"分块大小不符：第 {index} 块应为 {expected} 字节，实际 {len(data)} 字节",
                    details={"index": index, "expectedBytes": expected, "receivedBytes": len(data)},
                )

            chunk_path = self._dir(upload_id) / f"{index:05d}.part"
            temporary = chunk_path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(chunk_path)

            if index not in session.received_chunks:
                session.received_chunks = sorted({*session.received_chunks, index})
            self._write_session(session)
            return session

    def assemble(self, upload_id: str) -> tuple[Path, UploadSession]:
        with self._lock:
            session = self.status(upload_id)
            missing = sorted(set(range(session.total_chunks)) - set(session.received_chunks))
            if missing:
                raise ValidationFailedError(
                    f"还缺少 {len(missing)} 个分块，无法合并",
                    details={"missingChunks": missing[:20], "missingCount": len(missing)},
                )

            directory = self._dir(upload_id)
            assembled = directory / "assembled.part"
            with assembled.open("wb") as sink:
                for index in range(session.total_chunks):
                    with (directory / f"{index:05d}.part").open("rb") as source:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)

            actual = assembled.stat().st_size
            if actual != session.size_bytes:
                raise ValidationFailedError(
                    f"合并后大小与声明不一致（{actual} ≠ {session.size_bytes} 字节）",
                    details={"actualBytes": actual, "expectedBytes": session.size_bytes},
                )
            return assembled, session

    def abort(self, upload_id: str) -> None:
        with self._lock:
            directory = self._dir(upload_id)
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
                logger.info("已清理上传会话 %s", upload_id)

    def cleanup_expired(self) -> int:
        """删除超过 TTL 的残留会话（进程重启/用户关闭页面后产生）。"""
        if not self.settings.uploads_dir.exists():
            return 0
        deadline = datetime.now(timezone.utc) - timedelta(hours=self.settings.upload_ttl_hours)
        removed = 0
        for directory in self.settings.uploads_dir.iterdir():
            if not directory.is_dir():
                continue
            session_file = directory / "session.json"
            try:
                session = UploadSession.model_validate_json(session_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
                continue
            if session.created_at < deadline:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("清理过期上传会话 %s 个", removed)
        return removed
