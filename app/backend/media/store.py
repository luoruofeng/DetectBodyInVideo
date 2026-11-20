"""视频入库、hash 去重与索引持久化（任务 3.1）。"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.backend.core.errors import NotFoundError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.media.probe import VideoProbe, generate_thumbnail, probe_video

logger = get_logger("media.store")

INDEX_VERSION = 1


class VideoMetadata(VideoProbe):
    id: str
    filename: str
    sha256: str
    size_bytes: int
    extension: str = ".mp4"
    thumbnail_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VideoStore:
    """入库目录：``library/<id前两位>/<id><ext>``，索引为 ``library/index.json``。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self.settings.ensure_directories()

    # ---- 索引读写 ----
    @property
    def index_path(self) -> Path:
        return self.settings.library_dir / "index.json"

    def _load_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("索引文件损坏，已按空索引处理：%s", self.index_path)
            return {}
        if isinstance(payload, dict) and isinstance(payload.get("videos"), dict):
            return payload["videos"]
        return {}

    def _save_index(self, videos: dict[str, dict]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": INDEX_VERSION, "videos": videos}
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.index_path)

    # ---- 查询 ----
    def list(self) -> list[VideoMetadata]:
        with self._lock:
            entries = self._load_index()
            videos = [VideoMetadata.model_validate(entry) for entry in entries.values()]
        return sorted(videos, key=lambda item: item.created_at, reverse=True)

    def find_by_hash(self, sha256: str) -> VideoMetadata | None:
        with self._lock:
            for entry in self._load_index().values():
                if entry.get("sha256") == sha256:
                    return VideoMetadata.model_validate(entry)
        return None

    def get(self, video_id: str) -> VideoMetadata:
        with self._lock:
            entry = self._load_index().get(video_id)
        if entry is None:
            raise NotFoundError(f"视频不存在或已被清理：{video_id}", details={"videoId": video_id})
        return VideoMetadata.model_validate(entry)

    def path_for(self, video_id: str) -> Path:
        metadata = self.get(video_id)
        path = self._media_path(metadata.id, metadata.extension)
        if not path.exists():
            raise NotFoundError(
                f"视频文件已丢失：{metadata.filename}",
                details={"videoId": video_id, "path": str(path)},
            )
        return path

    def thumbnail_path(self, video_id: str) -> Path:
        return self.settings.thumbnails_dir / f"{video_id}.jpg"

    def _media_path(self, video_id: str, extension: str) -> Path:
        return self.settings.library_dir / video_id[:2] / f"{video_id}{extension}"

    # ---- 入库 ----
    def finalize(
        self,
        temp_path: Path,
        filename: str,
        sha256: str,
        size_bytes: int,
    ) -> tuple[VideoMetadata, bool]:
        """校验并入库；返回 (元数据, 是否命中已有相同文件)。"""
        with self._lock:
            existing = self.find_by_hash(sha256)
            if existing is not None:
                temp_path.unlink(missing_ok=True)
                logger.info("命中已有视频，跳过入库：%s", existing.filename)
                return existing, True

            probe = probe_video(
                temp_path,
                allowed_codecs=self.settings.allowed_codecs,
                rejected_codecs=self.settings.reject_codecs,
            )
            video_id = sha256[:16]
            extension = Path(filename).suffix.lower() or ".mp4"
            destination = self._media_path(video_id, extension)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), destination)

            thumbnail = self.thumbnail_path(video_id)
            thumbnail_ok = generate_thumbnail(
                destination,
                thumbnail,
                probe=probe,
                max_width=self.settings.thumbnail_max_width,
            )

            metadata = VideoMetadata(
                **probe.model_dump(),
                id=video_id,
                filename=filename,
                sha256=sha256,
                size_bytes=size_bytes,
                extension=extension,
                thumbnail_url=f"/api/videos/{video_id}/thumbnail" if thumbnail_ok else None,
            )
            index = self._load_index()
            index[video_id] = metadata.model_dump(mode="json")
            self._save_index(index)
            logger.info(
                "入库完成：%s（%sx%s, %.1fs, %s fps）",
                filename,
                metadata.display_width,
                metadata.display_height,
                metadata.duration_s,
                metadata.fps,
            )
            return metadata, False
