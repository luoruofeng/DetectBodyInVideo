"""视频接入接口（任务 3）：上传 / 分块上传 / 元信息 / 缩略图 / 分析计划。"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, File, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.backend.core.errors import NotFoundError, PayloadTooLargeError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.media.probe import VideoProbe
from app.backend.media.sampling import AnalysisPlan, SamplingConfig, build_analysis_plan
from app.backend.media.store import VideoMetadata, VideoStore
from app.backend.media.uploads import ChunkedUploadManager, UploadSession
from app.backend.media.validation import validate_extension, validate_size

router = APIRouter(prefix="/videos", tags=["videos"])
logger = get_logger("api.videos")

VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}

READ_BLOCK = 1024 * 1024


class UploadResponse(BaseModel):
    video: VideoMetadata
    deduped: bool = False


class UploadInitRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    chunk_size: int | None = Field(default=None, ge=1024)


def _store(request: Request) -> VideoStore:
    return request.app.state.video_store


def _uploads(request: Request) -> ChunkedUploadManager:
    return request.app.state.upload_manager


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _digest_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(READ_BLOCK):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def _finalize(
    request: Request,
    temp_path: Path,
    filename: str,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> UploadResponse:
    settings = _settings(request)
    if sha256 is None or size_bytes is None:
        sha256, size_bytes = _digest_file(temp_path)
    validate_size(size_bytes, settings.max_upload_bytes)
    metadata, deduped = _store(request).finalize(temp_path, filename, sha256, size_bytes)
    return UploadResponse(video=metadata, deduped=deduped)


# --------------------------------------------------------------------------- #
# 库
# --------------------------------------------------------------------------- #
@router.get("", response_model=list[VideoMetadata])
def list_videos(request: Request) -> list[VideoMetadata]:
    return _store(request).list()


# --------------------------------------------------------------------------- #
# 小文件：一次性上传（前端用 XHR 拿上传进度）
# --------------------------------------------------------------------------- #
@router.post("/upload", response_model=UploadResponse)
def upload_video(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    settings = _settings(request)
    filename = Path(file.filename or "video.mp4").name
    suffix = validate_extension(filename, settings.allowed_extensions)

    temporary = settings.temp_dir / f"upload-{uuid.uuid4().hex}{suffix}"
    hasher = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as sink:
            while block := file.file.read(READ_BLOCK):
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise PayloadTooLargeError(
                        f"文件超过上限 {settings.max_upload_bytes / 1024**3:.1f} GiB",
                        details={"maxBytes": settings.max_upload_bytes},
                    )
                hasher.update(block)
                sink.write(block)
        return _finalize(
            request,
            temporary,
            filename,
            sha256=hasher.hexdigest(),
            size_bytes=size,
        )
    finally:
        file.file.close()
        temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 大文件：分块上传 + 断点续传
# --------------------------------------------------------------------------- #
@router.post("/uploads", response_model=UploadSession)
def create_upload(request: Request, payload: UploadInitRequest) -> UploadSession:
    settings = _settings(request)
    validate_extension(payload.filename, settings.allowed_extensions)
    return _uploads(request).create(payload.filename, payload.size_bytes, payload.chunk_size)


@router.get("/uploads/{upload_id}", response_model=UploadSession)
def get_upload(request: Request, upload_id: str) -> UploadSession:
    return _uploads(request).status(upload_id)


@router.put("/uploads/{upload_id}/chunks/{index}", response_model=UploadSession)
def put_chunk(
    request: Request,
    upload_id: str,
    index: int,
    data: bytes = Body(..., media_type="application/octet-stream"),
) -> UploadSession:
    return _uploads(request).append_chunk(upload_id, index, data)


@router.post("/uploads/{upload_id}/complete", response_model=UploadResponse)
def complete_upload(request: Request, upload_id: str) -> UploadResponse:
    manager = _uploads(request)
    assembled, session = manager.assemble(upload_id)
    try:
        return _finalize(request, assembled, session.filename)
    finally:
        manager.abort(upload_id)


@router.delete("/uploads/{upload_id}", status_code=204)
def abort_upload(request: Request, upload_id: str) -> None:
    _uploads(request).abort(upload_id)


# --------------------------------------------------------------------------- #
# 单个视频
# --------------------------------------------------------------------------- #
@router.get("/{video_id}", response_model=VideoMetadata)
def get_video(request: Request, video_id: str) -> VideoMetadata:
    return _store(request).get(video_id)


@router.get("/{video_id}/thumbnail")
def get_thumbnail(request: Request, video_id: str) -> FileResponse:
    path = _store(request).thumbnail_path(video_id)
    if not path.exists():
        raise NotFoundError("缩略图不存在", details={"videoId": video_id})
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/{video_id}/file")
def get_video_file(request: Request, video_id: str) -> FileResponse:
    store = _store(request)
    metadata = store.get(video_id)
    path = store.path_for(video_id)
    return FileResponse(
        path,
        media_type=VIDEO_MEDIA_TYPES.get(metadata.extension, "application/octet-stream"),
        filename=metadata.filename,
        content_disposition_type="inline",
    )


@router.post("/{video_id}/plan", response_model=AnalysisPlan)
def plan_analysis(
    request: Request,
    video_id: str,
    config: SamplingConfig | None = None,
) -> AnalysisPlan:
    store = _store(request)
    metadata = store.get(video_id)
    probe = VideoProbe.model_validate(metadata.model_dump())
    return build_analysis_plan(store.path_for(video_id), probe, config or SamplingConfig())
