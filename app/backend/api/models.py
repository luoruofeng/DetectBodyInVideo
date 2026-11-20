"""模型检测与下载接口（任务 10.4）。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.backend.engine.model_downloads import (
    DownloadTask,
    ModelDownloadManager,
    ModelsPayload,
)

router = APIRouter(prefix="/models", tags=["models"])


class DownloadRequest(BaseModel):
    kind: Literal["pose", "hand", "face"] = "pose"
    tier: str = Field("full", description="lite / full / heavy")
    use_mirror: bool = False


def _manager(request: Request) -> ModelDownloadManager:
    return request.app.state.model_downloads


@router.get("", response_model=ModelsPayload)
def list_models(request: Request) -> ModelsPayload:
    """模型是否就绪 + 正在下载的任务（前端首启检查）。"""
    return _manager(request).status()


@router.post("/download", response_model=DownloadTask)
def download_model(request: Request, payload: DownloadRequest) -> DownloadTask:
    return _manager(request).start(
        kind=payload.kind,
        tier=payload.tier,
        use_mirror=payload.use_mirror,
    )


__all__ = ["DownloadRequest", "router"]
