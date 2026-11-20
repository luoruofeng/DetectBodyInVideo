"""FastAPI 应用入口。

开发：``uvicorn app.backend.main:app --reload``
桌面壳：``app.desktop.main`` 会在子线程里启动本应用。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.backend.api import api_router
from app.backend.core.errors import AppError
from app.backend.core.logging import setup_logging
from app.backend.core.settings import Settings, get_settings
from app.backend.engine.model_downloads import ModelDownloadManager
from app.backend.jobs import AnalysisJobManager
from app.backend.media.store import VideoStore
from app.backend.media.uploads import ChunkedUploadManager

logger = logging.getLogger("detect_body.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_directories()
    uploads: ChunkedUploadManager = app.state.upload_manager
    uploads.cleanup_expired()
    logger.info("%s %s 已启动，数据目录 %s", settings.app_name, settings.version, settings.data_dir)
    yield
    logger.info("服务已关闭")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)

    app = FastAPI(
        title="DetectBodyInVideo API",
        version=settings.version,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=None,
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    settings.ensure_directories()
    app.state.video_store = VideoStore(settings)
    app.state.upload_manager = ChunkedUploadManager(settings)
    app.state.analysis_jobs = AnalysisJobManager(settings, app.state.video_store)
    app.state.model_downloads = ModelDownloadManager(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.include_router(api_router, prefix=settings.api_prefix)

    # 生产模式：前端构建产物由后端托管（任务 10.1）；开发模式走 Vite。
    if settings.static_dir.is_dir() and any(settings.static_dir.iterdir()):
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
        logger.info("已挂载前端静态目录 %s", settings.static_dir)

    return app


app = create_app()
