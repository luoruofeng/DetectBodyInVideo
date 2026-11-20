"""API 路由聚合。"""

from __future__ import annotations

from fastapi import APIRouter

from app.backend.api.analysis import router as analysis_router
from app.backend.api.exports import router as exports_router
from app.backend.api.hardware import router as hardware_router
from app.backend.api.health import router as health_router
from app.backend.api.models import router as models_router
from app.backend.api.runtime import router as runtime_router
from app.backend.api.videos import router as videos_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(hardware_router)
api_router.include_router(videos_router)
api_router.include_router(analysis_router)
api_router.include_router(exports_router)
api_router.include_router(models_router)
api_router.include_router(runtime_router)

__all__ = ["api_router"]
