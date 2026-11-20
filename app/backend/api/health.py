"""健康检查：桌面壳启动时轮询该接口，确认服务可用。"""

from __future__ import annotations

import platform
import sys
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "name": settings.app_name,
        "version": settings.version,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
