"""硬件能力接口（任务 2.4）。"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request

from app.backend.core.settings import Settings
from app.backend.engine.acceleration import HardwareReport, build_hardware_report

router = APIRouter(prefix="/hardware", tags=["hardware"])

# 探测会调用 nvidia-smi / sysctl 等外部进程，短时间内复用结果
_CACHE_TTL_SECONDS = 60.0
_cache: dict[str, object] = {"at": 0.0, "report": None}


def _get_report(settings: Settings, refresh: bool) -> HardwareReport:
    now = time.monotonic()
    cached = _cache["report"]
    if not refresh and isinstance(cached, HardwareReport) and now - float(_cache["at"]) < _CACHE_TTL_SECONDS:  # type: ignore[arg-type]
        return cached
    report = build_hardware_report(settings)
    _cache.update({"at": now, "report": report})
    return report


@router.get("", response_model=HardwareReport)
def get_hardware(
    request: Request,
    refresh: bool = Query(False, description="跳过缓存重新探测"),
) -> HardwareReport:
    return _get_report(request.app.state.settings, refresh)


def reset_cache() -> None:
    _cache.update({"at": 0.0, "report": None})
