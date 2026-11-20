"""推理运行时接口：界面内安装 / 回滚 GPU 加速后端（onnxruntime-gpu / directml）。

安装走独立目录（``<data_dir>/runtime/<variant>``），不触碰当前 site-packages，
因此可以在应用运行期间安全下载；切换后由前端引导重启进程生效。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.backend.engine.accelerator_install import (
    AcceleratorInstallManager,
    AcceleratorPayload,
    InstallTask,
    schedule_restart,
)

router = APIRouter(prefix="/runtime", tags=["runtime"])


class InstallRequest(BaseModel):
    variant: Literal["cuda", "directml"]


def _manager(request: Request) -> AcceleratorInstallManager:
    return request.app.state.accelerators


@router.get("/accelerators", response_model=AcceleratorPayload)
def get_accelerators(request: Request) -> AcceleratorPayload:
    """可安装的加速后端 + 当前生效状态 + 正在进行的安装任务。"""
    return _manager(request).status()


@router.post("/accelerators/install", response_model=InstallTask, status_code=201)
def install_accelerator(request: Request, payload: InstallRequest) -> InstallTask:
    """启动后台安装；前端轮询 ``GET /runtime/accelerators`` 看 ``task``。"""
    return _manager(request).start(payload.variant)


@router.post("/accelerators/install/cancel", response_model=InstallTask | None)
def cancel_install(request: Request) -> InstallTask | None:
    return _manager(request).cancel()


@router.post("/accelerators/{variant}/activate", status_code=202)
def activate_accelerator(request: Request, variant: str) -> dict[str, bool]:
    """在已安装的多个运行时之间切换（重启后生效）。"""
    _manager(request).activate_variant(variant)
    return {"ok": True}


@router.delete("/accelerators/{variant}", status_code=204)
def remove_accelerator(request: Request, variant: str) -> Response:
    """删除已安装的运行时目录并回滚到 CPU 版 onnxruntime。"""
    _manager(request).remove(variant)
    return Response(status_code=204)


@router.post("/restart", status_code=202)
def restart_app() -> dict[str, bool]:
    """重启进程：onnxruntime 的 DLL 已被加载，无法在进程内热切换。"""
    schedule_restart()
    return {"restarting": True}


__all__ = ["InstallRequest", "router"]
