"""导出文件下载与 schema 存档接口。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.backend.core.errors import NotFoundError
from app.backend.core.settings import Settings
from app.backend.export.schema import SCHEMA_FILE

router = APIRouter(prefix="/exports", tags=["exports"])


class ExportFile(BaseModel):
    name: str
    bytes: int
    modified: float
    downloadUrl: str


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/schema", response_class=FileResponse)
def download_schema() -> FileResponse:
    """schema.json 存档：iOS 侧可直接用 JSON Schema 校验导出文件。"""
    if not SCHEMA_FILE.is_file():
        from app.backend.export.schema import write_schema_json  # noqa: PLC0415

        write_schema_json()
    return FileResponse(
        SCHEMA_FILE, media_type="application/json", filename="pose-export-schema.json"
    )


@router.get("", response_model=list[ExportFile])
def list_exports(request: Request) -> list[ExportFile]:
    folder = _settings(request).exports_dir
    if not folder.is_dir():
        return []
    items = sorted(
        (path for path in folder.glob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [
        ExportFile(
            name=path.name,
            bytes=path.stat().st_size,
            modified=round(path.stat().st_mtime, 3),
            downloadUrl=f"/api/exports/{path.name}",
        )
        for path in items
    ]


@router.get("/{filename}", response_class=FileResponse)
def download_export(request: Request, filename: str) -> FileResponse:
    safe = Path(filename).name
    if safe != filename or not safe.lower().endswith(".json"):
        raise NotFoundError("导出文件不存在", details={"filename": filename})
    path = _settings(request).exports_dir / safe
    if not path.is_file():
        raise NotFoundError("导出文件不存在", details={"filename": filename})
    return FileResponse(
        path, media_type="application/json", filename=safe, content_disposition_type="attachment"
    )


__all__ = ["ExportFile", "router"]
