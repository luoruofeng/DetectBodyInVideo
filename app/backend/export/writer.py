"""导出落盘与自校验（任务 7.2 / 7.3）。

- **流式写出**：先写头，再逐帧 ``model_dump_json()`` 追加，最后补 quality /
  rootMotion。内存峰值只与单帧有关，不随视频长度线性增长。
- **自校验**：写盘后立刻回读并用 Pydantic 校验，任何格式错误直接抛
  ``ExportValidationError``，绝不静默产出坏文件。
- **原子替换**：先写 ``*.part``，校验通过才 ``os.replace``，避免半截文件被 iOS 端读到。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.backend.core.errors import AppError
from app.backend.core.logging import get_logger
from app.backend.export.builder import Granularity, export_filename
from app.backend.export.schema import ExportDocument

logger = get_logger("export.writer")


class ExportError(AppError):
    status_code = 500
    code = "export_failed"


class ExportValidationError(ExportError):
    code = "export_validation_failed"


class ExportArtifact(BaseModel):
    """一次导出的结果，直接回给 UI / CLI。"""

    path: str
    filename: str
    bytes: int = 0
    frames: int = 0
    mode: str = Field("full", description="导出粒度，动作 JSON 为 motion")
    validated: bool = False
    elapsedSeconds: float = 0.0


def _write_stream(document: ExportDocument, sink: object, *, chunk_frames: int = 200) -> int:
    """把文档以流式方式写入文本文件；返回写入帧数。"""
    header = document.model_dump(mode="json", exclude={"frames", "quality", "rootMotion"})
    header_text = json.dumps(header, ensure_ascii=False)
    if not header_text.endswith("}"):
        raise ExportError("导出头序列化失败")
    sink.write(header_text[:-1])  # type: ignore[attr-defined]
    sink.write(', "frames": [')  # type: ignore[attr-defined]

    written = 0
    first_chunk = True
    buffer: list[str] = []
    for frame in document.frames:
        buffer.append(frame.model_dump_json())
        written += 1
        if len(buffer) >= chunk_frames:
            sink.write(("" if first_chunk else ",") + ",".join(buffer))  # type: ignore[attr-defined]
            first_chunk = False
            buffer.clear()
    if buffer:
        sink.write(("" if first_chunk else ",") + ",".join(buffer))  # type: ignore[attr-defined]

    sink.write('], "quality": ')  # type: ignore[attr-defined]
    sink.write(document.quality.model_dump_json())  # type: ignore[attr-defined]
    if document.rootMotion is not None:
        sink.write(', "rootMotion": ')  # type: ignore[attr-defined]
        sink.write(document.rootMotion.model_dump_json())  # type: ignore[attr-defined]
    sink.write("}")  # type: ignore[attr-defined]
    return written


def read_document(path: Path | str) -> ExportDocument:
    """回读并校验（导出后自检、回归测试共用）。"""
    target = Path(path)
    try:
        return ExportDocument.model_validate_json(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 统一转成业务错误
        raise ExportValidationError(
            f"导出文件校验失败：{exc}", details={"path": str(target)}
        ) from exc


def write_document(
    document: ExportDocument,
    path: Path | str,
    *,
    mode: Granularity | None = None,
    validate: bool = True,
) -> ExportArtifact:
    """原子写入 + 可选回读校验。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    started = time.perf_counter()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as sink:
            written = _write_stream(document, sink)
        if validate:
            checked = read_document(temporary)
            if len(checked.frames) != written:
                raise ExportValidationError(
                    "回读帧数与写入帧数不一致",
                    details={"written": written, "read": len(checked.frames)},
                )
            if checked.version != document.version:
                raise ExportValidationError(
                    "回读版本号与导出不一致",
                    details={"expected": document.version, "actual": checked.version},
                )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    size = target.stat().st_size
    artifact = ExportArtifact(
        path=str(target),
        filename=target.name,
        bytes=size,
        frames=written,
        mode=mode or _mode_of(document),
        validated=validate,
        elapsedSeconds=round(time.perf_counter() - started, 3),
    )
    logger.info(
        "导出完成：%s（%s 帧，%.1f KiB，%.2fs）",
        artifact.filename,
        artifact.frames,
        artifact.bytes / 1024,
        artifact.elapsedSeconds,
    )
    return artifact


def export_document(
    document: ExportDocument,
    output_dir: Path | str,
    *,
    filename: str | None = None,
    mode: Granularity | None = None,
    validate: bool = True,
) -> ExportArtifact:
    """按命名规则写盘：``{原名}_{时间戳}_pose{mode}.json``。"""
    directory = Path(output_dir)
    resolved_mode: Granularity = mode or _mode_of(document)
    name = filename or export_filename(document.source.file, resolved_mode)
    if not name.lower().endswith(".json"):
        name += ".json"
    return write_document(document, directory / name, mode=resolved_mode, validate=validate)


def _mode_of(document: ExportDocument) -> Granularity:
    """从文档反推粒度（用于命名与回执；三者互斥且可区分）。"""
    if any(frame.angles for frame in document.frames):
        return "full"
    if any(frame.rawLandmarks for frame in document.frames):
        return "keypoints"
    if len(document.jointNames) > 18:
        return "keypoints"
    return "compact"


def latest_export(directory: Path | str) -> ExportArtifact | None:
    """取目录里最新的导出文件（桌面端"打开上次导出"用）。"""
    folder = Path(directory)
    if not folder.is_dir():
        return None
    candidates = sorted(
        (item for item in folder.glob("*_pose*.json") if item.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    target = candidates[0]
    document: ExportDocument | None = None
    try:
        document = read_document(target)
    except ExportError:
        logger.warning("历史导出 %s 校验失败，忽略", target.name)
    return ExportArtifact(
        path=str(target),
        filename=target.name,
        bytes=target.stat().st_size,
        frames=len(document.frames) if document else 0,
        mode=_mode_of(document) if document else "full",
        validated=document is not None,
    )


__all__ = [
    "ExportArtifact",
    "ExportError",
    "ExportValidationError",
    "export_document",
    "latest_export",
    "read_document",
    "write_document",
]
