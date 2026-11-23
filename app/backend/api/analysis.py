"""分析任务与导出接口（任务 8 的后端支撑）。

- ``POST /analysis`` 起任务，返回 job id；前端轮询 ``GET /analysis/{id}`` 拿真实进度；
- ``GET /analysis/{id}/frames`` 拿逐帧 2D 关键点 + 米制关节 + 角度（骨架叠加 / 3D 预览）；
- ``POST /analysis/{id}/export`` 按粒度导出，回执含文件路径与下载地址。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.backend.core.errors import ValidationFailedError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import BoundingBox
from app.backend.export import (
    ExportArtifact,
    ExportConfig,
    Granularity,
    MotionClipConfig,
    build_motion_clip,
    document_from_analysis,
    export_document,
    motion_filename,
    source_from_probe,
    write_motion_clip,
)
from app.backend.export.motion_clip import MOTION_SCHEMA_VERSION
from app.backend.export.schema import SCHEMA_VERSION
from app.backend.jobs import AnalysisJobManager, JobStatus
from app.backend.perf.checkpoint import clear_checkpoints
from app.backend.pipeline import AnalysisOptions
from app.backend.skeleton.live import LiveFrameJoints

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = get_logger("api.analysis")

DEFAULT_FRAME_LIMIT = 2000
MAX_FRAME_LIMIT = 8000


class StartAnalysisRequest(BaseModel):
    video_id: str = Field(min_length=1)
    options: AnalysisOptions = Field(
        default_factory=AnalysisOptions, description="采样 / 后处理 / 骨骼 / 运行时配置"
    )


class PreviewPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


class PreviewJoint(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 0.0


class PreviewFrame(BaseModel):
    index: int
    t: float
    valid: bool
    interpolated: bool = False
    landmarks: list[PreviewPoint] = Field(default_factory=list)
    joints: dict[str, PreviewJoint] = Field(default_factory=dict)
    angles: dict[str, float] = Field(default_factory=dict)
    bbox: BoundingBox | None = None
    hipHeight: float | None = Field(
        None,
        description="髋到最低着地点（踝/脚尖）的竖直距离（米，已做零相位平滑）。"
        "3D 预览用它把脚底放到地面；缺省表示该帧没有可信的着地参考",
    )
    hipOffsetX: float = Field(
        0.0,
        description="髋相对双脚水平基准点的横向偏移（米，已做零相位平滑）。"
        "3D 预览把它加到髋位置上，脚才不会跟着髋在地面上滑动；缺省 0 表示没有水平锚定",
    )
    hipOffsetZ: float = Field(
        0.0,
        description="髋相对双脚水平基准点的前后偏移（米，深度的估计值）。含义同 hipOffsetX",
    )


class FramesPayload(BaseModel):
    jobId: str
    fps: float
    stride: int
    totalFrames: int
    offset: int
    limit: int
    hasMore: bool
    jointNames: list[str]
    frames: list[PreviewFrame] = Field(default_factory=list)


def _live_joints(entry: LiveFrameJoints, names: Sequence[str]) -> dict[str, PreviewJoint]:
    """把推理中缓存的扁平坐标还原成接口模型（只对请求窗口内的帧做，不占额外内存）。

    可见度为 0 的关节（该帧没检测到对应关键点）直接省略，避免 3D 预览把原点当关节画。
    """
    joints: dict[str, PreviewJoint] = {}
    for name, x, y, z, visibility in entry.as_joints(names):
        if visibility <= 0:
            continue
        joints[name] = PreviewJoint(
            x=round(x, 5),
            y=round(y, 5),
            z=round(z, 5),
            visibility=round(float(visibility), 4),
        )
    return joints


class MotionExportOptions(BaseModel):
    """动作 JSON 的转换参数（默认即 iOS App 的 10Hz / hip-local 约定）。"""

    sample_interval_ms: int = Field(100, ge=10, le=1000)
    loop: bool = False
    exercise_id: str | None = None
    name: str | None = None
    name_zh: str | None = None


class ExportRequest(BaseModel):
    format: Literal["pose", "motion"] = Field(
        "pose", description="pose = 原姿态 JSON；motion = iOS App 动作 JSON"
    )
    granularity: Granularity = "full"
    include_raw_landmarks: bool = False
    include_root_motion: bool = True
    output_dir: str | None = Field(None, description="留空写 ~/.detect-body/exports")
    filename: str | None = None
    motion: MotionExportOptions | None = None


class CacheClearResponse(BaseModel):
    removed: int = Field(0, description="删除的断点文件数量")
    directory: str


class ExportResponse(BaseModel):
    artifact: ExportArtifact
    downloadUrl: str | None = None
    outputDir: str
    schemaVersion: str = SCHEMA_VERSION
    format: Literal["pose", "motion"] = "pose"
    depthConfidence: float = 0.0
    warnings: list[str] = Field(
        default_factory=list, description="深度可信度提示；非空时 UI 应在导出成功提示里带一句"
    )


def _manager(request: Request) -> AnalysisJobManager:
    return request.app.state.analysis_jobs


def _settings(request: Request) -> Settings:
    return request.app.state.settings


@router.post("", response_model=JobStatus, status_code=201)
def start_analysis(request: Request, payload: StartAnalysisRequest) -> JobStatus:
    job = _manager(request).start(payload.video_id, payload.options)
    return job.status()


@router.get("", response_model=list[JobStatus])
def list_analysis(request: Request) -> list[JobStatus]:
    return _manager(request).list()


@router.get("/{job_id}", response_model=JobStatus)
def get_analysis(request: Request, job_id: str) -> JobStatus:
    return _manager(request).get(job_id).status()


@router.post("/{job_id}/cancel", response_model=JobStatus)
def cancel_analysis(request: Request, job_id: str) -> JobStatus:
    return _manager(request).cancel(job_id).status()


@router.delete("/cache", response_model=CacheClearResponse)
def clear_analysis_cache(request: Request) -> CacheClearResponse:
    """清空断点续算缓存：下次识别从第一帧重新开始，而不是接着上次的结果。"""
    manager = _manager(request)
    active = manager.active
    if active:
        raise ValidationFailedError(
            "有分析任务正在运行，请先取消或等待完成再清理缓存",
            details={"activeJobs": active},
        )
    settings = _settings(request)
    removed = clear_checkpoints(settings.checkpoints_dir)
    logger.info("清理断点缓存：%s 个文件（%s）", removed, settings.checkpoints_dir)
    return CacheClearResponse(removed=removed, directory=str(settings.checkpoints_dir))


@router.get("/{job_id}/frames", response_model=FramesPayload)
def get_frames(
    request: Request,
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_FRAME_LIMIT, ge=1, le=MAX_FRAME_LIMIT),
) -> FramesPayload:
    job = _manager(request).get(job_id)
    result = job.result
    preview = job.preview if result is None else None
    # 推理还在跑时只有"到目前为止"的部分帧：也能读，用于边算边看骨架。
    live = job.live_frames if result is None and preview is None else []
    if result is None and preview is None and not live:
        raise ValidationFailedError(
            "推理尚未完成，暂时无法读取帧数据",
            details={"jobId": job_id, "phase": job.status().phase},
        )

    # 后处理（大视频的 IK 拟合可能十几分钟）期间先用推理原始帧预览；
    # 任务完成后自动切回经过平滑/拟合/质检的最终结果。
    live_entries: dict[int, LiveFrameJoints] = {}
    if result is not None:
        pose_frames = result.postprocess.frames
        angles = result.postprocess.angles
        skeleton_frames = result.skeleton.frames
        joint_names = result.skeleton.joint_names
        stride = int(result.sampling.stride)
    elif preview is not None:
        pose_frames = preview.frames
        angles = [{} for _ in preview.frames]
        skeleton_frames = preview.skeleton.frames
        joint_names = preview.skeleton.joint_names
        stride = max(1, job.options.sampling.every_n_frames)
    else:
        # 推理进行中：2D 关键点直接来自引擎，米制关节/角度由 LiveSkeletonMapper
        # 增量映射（未平滑、未拟合），这样 3D 预览与白膜 Mesh 卡片也能跟着播放头同步。
        pose_frames = live
        live_entries = {entry.index: entry for entry in job.live_preview}
        angles = [live_entries[pose.index].angles if pose.index in live_entries else {} for pose in live]
        skeleton_frames = []
        joint_names = job.live_joint_names
        stride = max(1, job.options.sampling.every_n_frames)

    ordered = sorted(
        zip(pose_frames, angles, strict=False),
        key=lambda pair: pair[0].index,
    )
    skeleton_by_index = {frame.index: frame for frame in skeleton_frames}
    window = ordered[offset : offset + limit]
    frames: list[PreviewFrame] = []
    for pose, angles in window:
        converted = skeleton_by_index.get(pose.index)
        entry = live_entries.get(pose.index)
        frames.append(
            PreviewFrame(
                index=pose.index,
                t=round(pose.timestamp_ms / 1000.0, 4),
                valid=pose.valid,
                interpolated=pose.interpolated,
                landmarks=[
                    PreviewPoint(
                        x=round(item.x, 5),
                        y=round(item.y, 5),
                        z=round(item.z, 5),
                        visibility=round(float(item.visibility), 4),
                    )
                    for item in pose.landmarks
                ],
                joints=(
                    _live_joints(entry, joint_names)
                    if entry is not None
                    else {
                        name: PreviewJoint(
                            x=round(point.x, 5),
                            y=round(point.y, 5),
                            z=round(point.z, 5),
                            visibility=round(float(point.visibility), 4),
                        )
                        for name, point in (converted.joints.items() if converted else [])
                    }
                ),
                angles={name: round(value, 2) for name, value in angles.items()},
                bbox=pose.bbox,
                hipHeight=(
                    round(converted.ground_span, 4)
                    if converted is not None
                    and converted.ground_span is not None
                    and converted.ground_span > 0
                    else None
                ),
                hipOffsetX=round(converted.root.x, 4) if converted is not None else 0.0,
                hipOffsetZ=round(converted.root.z, 4) if converted is not None else 0.0,
            )
        )

    return FramesPayload(
        jobId=job.id,
        fps=round(float(job.probe.fps or 30.0), 4),
        stride=stride,
        totalFrames=len(ordered),
        offset=offset,
        limit=limit,
        hasMore=offset + len(frames) < len(ordered),
        jointNames=joint_names,
        frames=frames,
    )


@router.post("/{job_id}/export", response_model=ExportResponse)
def export_analysis(request: Request, job_id: str, payload: ExportRequest) -> ExportResponse:
    job = _manager(request).get(job_id)
    result = job.result
    if result is None:
        raise ValidationFailedError(
            "任务尚未完成，无法导出", details={"jobId": job_id, "phase": job.status().phase}
        )

    settings = _settings(request)
    output_dir = Path(payload.output_dir).expanduser() if payload.output_dir else settings.exports_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise ValidationFailedError(
            "导出路径不是目录", details={"outputDir": str(output_dir)}
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source = source_from_probe(job.metadata.filename, job.probe)

    if payload.format == "motion":
        clip_config = MotionClipConfig(
            **(payload.motion.model_dump(exclude_none=True) if payload.motion else {})
        )
        document = document_from_analysis(
            result,
            source,
            ExportConfig(
                granularity="keypoints",
                include_raw_landmarks=False,
                include_root_motion=True,
            ),
        )
        clip = build_motion_clip(document, clip_config)
        filename = payload.filename or motion_filename(source.file)
        if not filename.lower().endswith(".json"):
            filename += ".json"
        artifact = write_motion_clip(clip, output_dir / filename)
        downloadable = output_dir.resolve() == settings.exports_dir.resolve()
        return ExportResponse(
            artifact=artifact,
            downloadUrl=f"/api/exports/{artifact.filename}" if downloadable else None,
            outputDir=str(output_dir),
            schemaVersion=str(MOTION_SCHEMA_VERSION),
            format="motion",
            depthConfidence=document.depthConfidence,
            warnings=list(document.warnings),
        )

    document = document_from_analysis(
        result,
        source,
        ExportConfig(
            granularity=payload.granularity,
            include_raw_landmarks=payload.include_raw_landmarks,
            include_root_motion=payload.include_root_motion,
        ),
    )
    artifact = export_document(
        document,
        output_dir,
        filename=payload.filename,
        mode=payload.granularity,
    )

    downloadable = output_dir.resolve() == settings.exports_dir.resolve()
    return ExportResponse(
        artifact=artifact,
        downloadUrl=f"/api/exports/{artifact.filename}" if downloadable else None,
        outputDir=str(output_dir),
        format="pose",
        depthConfidence=document.depthConfidence,
        warnings=list(document.warnings),
    )


__all__ = [
    "CacheClearResponse",
    "ExportRequest",
    "ExportResponse",
    "FramesPayload",
    "MotionExportOptions",
    "PreviewFrame",
    "StartAnalysisRequest",
    "router",
]
