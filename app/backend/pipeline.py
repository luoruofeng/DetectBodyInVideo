"""分析流水线装配（任务 4 / 5 / 6 的串联层）。

视频 → 采样 → 引擎逐帧推理 → 跟踪 → 后处理 → 骨骼契约。
这一层只做编排：不实现算法、不碰 JSON 导出（任务 7）、不做 HTTP（任务 8）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.backend.core.errors import AnalysisCancelledError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings, get_settings
from app.backend.engine.acceleration import AccelerationPlan, build_hardware_report
from app.backend.engine.base import EngineSelection, PoseEngine, PoseFrame
from app.backend.engine.enhancements import (
    MediaPipeRefiner,
    RefinementConfig,
    RefinementResult,
    apply_refinements,
)
from app.backend.engine.registry import select_engine
from app.backend.media.probe import VideoProbe, probe_video
from app.backend.media.sampling import AnalysisPlan, SamplingConfig, build_analysis_plan
from app.backend.perf.config import RuntimeConfig
from app.backend.perf.runner import RuntimeProgress, run_inference
from app.backend.postprocess.ik_fit import FitConfig
from app.backend.postprocess.pipeline import (
    PostProcessConfig,
    PostProcessResult,
    run_postprocess,
)
from app.backend.skeleton.bundle import SkeletonBundle, SkeletonConfig, build_skeleton_bundle

logger = get_logger("pipeline")


class AnalysisOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    postprocess: PostProcessConfig = Field(default_factory=PostProcessConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    perf: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="生产/消费、批处理、断点续算（任务 9）"
    )
    ik_fit: FitConfig | None = Field(
        None,
        alias="ikFit",
        description="骨长约束拟合的快捷覆盖（HTTP/前端字段名 ikFit）；空则用 postprocess.ik_fit",
    )
    engine: Literal["auto", "mediapipe", "onnx"] = "auto"
    refinement: RefinementConfig = Field(
        default_factory=RefinementConfig,
        description="手部 21 点 / 面部语义点增强；缺模型或缺依赖时自动降级",
    )
    max_frames: int | None = Field(None, ge=1, description="调试用：最多分析多少帧")


class AnalysisProgress(BaseModel):
    frames_done: int = 0
    frames_total: int = 0
    processed_fps: float = 0.0
    eta_seconds: float | None = None
    message: str = ""
    # 推理结束后 frames_done 已满，但后处理（尤其 IK 拟合）还要跑很久。
    # UI 在 stage != "inference" 时按 stage_done/stage_total 显示真实阶段进度，
    # 避免把"后处理 100%"误当成已完成。
    stage: str = "inference"
    stage_done: int = 0
    stage_total: int = 0


class AnalysisResult(BaseModel):
    selection: EngineSelection
    sampling: AnalysisPlan
    postprocess: PostProcessResult
    skeleton: SkeletonBundle
    elapsed_seconds: float = 0.0
    processed_fps: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @property
    def frames(self) -> list[PoseFrame]:
        return self.postprocess.frames


ProgressCallback = Callable[[AnalysisProgress], None]
CancelCallback = Callable[[], bool]
# 推理一结束就把原始帧 + 骨骼映射交给调用方，用于后处理期间的可视化预览。
PreviewCallback = Callable[[list[PoseFrame], SkeletonBundle], None]
# 推理进行中反复回传"到目前为止的帧"，用于边算边看骨架（长视频不必干等）。
PartialFramesCallback = Callable[[list[PoseFrame]], None]


def _default_plan(settings: Settings) -> AccelerationPlan:
    return build_hardware_report(settings).acceleration


def analyze_video(
    path: Path | str,
    *,
    probe: VideoProbe | None = None,
    options: AnalysisOptions | None = None,
    plan: AccelerationPlan | None = None,
    settings: Settings | None = None,
    engine: PoseEngine | None = None,
    selection: EngineSelection | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    on_inference_done: PreviewCallback | None = None,
    on_partial_frames: PartialFramesCallback | None = None,
) -> AnalysisResult:
    """跑完整条链路；``engine`` 可注入（单测/复用已加载模型）。"""
    settings = settings or get_settings()
    options = options or AnalysisOptions()
    video_path = Path(path)
    probe = probe or probe_video(
        video_path,
        allowed_codecs=settings.allowed_codecs,
        rejected_codecs=settings.reject_codecs,
    )
    sampling_plan = build_analysis_plan(video_path, probe, options.sampling)

    owns_engine = engine is None
    if engine is None:
        plan = plan or _default_plan(settings)
        engine, selection = select_engine(
            plan, settings=settings, preferred=options.engine, runtime=options.perf
        )
    else:
        engine.apply_runtime(options.perf)
        if not engine.prepared:
            engine.prepare()
        if selection is None:
            selection = EngineSelection(
                engine=engine.name,
                display_name=engine.display_name,
                requested="injected",
                reasons=engine.notes,
                capabilities=engine.capabilities,
                model_tier=engine.capabilities.model_tier,
            )

    assert selection is not None
    frames: list[PoseFrame] = []
    notes: list[str] = list(sampling_plan.warnings)
    started = time.perf_counter()
    processed = 0

    # 手/脸增强：整帧级检测，逐帧挂到该帧第一个人。缺模型/缺依赖时静默降级。
    refiner: MediaPipeRefiner | None = None
    refine_notes: list[str] = []
    if options.refinement.hands or options.refinement.face:
        available, reason = MediaPipeRefiner.availability(options.refinement, settings)
        if not available:
            refine_notes.append(f"手/脸增强已跳过：{reason}")
        else:
            refiner = MediaPipeRefiner(options.refinement, settings=settings)
            try:
                refiner.prepare()
            except Exception as exc:  # noqa: BLE001 - 增强初始化失败不影响主链路
                refine_notes.append(f"手/脸增强已跳过：{exc}")
                refiner = None
    notes.extend(refine_notes)

    def refine(image: np.ndarray, detected: list[PoseFrame]) -> None:
        if refiner is None:
            return
        result: RefinementResult = refiner.infer(image)
        apply_refinements(detected, result)

    def forward(info: RuntimeProgress) -> None:
        if progress is not None:
            progress(AnalysisProgress(**info.model_dump()))

    try:
        outcome = run_inference(
            engine,
            video_path,
            probe,
            sampling_plan,
            config=options.perf,
            settings=settings,
            progress=forward,
            should_cancel=should_cancel,
            max_frames=options.max_frames,
            checkpoint_dir=settings.checkpoints_dir,
            on_frames=on_partial_frames,
            refine_frame=refine if refiner is not None else None,
        )
        frames = outcome.frames
        processed = outcome.processed
        notes.extend(outcome.notes)
        if refiner is not None:
            notes.extend(refiner.notes)
    finally:
        if refiner is not None:
            refiner.close()
        if owns_engine:
            engine.close()

    truncated = (
        options.max_frames is not None
        and (outcome.resumed_frames + processed) >= options.max_frames
    )
    if truncated:
        notes.append(f"达到 max_frames={options.max_frames}，仅分析前 {processed} 帧")

    display_aspect = (
        (probe.display_width / probe.display_height)
        if probe.display_height and probe.display_width
        else 9 / 16
    )
    skeleton_config = options.skeleton

    # 推理结束后立刻给出可预览的原始骨架：22623 帧的 IK 拟合可能要十几分钟，
    # 不能让用户在整个后处理期间面对空画面。
    if on_inference_done is not None:
        try:
            on_inference_done(frames, build_skeleton_bundle(frames, skeleton_config))
        except Exception as exc:  # noqa: BLE001 - 预览失败不应中断分析
            logger.warning("生成推理预览失败：%s", exc)

    def forward_post(stage: str, message: str, done: int, total: int) -> None:
        if progress is None:
            return
        progress(
            AnalysisProgress(
                frames_done=outcome.resumed_frames + processed,
                frames_total=sampling_plan.selected_frames,
                processed_fps=outcome.processed_fps,
                eta_seconds=None,
                message=message,
                stage=stage,
                stage_done=done,
                stage_total=total,
            )
        )

    if progress is not None:
        forward_post("postprocess", "后处理中（原始骨架已可预览）", 0, 1)

    post_updates: dict[str, object] = {
        "stride": max(1, sampling_plan.stride),
        "fps": probe.fps or 30.0,
        "aspect_ratio": display_aspect,
    }
    ik_config = options.ik_fit
    if ik_config is None:
        # 允许用配置 / 环境变量统一调整默认值（DETECT_BODY_IK_FIT_ENABLED 等）
        ik_config = options.postprocess.ik_fit.model_copy(
            update={
                "enabled": options.postprocess.ik_fit.enabled and settings.ik_fit_enabled,
                "max_nfev": settings.ik_fit_max_nfev
                if options.postprocess.ik_fit.max_nfev == FitConfig().max_nfev
                else options.postprocess.ik_fit.max_nfev,
            }
        )
    post_updates["ik_fit"] = ik_config
    post_config = options.postprocess.model_copy(update=post_updates)
    postprocess = run_postprocess(
        frames, post_config, progress=forward_post, should_cancel=should_cancel
    )

    skeleton = build_skeleton_bundle(postprocess.frames, skeleton_config)

    elapsed = time.perf_counter() - started
    result = AnalysisResult(
        selection=selection,
        sampling=sampling_plan,
        postprocess=postprocess,
        skeleton=skeleton,
        elapsed_seconds=round(elapsed, 3),
        processed_fps=round(processed / elapsed, 2) if elapsed > 0 else 0.0,
        notes=notes + skeleton.notes,
    )
    logger.info(
        "分析完成：%s 帧，%.1f fps，结论 %s",
        processed,
        result.processed_fps,
        postprocess.quality.verdict,
    )
    if progress is not None:
        progress(
            AnalysisProgress(
                frames_done=processed,
                frames_total=processed,
                processed_fps=result.processed_fps,
                eta_seconds=0.0,
                message="完成",
                stage="done",
                stage_done=processed,
                stage_total=processed,
            )
        )
    return result


def summarize(result: AnalysisResult) -> dict[str, object]:
    """给 HTTP 层用的轻量摘要（不含逐帧数据）。"""
    quality = result.postprocess.quality
    return {
        "engine": result.selection.engine,
        "engineDisplayName": result.selection.display_name,
        "degraded": result.selection.degraded,
        "reasons": result.selection.reasons,
        "frames": len(result.postprocess.frames),
        "processedFps": result.processed_fps,
        "elapsedSeconds": result.elapsed_seconds,
        "quality": quality.model_dump(mode="json"),
        "fit": {
            key: value
            for key, value in result.postprocess.fit.model_dump(mode="json").items()
            if key != "per_frame"
        },
        "hands": {
            "applied": result.postprocess.hands.applied,
            "framesWithHands": result.postprocess.hands.frames_with_hands,
            "sides": [
                item.model_dump(mode="json") for item in result.postprocess.hands.sides
            ],
            "notes": result.postprocess.hands.notes,
        },
        "tracks": [item.model_dump(mode="json") for item in result.postprocess.tracks],
        "primaryTrackId": result.postprocess.primary_track_id,
        "targetTrackId": result.postprocess.target_track_id,
        "notes": result.notes,
    }


__all__ = [
    "AnalysisCancelledError",
    "AnalysisOptions",
    "AnalysisProgress",
    "AnalysisResult",
    "PartialFramesCallback",
    "analyze_video",
    "summarize",
]
