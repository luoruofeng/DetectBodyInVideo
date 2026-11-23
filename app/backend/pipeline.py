"""分析流水线装配（任务 4 / 5 / 6 的串联层）。

视频 → 采样 → 引擎逐帧推理 → 跟踪 → 后处理 → 骨骼契约。
这一层只做编排：不实现算法、不碰 JSON 导出（任务 7）、不做 HTTP（任务 8）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
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
from app.backend.postprocess.depth_sign import DepthSignConfig
from app.backend.postprocess.ik_fit import FitConfig
from app.backend.postprocess.pipeline import (
    PostProcessConfig,
    PostProcessResult,
    run_postprocess,
)
from app.backend.skeleton.bundle import SkeletonBundle, SkeletonConfig, build_skeleton_bundle

logger = get_logger("pipeline")


class EngineOptions(BaseModel):
    """引擎侧的检测阈值与重跑策略（§5.1）。

    MediaPipe 的 ``min_detection_confidence`` / ``min_tracking_confidence`` 是"检测到的
    姿态本身算不算数"的闸门；下肢在画面里只占一小块、又常被遮挡时，默认 0.5 会把腿判成
    "没看见"，于是 z 只能靠躯干外推 —— 这正是前踢腿幅度失真的直接原因之一。
    """

    min_detection_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="留空用服务级配置（DETECT_BODY_MIN_DETECTION_CONFIDENCE）"
    )
    min_tracking_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="留空用服务级配置（DETECT_BODY_MIN_TRACKING_CONFIDENCE）"
    )
    lower_body_refine: bool | None = Field(
        None,
        description="MediaPipe 专用：下肢可见度低时对下半身裁剪再推理一次（ROI 重解，成本约 +1 次/坏帧）",
    )
    model_tier: Literal["lite", "full", "heavy"] | None = Field(
        None, description="条件式重跑时使用的档位；留空用 heavy"
    )
    retry_with_heavy: bool | None = Field(
        None, description="下肢可见度不足时是否重跑整段；留空跟随服务级配置"
    )


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
    engine_options: EngineOptions = Field(
        default_factory=EngineOptions,
        alias="engineOptions",
        description="检测阈值 / 下肢裁剪重解 / 条件式重跑档位",
    )
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
    # §5.1 条件式重跑：让 UI 能说明"这次为什么跑了两遍"
    retried_with_heavy: bool = False
    retry_reason: str = ""

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


# 髋 / 膝 / 踝 / 脚跟 / 脚尖（23~32）：腿部质量的判据只看这一段
LOWER_LIMB_INDICES: tuple[int, ...] = (23, 24, 25, 26, 27, 28, 29, 30, 31, 32)


def lower_limb_visibility(frames: Sequence[PoseFrame]) -> float:
    """下肢关键点的平均可见度（0~1）；没有有效帧返回 0。"""
    values: list[float] = []
    for frame in frames:
        if not frame.valid or len(frame.landmarks) < 33:
            continue
        for index in LOWER_LIMB_INDICES:
            values.append(float(frame.landmarks[index].visibility))
    return float(np.mean(values)) if values else 0.0


def lower_limb_unreliable_ratio(frames: Sequence[PoseFrame], threshold: float) -> float:
    """下肢平均可见度低于阈值的**帧**占比；这是"该不该重跑"的判据。"""
    counted = 0
    bad = 0
    for frame in frames:
        if not frame.valid or len(frame.landmarks) < 33:
            continue
        counted += 1
        visibility = float(
            np.mean([frame.landmarks[index].visibility for index in LOWER_LIMB_INDICES])
        )
        if visibility < threshold:
            bad += 1
    return bad / counted if counted else 0.0


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
    """跑完整条链路；``engine`` 可注入（单测/复用已加载模型）。

    这里只做一件事以外的编排：**条件式重跑**（§5.1）。下肢可见度不足时，用更高的
    模型档位重跑一遍整段（"重跑下肢区间"需要局部重解，当前 ROI 重解在 MediaPipe
    引擎内部由 ``lower_body_refine`` 完成，不再二次读取视频）。
    """
    settings = settings or get_settings()
    options = options or AnalysisOptions()
    result = _analyze_once(
        path,
        probe=probe,
        options=options,
        plan=plan,
        settings=settings,
        engine=engine,
        selection=selection,
        progress=progress,
        should_cancel=should_cancel,
        on_inference_done=on_inference_done,
        on_partial_frames=on_partial_frames,
    )
    if engine is not None:
        return result  # 注入引擎的调用方自己负责生命周期，不做二次推理
    retry_settings = _retry_settings_for_quality(result, options, settings)
    if retry_settings is None:
        return result

    logger.info("下肢可见度不足，按重档模型重跑：%s", retry_settings.prefer_model_tier)
    if progress is not None:
        progress(
            AnalysisProgress(
                frames_done=0,
                frames_total=result.sampling.total_frames,
                message="下肢关键点置信度偏低，正在用高精度模型重跑",
                stage="inference",
            )
        )
    retry = _analyze_once(
        path,
        probe=probe,
        options=options,
        plan=None,
        settings=retry_settings,
        engine=None,
        selection=None,
        progress=progress,
        should_cancel=should_cancel,
        on_inference_done=on_inference_done,
        on_partial_frames=on_partial_frames,
    )
    retry.retried_with_heavy = True
    retry.retry_reason = (
        f"下肢平均可见度 {lower_limb_visibility(result.postprocess.frames):.2f} 偏低，"
        f"已用 {retry.selection.model_tier} 档模型重跑整段"
    )
    retry.selection.reasons = [*retry.selection.reasons, retry.retry_reason]
    return retry


def _engine_settings(settings: Settings, overrides: EngineOptions) -> Settings:
    """把"本次任务的引擎参数"落到一份 settings 副本上（引擎构造只认 settings）。"""
    update: dict[str, object] = {}
    if overrides.min_detection_confidence is not None:
        update["min_detection_confidence"] = min(
            0.999, max(0.001, overrides.min_detection_confidence)
        )
    if overrides.min_tracking_confidence is not None:
        update["min_tracking_confidence"] = min(
            0.999, max(0.001, overrides.min_tracking_confidence)
        )
    if overrides.lower_body_refine is not None:
        update["lower_body_refine"] = overrides.lower_body_refine
    if overrides.model_tier is not None:
        update["prefer_model_tier"] = overrides.model_tier
    return settings.model_copy(update=update) if update else settings


def _retry_settings_for_quality(
    result: AnalysisResult, options: AnalysisOptions, settings: Settings
) -> Settings | None:
    """判定是否需要提高模型档位重跑；返回 None 表示不需要。"""
    if not settings.lower_limb_retry:
        return None
    if options.engine_options.retry_with_heavy is False:
        return None
    tier = result.selection.model_tier or result.selection.capabilities.model_tier
    target = options.engine_options.model_tier or "heavy"
    if tier == target:
        return None
    frames = result.postprocess.frames
    visibility = lower_limb_visibility(frames)
    ratio = lower_limb_unreliable_ratio(frames, settings.lower_limb_retry_visibility)
    if visibility >= settings.lower_limb_retry_visibility and ratio < settings.lower_limb_retry_ratio:
        return None
    logger.info(
        "条件式重跑触发：下肢可见度 %.2f，低于阈值的帧占 %.0f%%", visibility, ratio * 100
    )
    return settings.model_copy(update={"prefer_model_tier": target})


def _analyze_once(
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
            plan,
            settings=_engine_settings(settings, options.engine_options),
            preferred=options.engine,
            runtime=options.perf,
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
        # 传入本帧姿态：手部按腕点/肘点裁 ROI 再检测（远景全身视频整帧检不到手）。
        result: RefinementResult = refiner.infer(image, detected)
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
            notes.extend(refiner.stats_note())
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
    # 深度符号先验：请求体显式开启优先；服务级配置只能"打开默认值"，不能覆盖显式参数。
    depth_sign_config = options.postprocess.depth_sign
    if settings.depth_sign_enabled and not depth_sign_config.enabled:
        depth_sign_config = depth_sign_config.model_copy(update={"enabled": True})
    if depth_sign_config.mode == DepthSignConfig().mode:
        depth_sign_config = depth_sign_config.model_copy(update={"mode": settings.depth_sign_mode})
    post_updates["depth_sign"] = depth_sign_config
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
        "depthSign": {
            "applied": result.postprocess.depth_sign.applied,
            "mode": result.postprocess.depth_sign.mode,
            "segments": result.postprocess.depth_sign.segments,
            "flipsBefore": result.postprocess.depth_sign.flips_before,
            "flipsAfter": result.postprocess.depth_sign.flips_after,
            "disagreements": result.postprocess.depth_sign.disagreements,
            # §8.3：后端/自校准/方向向量计数都要能直接读到，不用去解析 notes 里的中文
            "lifter": result.postprocess.depth_sign.lifter,
            "autoCalibrateRatio": result.postprocess.depth_sign.auto_calibrate_ratio,
            "calibration": result.postprocess.depth_sign.calibration,
            "directionSegments": result.postprocess.depth_sign.direction_segments,
            "swingLeg": result.postprocess.depth_sign.swing_leg,
            "notes": result.postprocess.depth_sign.notes,
        },
        # §8.2：摆动腿与相位的结论单独一块，UI 不需要再从 notes 里猜
        "kick": {
            "swingLeg": result.postprocess.swing.swing_leg,
            "swingConfidence": result.postprocess.swing.confidence,
            "swingApplied": result.postprocess.swing.applied,
            "energy": result.postprocess.swing.energy,
            "detected": result.postprocess.kick_phases.detected,
            "side": result.postprocess.kick_phases.side,
            "phases": result.postprocess.kick_phases.phases,
            "notes": [*result.postprocess.swing.notes, *result.postprocess.kick_phases.notes],
        },
        "hands": {
            "applied": result.postprocess.hands.applied,
            "framesWithHands": result.postprocess.hands.frames_with_hands,
            "sides": [
                item.model_dump(mode="json") for item in result.postprocess.hands.sides
            ],
            "notes": result.postprocess.hands.notes,
        },
        # 直立基准：门控不满足时会静默跳过 —— 必须能读到。头相对躯干的前探不在这里，
        # 它的深度由「头骨段解剖长度 + 鼻尖反推」决定（ik_fit.head_depth / mapping）。
        "upright": {
            "applied": result.postprocess.upright_baseline.applied,
            "baselineMeters": result.postprocess.upright_baseline.baseline_meters,
            "eligibleFrames": result.postprocess.upright_baseline.eligible_frames,
            "adjustedFrames": result.postprocess.upright_baseline.adjusted_frames,
            "notes": result.postprocess.upright_baseline.notes,
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
