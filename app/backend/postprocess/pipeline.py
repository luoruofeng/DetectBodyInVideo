"""后处理流水线（任务 5.1~5.4 串起来）。

顺序（改动后）：
选目标 track → 补连续帧 → 标记无效帧 → 补短缺口 → 平滑 → **手部时序稳定** →
骨长标定 → yaw 估计 → **深度符号先验** → 固定骨长 IK 拟合 → 算角度 → 质检。

三个关键约定：
- 平滑只作用在有效帧上，避免空占位帧的 0 值把轨迹拉回原点；
- **IK 拟合放在平滑之后**：只有最后一步输出关节位置，才能保证骨长严格恒定、
  且与 2D 投影一致；放在平滑之前会被平滑再次破坏骨长；
- **手部稳定放在身体平滑之后**：补缺口/沿用旧手型时要把腕点贴到"已经平滑过"的
  人体腕点上，否则手会跟着身体抖动一下再被甩回来。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from app.backend.core.errors import AnalysisCancelledError
from app.backend.core.logging import get_logger
from app.backend.engine.base import PoseFrame
from app.backend.engine.tracking import (
    TrackSummary,
    filter_track,
    primary_track_id,
    summarize_tracks,
)
from app.backend.postprocess.angles import AngleConfig, attach_angles
from app.backend.postprocess.depth_sign import (
    DepthSignConfig,
    DepthSignReport,
    estimate_depth_signs,
)
from app.backend.postprocess.gaps import GapFillConfig, GapFillReport, densify, process_gaps
from app.backend.postprocess.hands import (
    HandStabilityConfig,
    HandStabilityReport,
    stabilize_hands,
)
from app.backend.postprocess.ik_fit import (
    FitConfig,
    FitReport,
    calibrate_bone_lengths,
    fit_frames,
)
from app.backend.postprocess.quality import QualityConfig, QualityReport, build_quality_report
from app.backend.postprocess.smoothing import (
    SmoothingConfig,
    SmoothingInfo,
    smooth_frames,
    smooth_world_pass,
)
from app.backend.skeleton.orientation import (
    YawConfig,
    YawEstimate,
    as_prior,
    estimate_yaws,
)

logger = get_logger("postprocess.pipeline")

# 阶段名、中文说明、已完成数、总数。长视频的 IK 拟合会密集上报，其余阶段只在结束时上报。
PostProgressCallback = Callable[[str, str, int, int], None]
PostCancelCallback = Callable[[], bool]


class PostProcessConfig(BaseModel):
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    gaps: GapFillConfig = Field(default_factory=GapFillConfig)
    hand_stability: HandStabilityConfig = Field(
        default_factory=HandStabilityConfig,
        description="手部 21 点的掉帧补全 / 姿态迟滞 / 平滑",
    )
    angles: AngleConfig = Field(default_factory=AngleConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    ik_fit: FitConfig = Field(default_factory=FitConfig)
    depth_sign: DepthSignConfig = Field(
        default_factory=DepthSignConfig,
        description="深度方向（每骨段 1 个比特）先验；默认关闭，开启后由 ik_fit 的 hinge 项执行",
    )
    yaw: YawConfig = Field(default_factory=YawConfig)
    estimate_orientation: bool = Field(
        True, description="用肩宽/髋宽比估计身体 yaw（作为拟合先验，也进 PostProcessResult）"
    )
    compute_angles: bool = True
    target_track_id: int | None = Field(None, description="跟随的目标人物；空则自动选最长 track")
    stride: int = Field(1, ge=1, description="采样步长，用于补齐连续帧")
    fps: float = Field(30.0, gt=0)
    aspect_ratio: float | None = Field(
        None, gt=0, description="画面宽高比（width/height）；IK 拟合与 yaw 估计需要它"
    )


def legacy_postprocess_config() -> PostProcessConfig:
    """**回滚入口**：把"深度精度改造"引入的默认值全部还原成改造前的行为。

    DoD 要求"关闭所有新开关时输出与改动前一致"。逐个开关还原：

    - ``smoothing.one_euro``：`*_z` 全留 ``None`` → 三轴共用 1.0/0.5；
    - ``smoothing.extra_world_pass=False``：不再单独平滑世界点；
    - ``ik_fit.depth_prior_weight=0.05`` / ``temporal_axes="xyz"``：拟合的旧残差权重；
    - ``ik_fit.bone_sample_min_projected_ratio=0`` / ``symmetrize_bones=False``：
      骨长回到"所有帧等权中位数、左右独立"，不再筛正对帧、不强制对称；
    - ``estimate_orientation=False``：不算 yaw、也不给拟合 yaw 先验；
    - ``hand_stability.enabled=False``：手部 21 点原样透传，不做补帧/迟滞/平滑；
    - ``ik_fit.hold_occluded_limbs=False`` / ``angles.hold_occluded_frames=0``：
      关闭遮挡肢体的相对角度保持和角度读数保持；
    - ``quality.enable_depth_gate=False`` / ``enable_framing_rules=False``：质检不因新指标降级。

    新增的字段（depthConfidence 等）仍会输出，但都是附加信息，不改变可用数值。
    """
    from app.backend.postprocess.one_euro import OneEuroParams

    return PostProcessConfig(
        smoothing=SmoothingConfig(
            one_euro=OneEuroParams(min_cutoff=1.0, beta=0.5, d_cutoff=1.0),
            extra_world_pass=False,
        ),
        hand_stability=HandStabilityConfig(enabled=False),
        angles=AngleConfig(hold_occluded_frames=0),
        ik_fit=FitConfig(
            depth_prior_weight=0.05,
            temporal_axes="xyz",
            bone_sample_min_projected_ratio=0.0,
            symmetrize_bones=False,
            hold_occluded_limbs=False,
            depth_sign_weight=0.0,
        ),
        depth_sign=DepthSignConfig(enabled=False),
        estimate_orientation=False,
        quality=QualityConfig(enable_depth_gate=False, enable_framing_rules=False),
    )


class PostProcessResult(BaseModel):
    frames: list[PoseFrame] = Field(default_factory=list)
    angles: list[dict[str, float]] = Field(default_factory=list)
    quality: QualityReport = Field(default_factory=QualityReport)
    smoothing: SmoothingInfo = Field(default_factory=SmoothingInfo)
    gaps: GapFillReport = Field(default_factory=GapFillReport)
    hands: HandStabilityReport = Field(default_factory=HandStabilityReport)
    fit: FitReport = Field(default_factory=FitReport)
    depth_sign: DepthSignReport = Field(default_factory=DepthSignReport)
    yaw: list[YawEstimate] = Field(default_factory=list)
    tracks: list[TrackSummary] = Field(default_factory=list)
    primary_track_id: int | None = None
    target_track_id: int | None = None


def run_postprocess(
    frames: Sequence[PoseFrame],
    config: PostProcessConfig | None = None,
    *,
    progress: PostProgressCallback | None = None,
    should_cancel: PostCancelCallback | None = None,
) -> PostProcessResult:
    """执行后处理；``progress``/``should_cancel`` 让长视频的阶段进度与取消可被 UI 感知。"""
    config = config or PostProcessConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    result = PostProcessResult(frames=[], angles=[], quality=QualityReport())

    def emit(stage: str, message: str, done: int, total: int) -> None:
        if should_cancel is not None and should_cancel():
            raise AnalysisCancelledError("后处理已取消")
        if progress is not None:
            progress(stage, message, max(0, done), max(1, total))

    if not ordered:
        emit("quality", "生成质检报告", 1, 1)
        result.quality = build_quality_report([], config.quality)
        return result

    has_tracks = any(frame.track_id is not None for frame in ordered)
    if has_tracks:
        emit("track", "筛选目标人物", 0, 1)
        result.tracks = summarize_tracks(ordered)
        result.primary_track_id = primary_track_id(ordered)
        target = config.target_track_id or result.primary_track_id
        if target is not None and any(frame.track_id == target for frame in ordered):
            ordered = filter_track(ordered, target)
            result.target_track_id = target
        emit("track", "筛选目标人物", 1, 1)

    total = max(1, len(ordered))
    emit("densify", "补齐连续帧", 0, total)
    dense = densify(ordered, stride=config.stride, fps=config.fps)
    emit("densify", "补齐连续帧", total, total)
    emit("gaps", "处理缺失帧", 0, total)
    with_gaps, gap_report = process_gaps(dense, config.gaps)
    emit("gaps", "处理缺失帧", total, total)

    smoothable = [frame for frame in with_gaps if frame.valid and frame.landmarks]
    emit("smooth", "平滑去抖", 0, max(1, len(smoothable)))
    smoothed, smoothing_info = smooth_frames(smoothable, config.smoothing)
    # 世界点是米制、归一化点是无量纲，噪声与速度量级差两个数量级，
    # 所以再单独平滑一趟（用 config.smoothing.world_one_euro，不能与上面共用参数）。
    if config.smoothing.extra_world_pass and smoothing_info.applied:
        world_frames, world_info = smooth_world_pass(smoothed, config.smoothing)
        if world_info.applied:
            world_by_index = {frame.index: frame for frame in world_frames}
            smoothed = [world_by_index.get(frame.index, frame) for frame in smoothed]
            for key, value in world_info.params.items():
                smoothing_info.params[f"world{key[0].upper()}{key[1:]}"] = value
            smoothing_info.notes.append(
                f"world_landmarks 单独平滑 {world_info.frames} 帧（米制参数，z 更弱）"
            )
    by_index = {frame.index: frame for frame in smoothed}
    final = [by_index.get(frame.index, frame) for frame in with_gaps]
    emit("smooth", "平滑去抖", max(1, len(smoothable)), max(1, len(smoothable)))

    # 0) 手部 21 点单独走时序稳定：掉帧补全 + 姿态迟滞 + 相对姿态平滑。
    #    必须放在身体平滑之后：补出来的手要贴到"平滑过"的腕点上，才不会跟着手抖。
    emit("hands", "稳定手部关节", 0, max(1, len(final)))
    hand_report = stabilize_hands(final, config.hand_stability)
    emit("hands", "稳定手部关节", len(final), max(1, len(final)))

    # 1) 骨长标定一次，yaw / 深度符号 / IK 三处共用同一个结果（避免各自重复标定）。
    fit_config = config.ik_fit
    if config.aspect_ratio and not fit_config.aspect_ratio:
        fit_config = fit_config.model_copy(update={"aspect_ratio": config.aspect_ratio})
    aspect = fit_config.aspect_ratio or 9 / 16.0
    lengths = calibrate_bone_lengths(final, fit_config) if fit_config.enabled else None

    # 2) 身体朝向：用肩线/髋线的投影宽度算 yaw（比读 world z 可靠一个数量级）
    yaw_estimates: list[YawEstimate] = []
    if config.estimate_orientation and fit_config.enabled and lengths is not None:
        emit("yaw", "估计身体朝向", 0, max(1, len(final)))
        yaw_estimates = estimate_yaws(final, lengths, config.yaw, aspect=aspect)
        emit("yaw", "估计身体朝向", len(final), max(1, len(final)))

    # 3) 深度符号：单目下每根骨段只剩"Δz 取正还是取负"一个比特，逐帧由引擎 z 决定
    #    就会来回翻（前踢腿被解成后踢腿）。这一步只产符号，不产坐标、不改幅度。
    sign_map: dict[int, dict[str, tuple[int, float]]] = {}
    if config.depth_sign.enabled and fit_config.enabled and lengths is not None:
        emit("lift", "修正前后方向", 0, max(1, len(final)))
        sign_map, depth_sign_report = estimate_depth_signs(
            final,
            config.depth_sign,
            lengths=lengths,
            aspect=aspect,
            progress=lambda done, total_segments: emit(
                "lift", "修正前后方向", done, total_segments
            ),
            should_cancel=should_cancel,
        )
        emit("lift", "修正前后方向", max(1, len(final)), max(1, len(final)))
    else:
        depth_sign_report = DepthSignReport(
            mode="off", frames=len(final), notes=["深度符号先验未启用"]
        )

    # 4) 固定骨长 + 2D 重投影拟合：只动 world_landmarks，无效帧原样透传。
    #    yaw 与深度符号都作为先验进残差：一个压住"肩线朝向"，一个只选"深度分支"。
    if fit_config.enabled:
        emit("ik_fit", "骨长约束拟合", 0, max(1, len(final)))
        final, fit_report = fit_frames(
            final,
            fit_config,
            lengths=lengths,
            yaws=as_prior(yaw_estimates) if yaw_estimates else None,
            signs=sign_map or None,
            progress=lambda done, total_steps: emit(
                "ik_fit", "骨长约束拟合", done, total_steps
            ),
            should_cancel=should_cancel,
        )
    else:
        fit_report = FitReport(applied=False, frames=len(final), notes=["ik_fit 已关闭"])

    emit("angles", "计算关节角度", 0, max(1, len(final)))
    angles = (
        attach_angles(final, config.angles)
        if config.compute_angles
        else [{} for _ in final]
    )
    emit("angles", "计算关节角度", len(final), max(1, len(final)))
    emit("quality", "生成质检报告", 0, max(1, len(final)))
    quality = build_quality_report(
        final,
        config.quality,
        smoothing=smoothing_info,
        gaps=gap_report,
        fit=fit_report,
        depth_sign=depth_sign_report,
        aspect=aspect,
    )
    emit("quality", "生成质检报告", len(final), max(1, len(final)))

    logger.debug(
        "后处理完成：%s 帧，有效 %.1f%%，结论 %s",
        len(final),
        quality.valid_frame_ratio * 100,
        quality.verdict,
    )
    return PostProcessResult(
        frames=final,
        angles=angles,
        quality=quality,
        smoothing=smoothing_info,
        gaps=gap_report,
        hands=hand_report,
        fit=fit_report,
        depth_sign=depth_sign_report,
        yaw=yaw_estimates,
        tracks=result.tracks,
        primary_track_id=result.primary_track_id,
        target_track_id=result.target_track_id,
    )
