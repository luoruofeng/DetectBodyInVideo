"""数据质检（任务 5.4 / 深度可信度任务 5.3）。

输出一个可随 JSON 导出的 ``QualityReport``：
- 有效帧比例、平均置信度（visibility / score）；
- 最大单帧跳变（判断是否有穿模级抖动）；
- **深度专项**：z 抖动、骨长变异系数、平均重投影误差、``depth_confidence``；
- 结论 ``ready`` / ``review`` / ``rerecord``，供 UI 直接显示"可直接使用 / 建议重录"。

深度专项的意义：单目视频的 z 是估计值，绝对深度不可信。上面四项指标合起来回答一个
问题——"这次的深度误差有没有大到污染动作"，并把答案压成 0~1 的 ``depth_confidence``，
供 UI 显示、供导出侧写 ``warnings``。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseFrame,
)
from app.backend.postprocess.depth_sign import DepthSignReport
from app.backend.postprocess.gaps import GapFillReport
from app.backend.postprocess.ik_fit import (
    FitReport,
    bone_length_cv,
    estimate_image_scale,
    node_positions,
)
from app.backend.postprocess.smoothing import SmoothingInfo
from app.backend.skeleton.topology import BONE_SEGMENTS, JITTER_GROUPS, node_map

logger = get_logger("postprocess.quality")

Verdict = Literal["ready", "review", "rerecord"]

VERDICT_LABELS: dict[str, str] = {
    "ready": "可直接使用",
    "review": "可用但建议复核",
    "rerecord": "建议重录",
}

# 只用四肢末端 + 髋做跳变检测：躯干点变化小，容易掩盖真实抖动
JUMP_LANDMARKS: tuple[int, ...] = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)

# 深度置信度的归一化尺度：指标达到这些值就认为"完全不可信"（惩罚项 = 1）。
# 来源：合成序列基线里 z 抖动 ≈ 0.03 m、骨长 CV ≈ 32%，取"半量级"作为 0→1 的标尺。
Z_JITTER_FULL_PENALTY = 0.03
BONE_CV_FULL_PENALTY = 0.30
REPROJ_FULL_PENALTY = 0.02

# 三项惩罚的权重：z 抖动最直观（角度直接受影响），骨长次之，重投影再次之
DEPTH_PENALTY_WEIGHTS = {"jitter": 0.5, "bone": 0.3, "reproj": 0.2}


class QualityConfig(BaseModel):
    min_ready_ratio: float = Field(0.95, ge=0, le=1)
    min_review_ratio: float = Field(0.8, ge=0, le=1)
    min_ready_visibility: float = Field(0.6, ge=0, le=1)
    max_jump_ready: float = Field(0.2, gt=0, description="归一化坐标下的单帧位移上限")
    max_jump_review: float = Field(0.35, gt=0)
    max_frame_gap: int = Field(2, ge=1, description="跳变检测允许的帧号间隔（采样视频用）")
    # ---- 深度专项 ----
    max_z_jitter_ready: float = Field(
        0.01, gt=0, description="z 相邻帧 RMS 上限（米），超过则不能判 ready"
    )
    max_z_jitter_review: float = Field(
        0.03,
        gt=0,
        description="z 相邻帧 RMS 超过该值直接给「建议从侧面重拍」；与 Z_JITTER_FULL_PENALTY 对齐",
    )
    max_bone_cv_ready: float = Field(0.10, gt=0, description="骨长变异系数上限（10%）")
    max_bone_cv_review: float = Field(
        0.30, gt=0, description="骨长变异系数上限（30%）；与 BONE_CV_FULL_PENALTY 对齐"
    )
    min_depth_confidence: float = Field(
        0.6, ge=0, le=1, description="低于该值不能判 ready（降级为 review）"
    )
    min_depth_confidence_review: float = Field(
        0.35, ge=0, le=1, description="低于该值提示重新拍摄（仍只降到 review，不强制重录）"
    )
    # ---- 拍摄质量（任务 6.2）----
    min_subject_height: float = Field(
        0.5, gt=0, le=1, description="人物占画面高度的下限，低于它提示离得太远"
    )
    camera_high_ratio: float = Field(
        1.2, gt=1, description="上下半身投影尺度比超过该值 → 机位偏高（俯拍）"
    )
    camera_low_ratio: float = Field(
        0.83, gt=0, le=1, description="上下半身投影尺度比低于该值 → 机位偏低（仰拍）"
    )
    max_perspective_deviation: float = Field(
        0.4, gt=0, description="|尺度比 - 1| 超过该值 → 透视过强（距离过近或广角）"
    )
    max_area_change: float = Field(
        0.02, gt=0, description="人物框面积的帧间变化率上限，超过说明在快速走近/远离"
    )
    # ---- 可回滚开关（DoD：「关闭所有新开关时输出与改动前一致」）----
    enable_depth_gate: bool = Field(
        True, description="深度不可信时是否把 ready 降级为 review；关闭后只报指标不降级"
    )
    enable_framing_rules: bool = Field(
        True, description="是否启用拍摄质量规则（机位/距离/走近）；关闭后不追加这些 issues"
    )
    # ---- 矢状面运动提示（深度方向歧义的"事前"提示）----
    enable_sagittal_hint: bool = Field(
        True,
        description="四肢长时间沿视线方向运动时提示补拍侧视机位。"
        "这类动作在正面机位下前后方向本来就没有观测，后处理只能猜，属于拍摄问题",
    )
    sagittal_projected_ratio: float = Field(
        0.5,
        gt=0,
        le=1,
        description="骨段投影比低于该值即视为「指向/远离镜头」，前后方向不可观测",
    )
    sagittal_min_frame_share: float = Field(
        0.25,
        gt=0,
        le=1,
        description="至少这么多比例的帧出现上述情况才提示；偶尔一次的伸展不该报警",
    )


class FramingReport(BaseModel):
    """拍摄质量指标（任务 6.2）。

    全部是**图像平面 + 米制骨长**的比值，不需要相机内参：

    - ``subject_height``：头顶到脚踝的归一化高度，< 0.5 说明离得太远；
    - ``perspective_ratio``：``(头宽/头宽米制) / (踝宽/踝宽米制)``。上下半身在同一人的
      世界坐标里都有米制宽度，所以比值直接反映"谁离镜头更近"——
      > 1 表示头比脚近（机位偏高/俯拍），< 1 表示脚比头近（机位偏低/仰拍）；
      偏离 1 很多则是距离过近或广角畸变（透视过强）；
    - ``area_change_per_frame``：人物包围框面积的帧间变化率中位数，快速朝镜头走动时会明显变大。

    为什么不用"肩中点与髋中点的垂直偏移"判俯仰：那个量在相机俯仰时是被
    ``cos(pitch)`` 整体压缩的，和"人物前倾/下蹲"无法区分，会把深蹲误判成机位问题。
    """

    subject_height: float = Field(0.0, ge=0, description="头顶到脚踝的归一化高度 0~1")
    perspective_ratio: float = Field(
        1.0, ge=0, description="上半身/下半身的投影尺度比，1 = 无透视"
    )
    area_change_per_frame: float = Field(0.0, ge=0, description="包围框面积帧间变化率（中位数）")
    torso_y_ratio: float = Field(
        0.0, ge=0, description="（肩中点→髋中点的 y 距离）/ 身高，信息项，不参与判级"
    )


class QualityReport(BaseModel):
    frames_total: int = 0
    frames_valid: int = 0
    valid_frame_ratio: float = 0.0
    interpolated_frames: int = 0
    mean_visibility: float = 0.0
    mean_score: float = 0.0
    max_jump: float = 0.0
    max_jump_frame: int | None = None
    # ---- 深度专项（任务 5.3）----
    z_jitter_rms: float = Field(0.0, ge=0, description="相邻帧 z 的 RMS（米）")
    bone_length_cv: float = Field(0.0, ge=0, description="骨长变异系数的最大值（0~1）")
    mean_reproj_error: float = Field(
        0.0, ge=0, description="3D 反投影与 2D 关键点的平均误差（归一化坐标）"
    )
    depth_confidence: float = Field(
        0.0, ge=0, le=1, description="深度可信度 0~1：1 = 深度误差未污染动作，0 = 完全不可信"
    )
    framing: FramingReport = Field(default_factory=FramingReport, description="拍摄质量（机位/距离）")
    verdict: Verdict = "rerecord"
    verdict_label: str = VERDICT_LABELS["rerecord"]
    issues: list[str] = Field(default_factory=list)
    smoothing: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def z_jitter_rms(frames: Sequence[PoseFrame]) -> float:
    """相邻有效帧 ``world_landmarks.z`` 的 RMS（米），与基准脚本口径一致。"""
    ordered = [frame for frame in sorted(frames, key=lambda item: item.index) if frame.valid]
    squared: list[float] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.index - previous.index != 1:
            continue
        if len(previous.world_landmarks) < LANDMARK_COUNT or len(current.world_landmarks) < LANDMARK_COUNT:
            continue
        for indices in JITTER_GROUPS.values():
            for index in indices:
                a = previous.world_landmarks[index]
                b = current.world_landmarks[index]
                if a.visibility <= 0 or b.visibility <= 0:
                    continue
                squared.append((b.z - a.z) ** 2)
    if not squared:
        return 0.0
    return float(math.sqrt(float(np.mean(squared))))


def depth_confidence(
    *,
    z_jitter: float,
    bone_cv: float,
    reproj: float,
    has_world: bool = True,
) -> float:
    """把三项深度指标压成 0~1 的可信度（1 = 最好）。

    惩罚项都是"超过标尺就记 1"的线性归一化，再用固定权重相加：
    ``confidence = 1 - (0.5*z抖动 + 0.3*骨长CV + 0.2*重投影)``。
    没有 world landmarks 时直接给最低分（0.0）——那种情况下深度根本不存在。
    """
    if not has_world:
        return 0.0
    jitter_penalty = min(1.0, max(0.0, z_jitter / Z_JITTER_FULL_PENALTY))
    bone_penalty = min(1.0, max(0.0, bone_cv / BONE_CV_FULL_PENALTY))
    reproj_penalty = min(1.0, max(0.0, reproj / REPROJ_FULL_PENALTY))
    total = (
        DEPTH_PENALTY_WEIGHTS["jitter"] * jitter_penalty
        + DEPTH_PENALTY_WEIGHTS["bone"] * bone_penalty
        + DEPTH_PENALTY_WEIGHTS["reproj"] * reproj_penalty
    )
    return float(min(1.0, max(0.0, 1.0 - total)))


def _max_bone_cv(frames: Sequence[PoseFrame]) -> float:
    cv = bone_length_cv(frames)
    return float(max(cv.values())) if cv else 0.0


# 只统计四肢长骨：前踢腿/前伸胳膊这类"指向镜头"的动作就是在这些骨段上发生的
LIMB_SEGMENT_KEYS: tuple[str, ...] = (
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def sagittal_motion_share(
    frames: Sequence[PoseFrame],
    *,
    aspect: float = 9 / 16.0,
    threshold: float = 0.5,
) -> float:
    """四肢骨段"沿视线方向"的帧占比（0~1）。

    判据：``2D 投影长度 / (k * 观测骨长) < threshold``。比例越小说明这条骨段越顺着
    镜头方向躺着 —— 此时 2D 已经把 ``|Δz|`` 定死，只剩符号（前面还是后面）没有观测。
    这是"前踢腿被解成后踢腿"这类问题的**拍摄期**根因，比事后调参更能解决问题。
    """
    if not frames:
        return 0.0
    nodes = node_map()
    hit = 0
    counted = 0
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        positions = node_positions(frame.world_landmarks)
        if not positions:
            continue
        scale, _, _ = estimate_image_scale(frame, aspect)
        if scale <= 1e-9:
            continue
        counted += 1
        for node in BONE_SEGMENTS:
            if node.key not in LIMB_SEGMENT_KEYS or node.index is None:
                continue
            parent = positions.get(node.parent or "")
            child = positions.get(node.key)
            if parent is None or child is None:
                continue
            length = float(np.linalg.norm(child - parent))
            expected = scale * length
            if expected <= 1e-9:
                continue
            # 投影长度必须用 **2D** 观测算（u = x*aspect），不能用世界点的 x/y：
            # 后者本身就是米制，除以 k·L 之后恒等于 1，判据会永远不触发。
            point = frame.landmarks[node.index]
            parent_node = nodes.get(node.parent or "")
            if parent_node is None or parent_node.index is None:
                continue
            anchor = frame.landmarks[parent_node.index]
            projected = math.hypot(
                (point.x - anchor.x) * aspect, point.y - anchor.y
            )
            if projected / expected < threshold:
                hit += 1
                break
    return hit / counted if counted else 0.0


# 头顶用脸部点（0~10 里最小的 y），脚底用两踝中点；米制缺省值用于没有世界坐标的引擎
FACE_INDICES = tuple(range(11))
DEFAULT_HEAD_WIDTH = 0.15
DEFAULT_ANKLE_WIDTH = 0.18


def _bbox_area(frame: PoseFrame) -> float:
    points = [landmark for landmark in frame.landmarks if landmark.visibility > 0]
    if len(points) < 2:
        return 0.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def framing_metrics(frames: Sequence[PoseFrame]) -> FramingReport:
    """拍摄质量指标（任务 6.2）：只用图像平面 + 米制骨长的比值，不需要相机内参。"""
    heights: list[float] = []
    ratios: list[float] = []
    torso_ratios: list[float] = []
    areas: list[float] = []

    for frame in frames:
        if len(frame.landmarks) < LANDMARK_COUNT:
            continue
        points = frame.landmarks
        face_y = min(points[index].y for index in FACE_INDICES)
        ankle_y = 0.5 * (points[LEFT_ANKLE].y + points[RIGHT_ANKLE].y)
        # 取绝对值：不同来源的归一化 y 方向可能相反（合成数据就是 y 向上）
        height = abs(ankle_y - face_y)
        if height <= 1e-6:
            continue
        heights.append(height)
        torso = abs(
            0.5 * (points[LEFT_SHOULDER].y + points[RIGHT_SHOULDER].y)
            - 0.5 * (points[LEFT_HIP].y + points[RIGHT_HIP].y)
        )
        torso_ratios.append(torso / height)
        areas.append(_bbox_area(frame))

        head_width = abs(points[LEFT_EAR].x - points[RIGHT_EAR].x)
        ankle_width = abs(points[LEFT_ANKLE].x - points[RIGHT_ANKLE].x)
        head_meters, ankle_meters = DEFAULT_HEAD_WIDTH, DEFAULT_ANKLE_WIDTH
        if len(frame.world_landmarks) >= LANDMARK_COUNT:
            world = frame.world_landmarks
            head_meters = math.dist(
                (world[LEFT_EAR].x, world[LEFT_EAR].y, world[LEFT_EAR].z),
                (world[RIGHT_EAR].x, world[RIGHT_EAR].y, world[RIGHT_EAR].z),
            )
            ankle_meters = math.dist(
                (world[LEFT_ANKLE].x, world[LEFT_ANKLE].y, world[LEFT_ANKLE].z),
                (world[RIGHT_ANKLE].x, world[RIGHT_ANKLE].y, world[RIGHT_ANKLE].z),
            )
        if head_width > 1e-6 and ankle_width > 1e-6 and head_meters > 1e-6 and ankle_meters > 1e-6:
            ratios.append((head_width / head_meters) / (ankle_width / ankle_meters))

    changes: list[float] = []
    for previous, current in zip(areas, areas[1:], strict=False):
        if previous > 1e-6:
            changes.append(abs(current - previous) / previous)

    def _median(values: list[float], fallback: float = 0.0) -> float:
        return float(statistics.median(values)) if values else fallback

    return FramingReport(
        subject_height=round(_median(heights), 4),
        perspective_ratio=round(_median(ratios, 1.0), 4),
        area_change_per_frame=round(_median(changes), 5),
        torso_y_ratio=round(_median(torso_ratios), 4),
    )


def max_frame_jump(
    frames: Sequence[PoseFrame], config: QualityConfig | None = None
) -> tuple[float, int | None]:
    """相邻有效帧之间四肢末端的最大位移（归一化坐标）。"""
    config = config or QualityConfig()
    ordered = sorted(
        (frame for frame in frames if frame.valid and frame.landmarks),
        key=lambda item: item.index,
    )
    best = 0.0
    best_index: int | None = None
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.index - previous.index > config.max_frame_gap:
            continue
        for index in JUMP_LANDMARKS:
            a, b = previous.landmark(index), current.landmark(index)
            if a is None or b is None:
                continue
            if a.visibility <= 0 or b.visibility <= 0:
                continue
            distance = float(np.hypot(a.x - b.x, a.y - b.y))
            if distance > best:
                best, best_index = distance, current.index
    return best, best_index


def build_quality_report(
    frames: Sequence[PoseFrame],
    config: QualityConfig | None = None,
    *,
    smoothing: SmoothingInfo | None = None,
    gaps: GapFillReport | None = None,
    fit: FitReport | None = None,
    depth_sign: DepthSignReport | None = None,
    aspect: float | None = None,
) -> QualityReport:
    config = config or QualityConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    total = len(ordered)
    valid_frames = [frame for frame in ordered if frame.valid and frame.landmarks]
    ratio = len(valid_frames) / total if total else 0.0

    visibilities = [frame.mean_visibility() for frame in valid_frames]
    mean_visibility = float(np.mean(visibilities)) if visibilities else 0.0
    scores = [frame.score for frame in valid_frames if frame.score > 0]
    mean_score = float(np.mean(scores)) if scores else 0.0

    jump, jump_frame = max_frame_jump(ordered, config)
    issues: list[str] = []
    if ratio < config.min_review_ratio:
        issues.append(f"有效帧比例仅 {ratio:.1%}，人物可能长时间被遮挡或离开画面")
    elif ratio < config.min_ready_ratio:
        issues.append(f"有效帧比例 {ratio:.1%}，存在可见缺口")
    if mean_visibility and mean_visibility < config.min_ready_visibility:
        issues.append(f"平均置信度 {mean_visibility:.2f} 偏低，画面可能模糊或逆光")
    if jump > config.max_jump_review:
        issues.append(f"第 {jump_frame} 帧附近出现 {jump:.2f} 的跳变，可能是跟踪串人")
    elif jump > config.max_jump_ready:
        issues.append(f"存在 {jump:.2f} 的明显位移，动作过快或关键点抖动")

    # ---- 深度专项（任务 5.3）----
    z_jitter = z_jitter_rms(ordered)
    bone_cv = _max_bone_cv(ordered)
    has_world = any(len(frame.world_landmarks) >= LANDMARK_COUNT for frame in valid_frames)
    reproj = float(fit.reproj_error_norm) if fit is not None and fit.applied else 0.0
    confidence = depth_confidence(
        z_jitter=z_jitter, bone_cv=bone_cv, reproj=reproj, has_world=has_world
    )

    if not has_world:
        issues.append("当前引擎没有输出米制世界坐标，深度（前后方向）完全不可用，只能判读图像平面内的动作")
    elif confidence < config.min_depth_confidence:
        # 只报"主导项"：z 抖动归 z 抖动，骨长归骨长，避免出现"深度抖动 0.0 cm"这种自相矛盾的提示
        reasons: list[str] = []
        if z_jitter > config.max_z_jitter_ready:
            reasons.append(f"深度抖动 {z_jitter * 100:.1f} cm/帧")
        if bone_cv > config.max_bone_cv_ready:
            reasons.append(f"骨长波动 {bone_cv:.1%}")
        if reproj > 0:
            reasons.append(f"2D 一致性 {reproj:.4f}")
        detail = "、".join(reasons) if reasons else "综合指标"
        if z_jitter > config.max_z_jitter_review or bone_cv > config.max_bone_cv_review:
            hint = "建议从侧面重拍：图像平面可信，前后方向（深度）误差过大"
        else:
            hint = "矢状面动作建议补拍一个侧面机位"
        issues.append(f"深度可信度 {confidence:.2f} 偏低（{detail}），{hint}")

    if (
        ratio >= config.min_ready_ratio
        and (not mean_visibility or mean_visibility >= config.min_ready_visibility)
        and jump <= config.max_jump_ready
    ):
        verdict: Verdict = "ready"
    elif ratio >= config.min_review_ratio and jump <= config.max_jump_review:
        verdict = "review"
    else:
        verdict = "rerecord"

    # 深度不可信时最多只到 review：不推翻"帧质量没问题"，但明确要求人工复核。
    # 引擎根本不输出世界坐标时（has_world=False）不做降级 —— 那种情况下 z 不参与任何计算，
    # 强行降级只会让所有结果都变成"需复核"，把真正的深度问题淹没掉。
    if (
        config.enable_depth_gate
        and has_world
        and verdict == "ready"
        and confidence < config.min_depth_confidence
    ):
        verdict = "review"

    # ---- 拍摄质量（任务 6.2）----
    framing = framing_metrics(valid_frames)
    if config.enable_framing_rules:
        if framing.subject_height and framing.subject_height < config.min_subject_height:
            issues.append(
                f"人物只占画面高度 {framing.subject_height:.0%}：离镜头太远，深度估计更不可靠，"
                "建议让头到脚占画面 60%~90%"
            )
        persp = framing.perspective_ratio
        if persp > 0 and abs(persp - 1.0) > config.max_perspective_deviation:
            side = "机位偏高（俯拍，头大脚小）" if persp > 1 else "机位偏低（仰拍，头小脚大）"
            issues.append(
                f"透视过强：头部与脚部的投影尺度比 {persp:.2f}，{side}；"
                "请把机位摆到与胸口齐平、退到 3~5 米外用标准焦段拍摄"
            )
        elif persp > config.camera_high_ratio:
            issues.append(f"机位偏高（头大脚小，尺度比 {persp:.2f}）：建议把相机降到与胸口齐平")
        elif 0 < persp < config.camera_low_ratio:
            issues.append(f"机位偏低（头小脚大，尺度比 {persp:.2f}）：建议把相机抬到与胸口齐平")
        if framing.area_change_per_frame > config.max_area_change:
            issues.append(
                f"人物框面积每帧变化 {framing.area_change_per_frame:.1%}：正在快速走近/远离镜头，"
                "前后（深度）位移不可信；建议横向移动或保持距离固定"
            )
        if config.enable_sagittal_hint:
            share = sagittal_motion_share(
                valid_frames,
                aspect=aspect or 9 / 16.0,
                threshold=config.sagittal_projected_ratio,
            )
            if share >= config.sagittal_min_frame_share:
                issues.append(
                    f"约 {share:.0%} 的帧里四肢顺（或逆）着镜头方向伸展，"
                    "前后方向只有符号没有观测：这类动作（如向前踢腿）从正面拍必然存在"
                    "「朝前来还是朝后去」的二义性；建议补拍一个正侧机位"
                )

    notes: list[str] = []
    if gaps is not None and gaps.frames_filled:
        notes.append(f"已插值补齐 {gaps.frames_filled} 帧（标记 interpolated=true）")
    if gaps is not None and gaps.frames_missing:
        notes.append(f"{gaps.frames_missing} 帧缺失未补，导出时 valid=false")
    if smoothing is not None and smoothing.applied:
        notes.append(f"平滑方法：{smoothing.method}")
    if fit is not None and fit.applied:
        rigid = (
            f"，另 {fit.frames_rigidified} 帧未做 2D 拟合但已按标定骨长重建"
            if fit.frames_rigidified
            else ""
        )
        notes.append(
            f"骨长约束拟合：{fit.frames_fitted}/{fit.frames} 帧，"
            f"重投影 {fit.reproj_error_norm:.4f}（归一化坐标），{fit.milliseconds_per_frame:.1f} ms/帧"
            + rigid
        )
    if depth_sign is not None and depth_sign.notes and depth_sign.mode != "off":
        notes.extend(depth_sign.notes)
    notes.append(
        f"深度可信度 {confidence:.2f}（z 抖动 {z_jitter * 100:.2f} cm、骨长波动 {bone_cv:.2%}、"
        f"重投影 {reproj:.4f}）；x/y 来自 2D 检测，可信；z 仅为单目估计值"
    )

    return QualityReport(
        frames_total=total,
        frames_valid=len(valid_frames),
        valid_frame_ratio=round(ratio, 4),
        interpolated_frames=sum(1 for frame in ordered if frame.interpolated),
        mean_visibility=round(mean_visibility, 4),
        mean_score=round(mean_score, 4),
        max_jump=round(jump, 4),
        max_jump_frame=jump_frame,
        z_jitter_rms=round(z_jitter, 6),
        bone_length_cv=round(bone_cv, 6),
        mean_reproj_error=round(reproj, 6),
        depth_confidence=round(confidence, 4),
        framing=framing,
        verdict=verdict,
        verdict_label=VERDICT_LABELS[verdict],
        issues=issues,
        smoothing=smoothing.model_dump(mode="json") if smoothing is not None else {},
        notes=notes,
    )
