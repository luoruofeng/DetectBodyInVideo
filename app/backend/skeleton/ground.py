"""脚部着地锚定（ground contact）。

深蹲时 MediaPipe world landmarks 的原点始终跟着髋走，髋的真实下沉被抹掉；
如果直接把「髋相对坐标」交给渲染器，膝一弯，脚踝反而向髋靠近，看起来就是
**双脚离地**。

本模块用一条保守的地面约束修正这件事：

1. 逐帧取左右脚各自的最低可见点（踝 / 脚尖），再用**滞回**挑出支撑脚，
   避免左右脚因噪声来回切换造成的阶跃；
2. 在整条人物轨迹上取较高分位数作为「站立参考腿长」``span_ref``；
3. 支撑脚跨度先做**零相位高斯低通**再参与计算（整段视频离线处理，用对称核卷积，
   所以压噪声不会引入延迟，也不会有 Butterworth 的过冲）；原地动作时脚几乎不动，
   逐帧检测噪声本来会被原样放大成整段髋部的上下抖动，"脚踩不实地面"就是它造成的；
4. 把 ``root.y`` 调整为 ``span(t) - span_ref``，使脚底保持在固定高度，髋随下蹲下沉；
5. 再把支撑脚（踝 / 脚尖里更低那个）钉到 ``-span(t)`` 上，并用两骨 IK 重排膝的
   位置（``lock_support_foot``）。只做第 4 步的话，髋虽然稳了，脚相对髋的竖直
   位置仍是逐帧估计值，看到的就是"整个人在上下颠 + 脚在离地/落地之间抖"。

竖直方向稳定之后还剩一个问题：``joints`` 是**髋局部坐标**，髋左右/前后一晃，
脚在渲染里就跟着晃 —— 这就是"原地站着却一直没踩实、脚在地上漂"的来源。
第 6 步专门处理它：

6. 取两脚踝点、按"脚越低权重越大"合成水平基准点，先做零相位低通，再按
   "脚在地上时它相对身体的位置只因身体位移而变"逐帧累加成 ``root.x`` / ``root.z``
   （``anchor_horizontal``）。脚在水平面里被固定在原地，髋绕着脚晃，而不是脚跟着
   髋漂。用增量（而不是"锚点 − 基准"）是为了换脚/单脚离地时 root 保持连续、不跳变。

``joints`` 仍是髋局部坐标（导出契约不变）。水平锚定只是把"身体相对脚"的位移
写进 ``root.x/z``，所以 world 模式的 ``rootMotion.translation`` 现在也带水平分量：
它描述的是"脚踩住地面时髋相对脚怎么动"，**不是绝对走位**（人物真的横移时，
水平位移会被吃掉一部分，等同原地演示）。腾空帧（span 大于参考值）表现为髋略高于
参考位，而不是被强行按回地面。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.backend.skeleton.transform import ConvertedFrame, JointPosition, Vector3

#  ankle / foot_index 在 ARKit 命名下的脚部关节；取其中最低的可见点作为着地点。
FOOT_JOINTS: tuple[str, ...] = (
    "left_foot_joint",
    "right_foot_joint",
    "left_toes_joint",
    "right_toes_joint",
)

# 左右脚各自的 (踝, 脚尖)。支撑脚判定按"单只脚的最低点"来做，
# 这样同一只脚的踝/脚尖不会互相抢着当脚底。
SIDE_FOOT_JOINTS: dict[str, tuple[str, str]] = {
    "left": ("left_foot_joint", "left_toes_joint"),
    "right": ("right_foot_joint", "right_toes_joint"),
}

# 锁脚需要的腿链：髋 → 膝 → 踝，外加随踝刚体平移的脚部件。
LEG_JOINTS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "left": (
        "left_upLeg_joint",
        "left_leg_joint",
        "left_foot_joint",
        ("left_toes_joint", "left_toesEnd_joint"),
    ),
    "right": (
        "right_upLeg_joint",
        "right_leg_joint",
        "right_foot_joint",
        ("right_toes_joint", "right_toesEnd_joint"),
    ),
}

# 脚到髋的竖直跨度超出这个区间基本是误检（或人物只露出上半身），不参与锚定。
MIN_FOOT_SPAN_METERS = 0.25
MAX_FOOT_SPAN_METERS = 1.30

# 少于这么多帧就估不出噪声，直接不平滑（短片段按原始值处理）。
MIN_SMOOTH_FRAMES = 5
DEFAULT_FPS = 30.0


class GroundContactConfig(BaseModel):
    """地面锚定参数。默认开启，``enabled=False`` 时逐位透传输入。"""

    enabled: bool = True
    quantile: float = Field(
        0.85, ge=0.5, le=1.0, description="站立参考腿长的分位数；偏高可避开深蹲帧"
    )
    min_visibility: float = Field(0.3, ge=0.0, le=1.0)
    min_span_meters: float = Field(MIN_FOOT_SPAN_METERS, gt=0)
    max_span_meters: float = Field(MAX_FOOT_SPAN_METERS, gt=0)
    smooth_cutoff_hz: float = Field(
        3.0,
        ge=0.0,
        le=20.0,
        description="支撑脚跨度的零相位高斯低通截止频率（Hz）。原地动作的髋部上下抖动主要来自"
        "逐帧检测噪声（>8 Hz），而真实的下蹲/起伏在 2 Hz 以内，3 Hz 能压掉抖动又不拖慢动作。"
        "0 = 关闭平滑",
    )
    switch_margin_meters: float = Field(
        0.02,
        ge=0.0,
        le=0.2,
        description="支撑脚切换滞回（米）：另一只脚必须明显更低才换支撑脚，"
        "避免左右脚在噪声里来回切换导致 root.y 阶跃",
    )
    lock_support_foot: bool = Field(
        True,
        description="支撑脚锁定：把支撑脚（踝/脚尖）按平滑后的跨度钉在固定高度上，"
        "并用两骨 IK 重排膝的位置。只调 root.y 时，脚自身的关键点噪声仍会让人看到"
        "脚在离地/落地之间抖；锁定后脚底在整段里高度恒定",
    )
    max_foot_shift_meters: float = Field(
        0.08,
        gt=0,
        le=0.5,
        description="单帧允许修正的脚部位移上限（米）。超过说明这一帧的跨度和脚观测本身就矛盾"
        "（误检/极端遮挡），该帧跳过锁定而不是把腿掰断",
    )
    anchor_horizontal: bool = Field(
        True,
        description="水平脚底锚定：把双脚踝点轨迹按着地权重合成水平基准点，做零相位低通后"
        "逐帧累加成 root.x/z，使脚在水平面上固定在原地、髋绕脚晃动。关闭后水平方向退回"
        "旧行为（脚随髋局部坐标漂移）",
    )
    horizontal_cutoff_hz: float = Field(
        3.0,
        ge=0.0,
        le=20.0,
        description="水平基准点的零相位高斯低通截止频率（Hz）。0 = 不平滑"
        "（脚被逐帧钉死，髋会跟着关键点噪声左右抖）",
    )
    horizontal_weight_meters: float = Field(
        0.15,
        gt=0,
        le=1.0,
        description="双脚着地加权的软阈值（米）：某只脚比最低的那只高出该量时，"
        "它在水平基准点里的权重降到 1/e。取大值（≈0.15）等于「两脚都在地上时取平均」"
        "——两只脚的世界坐标噪声是独立的，平均能把噪声减半；取很小的值就等于"
        "「只用支撑脚」，抬腿时不会把支撑脚拽走，代价是非支撑脚自己的噪声全留在画面上",
    )
    max_horizontal_offset_meters: float = Field(
        0.5,
        gt=0,
        le=2.0,
        description="root 水平偏移的硬上限（米，相对首个有效帧）。正常站立只有几厘米；"
        "触及上限说明脚的关键点本身崩了（误检/长时遮挡），"
        "此时宁可留一点漂移也不要把人挪出画面",
    )
    max_horizontal_step_meters: float = Field(
        0.1,
        gt=0,
        le=1.0,
        description="单帧允许累加的 root 水平修正上限（米）。超过说明这一帧的脚点跳变"
        "（误检 / 支撑脚切换瞬间），按比例缩放而不是让整个人弹一下",
    )
    horizontal_min_both_ratio: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="双脚同时可见的帧占比低于该值时不启用水平锚定：只有一只脚时"
        "支撑脚判定和脚点都不可靠，宁可不锚定也不要凭空把身体挪走",
    )


class GroundReference(BaseModel):
    """单个人物轨迹的地面参考。"""

    track_id: int | None = None
    foot_span_meters: float = 0.0
    frames_used: int = 0
    span_jitter_before_meters: float = Field(
        0.0, description="平滑前支撑脚跨度的逐帧变化 RMS（米），用于说明抖动被压掉多少"
    )
    span_jitter_after_meters: float = Field(
        0.0, description="平滑后支撑脚跨度的逐帧变化 RMS（米）"
    )
    horizontal_locked: bool = Field(
        False, description="是否启用了水平脚底锚定（结果写进 root.x/root.z）"
    )
    horizontal_drift_before_meters: float = Field(
        0.0, description="锚定前脚在水平面里偏离中位的 90 分位半径（米），即肉眼看到的滑步"
    )
    horizontal_drift_after_meters: float = Field(
        0.0, description="锚定后残余的水平漂移 RMS（米）：只剩被低通滤掉的逐帧噪声"
    )


def side_foot_contact(
    frame: ConvertedFrame,
    side: str,
    config: GroundContactConfig | None = None,
) -> tuple[str, float] | None:
    """单只脚的最低着地点：返回 ``(关节名, 到髋的竖直距离)``。

    取踝 / 脚尖里更低的那个 —— 它才是这一帧真正踩地的点，锁脚时也要动它。
    """
    config = config or GroundContactConfig()
    if not frame.valid:
        return None
    best: tuple[str, float] | None = None
    for name in SIDE_FOOT_JOINTS.get(side, ()):  # 未知 side 返回空元组，静默跳过
        point = frame.joints.get(name)
        if point is None or point.visibility < config.min_visibility:
            continue
        span = -point.y
        if not (
            math.isfinite(span)
            and config.min_span_meters <= span <= config.max_span_meters
        ):
            continue
        if best is None or span > best[1]:
            best = (name, span)
    return best


def side_foot_span(
    frame: ConvertedFrame,
    side: str,
    config: GroundContactConfig | None = None,
) -> float | None:
    """单只脚的最低点（踝 / 脚尖中更低者）到髋的竖直距离（米）。"""
    contact = side_foot_contact(frame, side, config)
    return contact[1] if contact is not None else None


def foot_span(frame: ConvertedFrame, config: GroundContactConfig | None = None) -> float | None:
    """髋到最低可见脚的竖直距离（米）。``joints`` 是 y 向上的髋局部坐标。"""
    config = config or GroundContactConfig()
    values = [
        span
        for span in (
            side_foot_span(frame, "left", config),
            side_foot_span(frame, "right", config),
        )
        if span is not None
    ]
    return max(values) if values else None


def side_foot_base(
    frame: ConvertedFrame,
    side: str,
    config: GroundContactConfig | None = None,
) -> tuple[float, float, float] | None:
    """单只脚的水平参考点：返回 ``(x, z, 到髋的竖直距离)``。

    优先用踝点：脚尖点在 world 坐标里前后摆动更明显（MediaPipe 对脚趾的估计更差），
    拿它当基准会把脚尖噪声写进 root.z。踝点不可见时才退回脚尖。
    """
    config = config or GroundContactConfig()
    span = side_foot_span(frame, side, config)
    if span is None:
        return None
    for name in SIDE_FOOT_JOINTS.get(side, ()):
        point = frame.joints.get(name)
        if point is None or point.visibility < config.min_visibility:
            continue
        if not (math.isfinite(point.x) and math.isfinite(point.z)):
            continue
        return (point.x, point.z, span)
    return None


def _combine_foot_base(
    entries: Sequence[tuple[float, float, float]], weight_scale: float
) -> tuple[float, float] | None:
    """按"脚越低权重越大"把左右脚的位置合成一个水平基准点（诊断/工具用）。"""
    if not entries:
        return None
    if len(entries) == 1:
        return (entries[0][0], entries[0][1])
    lowest = max(entry[2] for entry in entries)
    weights = [
        math.exp(-max(0.0, lowest - entry[2]) / weight_scale) for entry in entries
    ]
    total = sum(weights)
    if total <= 0:  # pragma: no cover - 权重恒为正，不会走到
        return (entries[0][0], entries[0][1])
    return (
        sum(weight * entry[0] for weight, entry in zip(weights, entries, strict=True)) / total,
        sum(weight * entry[1] for weight, entry in zip(weights, entries, strict=True)) / total,
    )


def foot_base_xz(
    frame: ConvertedFrame, config: GroundContactConfig | None = None
) -> tuple[float, float] | None:
    """双脚水平基准点（米，髋局部坐标，按着地高度加权）。

    只作诊断/工具用途：真正参与 ``root.x/z`` 反解的是 ``_horizontal_offsets`` 里的
    **增量式脚底锁定**（见该函数的说明），不是这个单帧基准点。
    """
    config = config or GroundContactConfig()
    entries = [
        entry
        for entry in (
            side_foot_base(frame, "left", config),
            side_foot_base(frame, "right", config),
        )
        if entry is not None
    ]
    return _combine_foot_base(entries, config.horizontal_weight_meters)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _support_series(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig
) -> dict[int | None, dict[int, tuple[str, str, float]]]:
    """逐帧支撑脚，按人物轨迹分组：``{track: {index: (side, 着地关节名, 跨度)}}``。

    支撑脚带滞回：只有另一只脚比当前支撑脚低出 ``switch_margin_meters`` 才切换。
    两只脚差不多高时（原地站立/小幅度律动）锁死上一次的选择，跨度就不会因为
    噪声在左右脚之间来回跳。
    """
    series: dict[int | None, dict[int, tuple[str, str, float]]] = {}
    support: dict[int | None, str] = {}
    for frame in sorted(frames, key=lambda item: item.index):
        candidates: dict[str, tuple[str, float]] = {}
        for side in ("left", "right"):
            contact = side_foot_contact(frame, side, config)
            if contact is not None:
                candidates[side] = contact
        if not candidates:
            continue
        current = support.get(frame.track_id)
        if current not in candidates:
            current = max(candidates, key=lambda side: candidates[side][1])
        else:
            other = next((side for side in candidates if side != current), None)
            if (
                other is not None
                and candidates[other][1]
                > candidates[current][1] + config.switch_margin_meters
            ):
                current = other
        support[frame.track_id] = current
        joint, span = candidates[current]
        series.setdefault(frame.track_id, {})[frame.index] = (current, joint, span)
    return series


def _spans_from_series(
    series: dict[int | None, dict[int, tuple[str, str, float]]]
) -> dict[int | None, dict[int, float]]:
    return {
        track_id: {index: entry[2] for index, entry in by_index.items()}
        for track_id, by_index in series.items()
    }


def _references_from_series(
    series: dict[int | None, dict[int, float]], config: GroundContactConfig
) -> dict[int | None, GroundReference]:
    references: dict[int | None, GroundReference] = {}
    for track_id, span_by_index in series.items():
        spans = list(span_by_index.values())
        if not spans:
            continue
        references[track_id] = GroundReference(
            track_id=track_id,
            # 保留全精度：参考值提前四舍五入会让恒定站姿产生 1e-6 的虚假位移。
            foot_span_meters=_percentile(spans, config.quantile),
            frames_used=len(spans),
        )
    return references


def estimate_ground_references(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig | None = None
) -> dict[int | None, GroundReference]:
    """按人物轨迹估计站立参考腿长。"""
    config = config or GroundContactConfig()
    return _references_from_series(_spans_from_series(_support_series(frames, config)), config)


def _frame_step_seconds(frames: Sequence[ConvertedFrame]) -> float:
    """相邻帧的时间间隔（秒），取中位数以抗采样不均匀。"""
    times = sorted(frame.t for frame in frames if frame.t >= 0)
    diffs = [later - earlier for earlier, later in zip(times, times[1:]) if later > earlier]
    if not diffs:
        return 1.0 / DEFAULT_FPS
    return float(np.median(diffs))


def _finite_diff_rms(values: np.ndarray) -> float:
    """逐帧变化量的 RMS：衡量"抖动"大小（米/帧）。"""
    if values.size < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(values) ** 2)))


def _planar_scatter_meters(xs: np.ndarray, zs: np.ndarray) -> float:
    """水平面里偏离中位的 90 分位半径（米）：脚在地面上"漂"多远。

    用分位数而不是 RMS：单帧误检（脚点飞到半米外）会把 RMS 整个带偏，
    而报告这个数字是为了说明"漂移被压掉多少"，必须对离群值稳健。
    """
    if xs.size == 0:
        return 0.0
    radii = np.hypot(xs - float(np.median(xs)), zs - float(np.median(zs)))
    return float(np.percentile(radii, 90))


def _zero_phase_lowpass(
    values: np.ndarray, step_seconds: float, cutoff_hz: float
) -> np.ndarray:
    """整段序列的零相位高斯低通。

    离线处理整段视频，所以可以直接用**对称核卷积**：没有相位延迟、也没有
    Butterworth 那种"下蹲起步时先向上冲一下"的过冲（过冲会被误看成髋部上浮）。
    截止频率按 ``H(f) = exp(-2π²σ²f²)`` 换算成时间域标准差 ``σ = 1/(2π·fc)``。

    边界用镜像填充（``reflect``）：抖动序列在两端会被镜像延续，不会像
    ``edge`` / ``odd`` 那样在首尾几帧里造出一段假的上升或下降斜坡。
    """
    if cutoff_hz <= 0 or values.size < MIN_SMOOTH_FRAMES or step_seconds <= 0:
        return values.copy()
    sigma = 1.0 / (2.0 * math.pi * cutoff_hz * step_seconds)
    if not math.isfinite(sigma) or sigma <= 0:
        return values.copy()
    radius = max(1, min(int(math.ceil(3.0 * sigma)), values.size - 1))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    total = float(kernel.sum())
    if total <= 0:
        return values.copy()
    kernel /= total
    padded = np.pad(values, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _fill_missing(
    span_by_index: dict[int, float], valid_indexes: Sequence[int]
) -> dict[int, float]:
    """把"有效帧但这一帧看不见脚"的帧按最近可用跨度补上。

    不补的话，脚被短暂遮挡的那一帧 ``root.y`` 会瞬间回到 0（等于站立），
    深蹲到一半时表现为整个身体弹一下。
    """
    if not span_by_index:
        return {}
    available = sorted(span_by_index)
    filled = dict(span_by_index)
    for index in valid_indexes:
        if index in filled:
            continue
        nearest = min(available, key=lambda candidate: abs(candidate - index))
        filled[index] = span_by_index[nearest]
    return filled


def _fill_missing_points(
    points_by_index: dict[int, tuple[float, float]], valid_indexes: Sequence[int]
) -> dict[int, tuple[float, float]]:
    """同 ``_fill_missing``，但补的是 ``(x, z)`` 水平基准点。"""
    if not points_by_index:
        return {}
    available = sorted(points_by_index)
    filled = dict(points_by_index)
    for index in valid_indexes:
        if index in filled:
            continue
        nearest = min(available, key=lambda candidate: abs(candidate - index))
        filled[index] = points_by_index[nearest]
    return filled


def _side_trajectories(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig
) -> dict[int | None, dict[str, dict[int, tuple[float, float, float]]]]:
    """逐帧逐脚的水平位置：``{track: {side: {index: (x, z, 到髋的竖直距离)}}}``。"""
    series: dict[int | None, dict[str, dict[int, tuple[float, float, float]]]] = {}
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid:
            continue
        for side in ("left", "right"):
            base = side_foot_base(frame, side, config)
            if base is None:
                continue
            series.setdefault(frame.track_id, {}).setdefault(side, {})[frame.index] = base
    return series


def _horizontal_offsets(
    frames: Sequence[ConvertedFrame],
    config: GroundContactConfig,
    valid_indexes: dict[int | None, list[int]],
) -> tuple[
    dict[int | None, dict[int, tuple[float, float]]],
    dict[int | None, tuple[float, float]],
]:
    """水平脚底锚定：返回 ``({track: {index: root 水平偏移}}, {track: (锚定前, 锚定后)})``。

    **增量式脚底锁定**：脚踩在地上时，它相对身体的位置只会因为身体整体位移而变，
    所以 ``root(t) = root(t-1) − 双脚逐帧位移的加权平均``。位移里的共同分量
    （身体晃动、真实走位）被完整抵掉，脚因此钉在原地、髋绕脚晃；只剩被低通滤掉的
    逐帧噪声留在脚上（厘米级），而不是整段慢慢漂走。

    几个刻意的选择：

    - 平均的对象是**逐帧位移**，不是位置：两脚都着地时它们相对身体的位移相同，
      平均只会压低各自的独立噪声；抬起的那只脚位移完全不同，按高度权重排掉即可。
      如果改平均位置，换脚/抬腿会让基准点整体平移，支撑脚反而被拽着滑半个站距。
    - 用**增量**而不是"锚点 − 基准"：同样是为了让 root 在换脚、遮挡时保持连续，不跳变。
    - 逐帧增量在 ``max_horizontal_step_meters`` 处按比例缩放：脚点误检不至于让整个人弹一下。
    - 轨迹先做零相位低通，避免把逐帧关键点噪声积分成一段随机游走。
    """
    if not config.anchor_horizontal:
        return {}, {}
    trajectories = _side_trajectories(frames, config)
    if not trajectories:
        return {}, {}
    step_seconds = _frame_step_seconds(frames)
    max_offset = config.max_horizontal_offset_meters
    max_step = config.max_horizontal_step_meters
    offsets: dict[int | None, dict[int, tuple[float, float]]] = {}
    metrics: dict[int | None, tuple[float, float]] = {}
    for track_id, sides in trajectories.items():
        index_set = set(sides.get("left", {})) | set(sides.get("right", {}))
        if len(index_set) < 2:
            continue
        both = sum(
            1
            for index in index_set
            if index in sides.get("left", {}) and index in sides.get("right", {})
        )
        if both / len(index_set) < config.horizontal_min_both_ratio:
            # 长期只有一只脚可见：基准点会随遮挡来回跳，宁可不锚定。
            continue

        smoothed: dict[str, dict[int, tuple[float, float]]] = {}
        for side, by_index in sides.items():
            keys = sorted(by_index)
            raw_x = np.asarray([by_index[index][0] for index in keys], dtype=np.float64)
            raw_z = np.asarray([by_index[index][1] for index in keys], dtype=np.float64)
            smooth_x = _zero_phase_lowpass(raw_x, step_seconds, config.horizontal_cutoff_hz)
            smooth_z = _zero_phase_lowpass(raw_z, step_seconds, config.horizontal_cutoff_hz)
            smoothed[side] = _fill_missing_points(
                {
                    index: (float(x), float(z))
                    for index, x, z in zip(keys, smooth_x.tolist(), smooth_z.tolist(), strict=True)
                },
                valid_indexes.get(track_id, []),
            )

        root_x = 0.0
        root_z = 0.0
        previous: dict[str, tuple[float, float, float]] | None = None
        by_index_out: dict[int, tuple[float, float]] = {}
        raw_points: list[tuple[float, float]] = []
        locked_points: list[tuple[float, float]] = []
        for index in sorted(index_set):
            current: dict[str, tuple[float, float, float]] = {}
            for side in ("left", "right"):
                raw = sides.get(side, {}).get(index)
                smooth = smoothed.get(side, {}).get(index)
                if raw is None or smooth is None:
                    continue
                current[side] = (smooth[0], smooth[1], raw[2])
            if not current:
                continue
            if previous:
                # 关键：平均的是**逐帧位移**而不是位置。两脚都着地时它们相对身体的
                # 位移相同，平均只压低噪声；抬起的那只脚位移完全不同，按高度权重
                # 排掉即可，不会像"平均位置"那样把基准点整体拽到支撑脚上。
                lowest = max(entry[2] for entry in current.values())
                steps: list[tuple[float, float, float]] = []
                for side, (smooth_x, smooth_z, span) in current.items():
                    prev = previous.get(side)
                    if prev is None:
                        continue
                    weight = math.exp(
                        -max(0.0, lowest - span) / config.horizontal_weight_meters
                    )
                    steps.append(
                        (
                            prev[0] - smooth_x,
                            prev[1] - smooth_z,
                            weight,
                        )
                    )
                total_weight = sum(step[2] for step in steps)
                if total_weight > 0:
                    step_x = sum(step[0] * step[2] for step in steps) / total_weight
                    step_z = sum(step[1] * step[2] for step in steps) / total_weight
                    length = math.hypot(step_x, step_z)
                    if length > max_step:
                        scale = max_step / length
                        step_x *= scale
                        step_z *= scale
                    root_x = min(max_offset, max(-max_offset, root_x + step_x))
                    root_z = min(max_offset, max(-max_offset, root_z + step_z))
            by_index_out[index] = (root_x, root_z)
            previous = current
            raw_entries = [
                sides[side][index] for side in current if index in sides.get(side, {})
            ]
            raw_base = _combine_foot_base(raw_entries, config.horizontal_weight_meters)
            if raw_base is not None:
                raw_points.append(raw_base)
                locked_points.append((root_x + raw_base[0], root_z + raw_base[1]))
        if not by_index_out:
            continue

        # 脚被遮挡 / 无效帧沿用最近的修正值，不要突然回到 0。
        filled: dict[int, tuple[float, float]] = {}
        latest = (0.0, 0.0)
        for index in valid_indexes.get(track_id, []):
            latest = by_index_out.get(index, latest)
            filled[index] = latest
        offsets[track_id] = filled
        metrics[track_id] = (
            _planar_scatter_meters(
                np.asarray([point[0] for point in raw_points], dtype=np.float64),
                np.asarray([point[1] for point in raw_points], dtype=np.float64),
            ),
            _planar_scatter_meters(
                np.asarray([point[0] for point in locked_points], dtype=np.float64),
                np.asarray([point[1] for point in locked_points], dtype=np.float64),
            ),
        )
    return offsets, metrics


def _leg_ik(
    hip: tuple[float, float, float],
    knee: tuple[float, float, float],
    ankle: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """两骨 IK：把踝放到 ``target``，同时保持大腿/小腿长度，膝朝观测方向弯。

    返回 ``(新膝, 新踝)``；腿长不可用或目标不可达时返回 ``None``。
    """
    def _sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _norm(v):
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    thigh = _norm(_sub(knee, hip))
    shank = _norm(_sub(ankle, knee))
    if thigh <= 1e-6 or shank <= 1e-6:
        return None

    vector = _sub(target, hip)
    distance = _norm(vector)
    if distance <= 1e-9 or distance > (thigh + shank) * 0.999:
        # 目标超出腿长：这一帧的"跨度"和脚的观测本身矛盾，放弃锁定而不是拉长骨段。
        return None

    along = (thigh * thigh - shank * shank + distance * distance) / (2.0 * distance)
    height_sq = max(0.0, thigh * thigh - along * along)
    height = math.sqrt(height_sq)
    axis = (vector[0] / distance, vector[1] / distance, vector[2] / distance)

    # 让膝盖留在"观测到的膝关节所在的垂直方向"上：只改腿的伸缩，不改朝向。
    observed = _sub(knee, hip)
    projection = observed[0] * axis[0] + observed[1] * axis[1] + observed[2] * axis[2]
    perp = (
        observed[0] - axis[0] * projection,
        observed[1] - axis[1] * projection,
        observed[2] - axis[2] * projection,
    )
    perp_norm = _norm(perp)
    if perp_norm <= 1e-6:  # 腿是直的（观测膝正好落在髋-踝连线上），给一个稳定的兜底方向
        fallback = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
        cross = (
            axis[1] * fallback[2] - axis[2] * fallback[1],
            axis[2] * fallback[0] - axis[0] * fallback[2],
            axis[0] * fallback[1] - axis[1] * fallback[0],
        )
        perp = cross
        perp_norm = _norm(perp)
        if perp_norm <= 1e-6:
            return None
    perp = (perp[0] / perp_norm, perp[1] / perp_norm, perp[2] / perp_norm)

    new_knee = (
        hip[0] + axis[0] * along + perp[0] * height,
        hip[1] + axis[1] * along + perp[1] * height,
        hip[2] + axis[2] * along + perp[2] * height,
    )
    return new_knee, target


def _lock_support_foot(
    frame: ConvertedFrame,
    side: str,
    contact_joint: str,
    target_y: float,
    config: GroundContactConfig,
) -> dict[str, JointPosition] | None:
    """把支撑脚（踝 / 脚尖）的竖直位置钉到 ``target_y``，膝按两骨 IK 重排。

    为什么只调 ``root.y`` 不够：髋被放到"平滑后的跨度"上之后，脚相对髋的位置
    仍然是逐帧估计出来的，关键点噪声会让脚在离地 1~2 cm 之间来回跳。把脚钉住、
    让膝去吸收这点差量，脚底才会真的稳定在地面上。
    """
    names = LEG_JOINTS.get(side)
    if names is None:
        return None
    hip_name, knee_name, ankle_name, foot_names = names
    hip = frame.joints.get(hip_name)
    knee = frame.joints.get(knee_name)
    ankle = frame.joints.get(ankle_name)
    contact = frame.joints.get(contact_joint)
    if hip is None or knee is None or ankle is None or contact is None:
        return None
    if contact.visibility < config.min_visibility:
        return None

    shift = target_y - contact.y
    if not math.isfinite(shift) or abs(shift) < 1e-6:
        return None
    if abs(shift) > config.max_foot_shift_meters:
        # 差得太多说明这一帧的"跨度"和脚的观测本身矛盾（误检/极端遮挡），不硬掰。
        return None
    target = (ankle.x, ankle.y + shift, ankle.z)
    solved = _leg_ik(
        (hip.x, hip.y, hip.z),
        (knee.x, knee.y, knee.z),
        (ankle.x, ankle.y, ankle.z),
        target,
    )
    if solved is None:
        return None
    new_knee, new_ankle = solved

    # 脚掌按踝的实际位移刚体平移，脚踝→脚尖的长度与朝向都不变。
    move = (new_ankle[1] - ankle.y)
    updated: dict[str, JointPosition] = {
        ankle_name: ankle.model_copy(
            update={"x": new_ankle[0], "y": new_ankle[1], "z": new_ankle[2]}
        ),
        knee_name: knee.model_copy(
            update={"x": new_knee[0], "y": new_knee[1], "z": new_knee[2]}
        ),
    }
    for name in foot_names:
        point = frame.joints.get(name)
        if point is None:
            continue
        updated[name] = point.model_copy(update={"y": point.y + move})
    return updated


def anchor_feet_to_ground(
    frames: Sequence[ConvertedFrame], config: GroundContactConfig | None = None
) -> tuple[list[ConvertedFrame], list[GroundReference]]:
    """把脚底固定在地面、髋随下蹲下沉，返回新帧与参考列表。

    做三件事：
    1. ``root.y = span(t) - span_ref``：恢复 world 模式被髋原点抹掉的竖直位移；
    2. 可选地把支撑脚钉在平滑后的跨度上并用两骨 IK 重排膝（``lock_support_foot``），
       让脚底高度在整段里恒定，而不是跟着关键点噪声上下跳；
    3. 可选的把双脚水平基准点低通后反解成 ``root.x/z``（``anchor_horizontal``），
       脚在水平面里固定在原地，髋绕脚晃动 —— 解决"原地站着脚却在地上漂"。
    """
    config = config or GroundContactConfig()
    if not config.enabled:
        return list(frames), []

    series = _support_series(frames, config)
    spans = _spans_from_series(series)
    references = _references_from_series(spans, config)
    if not references:
        return list(frames), []

    valid_indexes: dict[int | None, list[int]] = {}
    for frame in frames:
        if frame.valid:
            valid_indexes.setdefault(frame.track_id, []).append(frame.index)

    step_seconds = _frame_step_seconds(frames)
    smoothed: dict[int | None, dict[int, float]] = {}
    for track_id, span_by_index in spans.items():
        indexes = sorted(span_by_index)
        raw = np.asarray([span_by_index[index] for index in indexes], dtype=np.float64)
        filtered = _zero_phase_lowpass(raw, step_seconds, config.smooth_cutoff_hz)
        smoothed[track_id] = {
            index: float(value) for index, value in zip(indexes, filtered.tolist())
        }
        reference = references.get(track_id)
        if reference is not None:
            references[track_id] = reference.model_copy(
                update={
                    "span_jitter_before_meters": round(_finite_diff_rms(raw), 6),
                    "span_jitter_after_meters": round(_finite_diff_rms(filtered), 6),
                }
            )

    for track_id, span_by_index in smoothed.items():
        smoothed[track_id] = _fill_missing(span_by_index, valid_indexes.get(track_id, []))

    horizontal, horizontal_metrics = _horizontal_offsets(frames, config, valid_indexes)
    for track_id, (before, after) in horizontal_metrics.items():
        reference = references.get(track_id)
        if reference is None:
            continue
        references[track_id] = reference.model_copy(
            update={
                "horizontal_locked": True,
                "horizontal_drift_before_meters": round(before, 6),
                "horizontal_drift_after_meters": round(after, 6),
            }
        )

    outputs: list[ConvertedFrame] = []
    for frame in frames:
        reference = references.get(frame.track_id)
        span = smoothed.get(frame.track_id, {}).get(frame.index)
        if reference is None or span is None:
            outputs.append(frame)
            continue

        delta = round(span - reference.foot_span_meters, 6)
        offset_x, offset_z = horizontal.get(frame.track_id, {}).get(frame.index, (0.0, 0.0))
        update: dict[str, object] = {
            "root": Vector3(x=round(offset_x, 6), y=delta, z=round(offset_z, 6)),
            # 预览用：髋到脚底的高度（已平滑），直接把髋放到地面上，
            # 不要再自己取"当前帧最低脚"，否则会把逐帧噪声又画一遍。
            "ground_span": round(span, 6),
        }
        contact = series.get(frame.track_id, {}).get(frame.index)
        if config.lock_support_foot and contact is not None:
            side, contact_joint, _raw_span = contact
            locked = _lock_support_foot(frame, side, contact_joint, -span, config)
            if locked:
                # 只替换被锁的腿，其余关节保持原对象（不改调用方数据）。
                update["joints"] = {**frame.joints, **locked}
        outputs.append(frame.model_copy(update=update))
    return outputs, list(references.values())


__all__ = [
    "FOOT_JOINTS",
    "LEG_JOINTS",
    "SIDE_FOOT_JOINTS",
    "GroundContactConfig",
    "GroundReference",
    "anchor_feet_to_ground",
    "estimate_ground_references",
    "foot_base_xz",
    "foot_span",
    "side_foot_base",
    "side_foot_contact",
    "side_foot_span",
]
