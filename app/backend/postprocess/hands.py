"""手部 21 点的时序稳定（掉帧补全 + 姿态迟滞 + 相对姿态平滑）。

**为什么不能复用身体那套**：
- ``gaps.py`` 只处理 ``landmarks[33]``，``frame.hands`` 完全不参与。手一旦在某帧
  没被检出，该帧就是 ``hands=None``，导出端把 48 个指关节整段丢掉，播放时退回模型
  默认手型 —— 用户看到的就是"上一帧还握着拳，下一帧突然松开"；
- 手部动作频率远高于躯干，身体那套「≤5 帧才插值」既补不上，也不该把手指姿态抹平。

三段逻辑（左右手各自独立处理，绝不互相串）：

1. **左右手配对**：用「手部腕点 ↔ 人体腕点 + 上一帧腕点」打分，纠正 handedness 的
   偶发翻转（两手交叉时最明显）；
2. **掉帧补全**：短缺口在两端观测之间线性插值；超出插值窗口后沿用上一帧的**相对
   手型**，并把腕点贴到当前帧的人体腕点上 —— 手跟着手臂走，只有手指姿态保持不变；
3. **姿态迟滞**：单帧姿态相对上一帧突变（握拳 ↔ 张开这类）**不立刻采纳**，必须连续
   ``confirm_frames`` 帧观测一致才切换，并在确认窗口内做过渡。这就是"要看到另一个
   手指关节动作被连续证实，手型才允许改变"。

不做的事：长缺口不凭空生成新姿态（超过 ``hold_max_frames`` 仍然留空）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LEFT_WRIST, RIGHT_WRIST, Landmark, PoseFrame
from app.backend.engine.enhancements import HAND_LANDMARK_COUNT
from app.backend.postprocess.one_euro import (
    OneEuroParams,
    OneEuroVectorFilter,
    default_hand_one_euro_params,
    timestamps_to_dt,
)

logger = get_logger("postprocess.hands")

HAND_SIDES: tuple[str, str] = ("left", "right")
HAND_SIDE_LABELS: dict[str, str] = {"left": "Left", "right": "Right"}
_BODY_WRIST_INDEX: dict[str, int] = {"left": LEFT_WRIST, "right": RIGHT_WRIST}

# handedness 与空间位置冲突时的代价（单位：手长）。取 2 表示"宁可相信腕点位置，
# 也不相信单帧 handedness"——两手交叉时 MediaPipe 经常把左右报反。
_LABEL_PENALTY = 2.0
# 上一帧腕点在"左右配对"里的权重：它负责跨帧连续性，位置（人体腕点）负责正确性。
_PREV_WRIST_WEIGHT = 0.5
# 姿态差取"位移最大的 5 个点"均值：指尖才是握拳/张开的判据，掌心点几乎不动。
_TOP_K = 5

ObservedKind = str  # "observed" / "interpolated" / "held" / "missing"


class HandStabilityConfig(BaseModel):
    """手部时序稳定参数。默认值面向"30fps、全身入镜、单手或双手"的常见素材。"""

    enabled: bool = True
    max_gap_frames: int = Field(6, ge=0, le=60, description="≤N 帧的掉帧用两端观测插值补回")
    hold_max_frames: int = Field(
        24, ge=0, le=300, description="插值窗口之外最多沿用上一帧手型多少帧（约 0.8s@30fps）"
    )
    smooth: bool = True
    one_euro: OneEuroParams = Field(
        default_factory=default_hand_one_euro_params,
        description="手部**相对姿态**的平滑参数（比躯干更跟手）",
    )
    motion_threshold: float = Field(
        0.6, gt=0, le=10, description="相对手长的姿态跳变阈值；超过即要求连续确认"
    )
    confirm_frames: int = Field(
        2, ge=1, le=6, description="姿态切换需要连续几帧观测一致；1 = 关闭迟滞"
    )


class HandSideStats(BaseModel):
    side: str
    frames_observed: int = 0
    frames_interpolated: int = 0
    frames_held: int = 0
    frames_missing: int = 0
    rejected_jumps: int = 0

    @property
    def frames_present(self) -> int:
        return self.frames_observed + self.frames_interpolated + self.frames_held


class HandStabilityReport(BaseModel):
    applied: bool = False
    frames_total: int = 0
    frames_with_hands: int = 0
    sides: list[HandSideStats] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def side(self, name: str) -> HandSideStats | None:
        for item in self.sides:
            if item.side == name:
                return item
        return None


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def stabilize_hands(
    frames: Sequence[PoseFrame], config: HandStabilityConfig | None = None
) -> HandStabilityReport:
    """原地稳定 ``frames`` 里的 ``hands`` / ``hand_sides``，返回统计报告。"""
    config = config or HandStabilityConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    report = HandStabilityReport(frames_total=len(ordered))
    if not ordered:
        return report
    if not config.enabled:
        report.notes.append("手部时序稳定已关闭（postprocess.hand_stability.enabled=false）")
        return report

    series = _collect_series(ordered)
    if not any(item is not None for values in series.values() for item in values):
        report.notes.append("整段没有检测到手部，跳过手部稳定")
        return report

    kinds: dict[str, list[ObservedKind]] = {}
    rejected: dict[str, int] = {}
    for side in HAND_SIDES:
        kinds[side] = _fill_gaps(ordered, series[side], side, config)
        rejected[side] = _apply_hysteresis(series[side], kinds[side], config)
        if config.smooth:
            _smooth_relative(series[side], ordered, config.one_euro)

    stats = _write_back(ordered, series, kinds, rejected)
    report.sides = stats
    report.frames_with_hands = sum(1 for frame in ordered if frame.hands)
    report.applied = True
    report.notes.extend(_report_notes(report))
    logger.debug(
        "手部稳定：%s 帧有手（总 %s 帧），%s",
        report.frames_with_hands,
        report.frames_total,
        "；".join(report.notes) or "无异常",
    )
    return report


# --------------------------------------------------------------------------- #
# 1) 逐帧收集：左右手配对
# --------------------------------------------------------------------------- #
def _collect_series(
    frames: Sequence[PoseFrame],
) -> dict[str, list[list[Landmark] | None]]:
    """把逐帧的 ``hands`` 拆成左右两条时间线；缺失的帧留 ``None``。"""
    series: dict[str, list[list[Landmark] | None]] = {
        side: [None] * len(frames) for side in HAND_SIDES
    }
    previous_wrists: dict[str, Landmark] = {}
    for position, frame in enumerate(frames):
        hands = [hand for hand in (frame.hands or []) if len(hand) >= HAND_LANDMARK_COUNT]
        if not hands:
            continue
        assigned = _assign_sides(hands, frame.hand_sides or [], frame, previous_wrists)
        for side, hand in assigned.items():
            series[side][position] = hand
            previous_wrists[side] = hand[0]
    return series


def _assign_sides(
    hands: Sequence[list[Landmark]],
    labels: Sequence[str],
    frame: PoseFrame,
    previous_wrists: dict[str, Landmark],
) -> dict[str, list[Landmark]]:
    """给最多两只手分配左右侧，返回 ``{side: hand}``。

    打分只有三项：handedness 标签（冲突重罚）、与人体对应腕点的距离、
    与上一帧同侧腕点的距离。位置项按手长归一化，因此远近景、大小手都可比。
    """
    candidates = list(hands[: len(HAND_SIDES)])
    if not candidates:
        return {}

    body_wrists: dict[str, Landmark | None] = {}
    for side in HAND_SIDES:
        point = frame.landmark(_BODY_WRIST_INDEX[side])
        body_wrists[side] = point if point is not None and point.visibility > 0.05 else None

    scales = [max(_hand_scale(hand), 1e-4) for hand in candidates]
    label_list = [
        str(labels[index]).strip().lower() if index < len(labels) else ""
        for index in range(len(candidates))
    ]

    def cost(index: int, side: str) -> float:
        hand = candidates[index]
        total = 0.0
        label = label_list[index]
        if label in HAND_SIDES and label != side:
            total += _LABEL_PENALTY
        wrist = body_wrists.get(side)
        if wrist is not None:
            total += _distance_2d(hand[0], wrist) / scales[index]
        previous = previous_wrists.get(side)
        if previous is not None:
            total += _PREV_WRIST_WEIGHT * _distance_2d(hand[0], previous) / scales[index]
        return total

    if len(candidates) == 1:
        best = min(HAND_SIDES, key=lambda side: cost(0, side))
        return {best: candidates[0]}

    # 两只手：直接枚举左右两种分配（只有 2 种，比匈牙利算法直观得多）
    direct = (HAND_SIDES[0], HAND_SIDES[1])
    swapped = (HAND_SIDES[1], HAND_SIDES[0])
    direct_cost = cost(0, direct[0]) + cost(1, direct[1])
    swapped_cost = cost(0, swapped[0]) + cost(1, swapped[1])
    chosen = direct if direct_cost <= swapped_cost else swapped
    return {chosen[0]: candidates[0], chosen[1]: candidates[1]}


# --------------------------------------------------------------------------- #
# 2) 掉帧补全
# --------------------------------------------------------------------------- #
def _fill_gaps(
    frames: Sequence[PoseFrame],
    series: list[list[Landmark] | None],
    side: str,
    config: HandStabilityConfig,
) -> list[ObservedKind]:
    """补掉帧，返回逐帧来源标记。"""
    kinds: list[ObservedKind] = ["observed" if item is not None else "missing" for item in series]
    total = len(series)
    position = 0
    while position < total:
        if series[position] is not None:
            position += 1
            continue
        start = position
        while position < total and series[position] is None:
            position += 1
        end = position
        length = end - start
        before = series[start - 1] if start > 0 else None
        after = series[end] if end < total else None

        if before is not None and after is not None and length <= config.max_gap_frames:
            for offset in range(start, end):
                ratio = (offset - start + 1) / (length + 1)
                series[offset] = _lerp_hand(before, after, ratio)
                kinds[offset] = "interpolated"
            continue

        # 只剩"沿用上一帧手型"这一种合法补法；片头缺口（before 为空）不伪造。
        if before is not None and length <= config.max_gap_frames + config.hold_max_frames:
            for offset in range(start, end):
                wrist = _body_wrist(frames[offset], side)
                series[offset] = _copy_hand(before, anchor=wrist)
                kinds[offset] = "held"
    return kinds


# --------------------------------------------------------------------------- #
# 3) 姿态迟滞
# --------------------------------------------------------------------------- #
def _apply_hysteresis(
    series: list[list[Landmark] | None], kinds: list[ObservedKind], config: HandStabilityConfig
) -> int:
    """单帧姿态突变必须被连续帧证实才采纳；返回被拒绝的突变次数。"""
    rejected = 0
    last: list[Landmark] | None = None
    pending: list[tuple[int, list[Landmark]]] = []
    held_positions: list[int] = []
    for position, hand in enumerate(series):
        if hand is None:
            continue
        if last is None:
            last = hand
            pending.clear()
            held_positions.clear()
            continue
        if _pose_delta(hand, last) <= config.motion_threshold:
            # 回到旧手型 = 之前那一下确实是误检，暂扣的计数保留为"拦下的跳变"
            last = hand
            pending.clear()
            held_positions.clear()
            continue

        pending.append((position, hand))
        window = pending[-config.confirm_frames :]
        confirmed = len(window) >= config.confirm_frames and all(
            _pose_delta(item, window[-1][1]) <= config.motion_threshold for _, item in window
        )
        if confirmed:
            # 新姿态被连续证实：暂扣期间的"拒绝"撤销，并在确认窗口内做过渡，
            # 避免一帧直接跳到位（真动作只是晚 1 帧、且是渐变进入的）。
            rejected -= len(held_positions)
            start_pose = last
            accepted = window[-1][1]
            for order, (slot, _) in enumerate(window):
                ratio = (order + 1) / len(window)
                series[slot] = _lerp_hand(start_pose, accepted, ratio)
                if kinds[slot] == "held":
                    kinds[slot] = "interpolated"
            last = accepted
            pending.clear()
            held_positions.clear()
            continue

        # 单帧突变、后续帧还没跟上：保持上一手型，只把腕点平移跟随当前观测。
        series[position] = _copy_hand(last, anchor=hand[0])
        kinds[position] = "held"
        held_positions.append(position)
        rejected += 1
    return rejected


# --------------------------------------------------------------------------- #
# 4) 平滑（只平滑相对腕点的姿态，腕点本身跟随观测，避免手和手臂脱开）
# --------------------------------------------------------------------------- #
def _smooth_relative(
    series: list[list[Landmark] | None],
    frames: Sequence[PoseFrame],
    params: OneEuroParams,
) -> None:
    deltas = timestamps_to_dt([frame.timestamp_ms for frame in frames])
    filters = [OneEuroVectorFilter(3, params) for _ in range(HAND_LANDMARK_COUNT)]
    for position, hand in enumerate(series):
        if hand is None:
            continue
        wrist = hand[0]
        smoothed: list[Landmark] = [wrist.model_copy()]
        for index in range(1, min(len(hand), HAND_LANDMARK_COUNT)):
            point = hand[index]
            value = filters[index].filter(
                (point.x - wrist.x, point.y - wrist.y, point.z - wrist.z), deltas[position]
            )
            smoothed.append(
                Landmark(
                    x=wrist.x + float(value[0]),
                    y=wrist.y + float(value[1]),
                    z=wrist.z + float(value[2]),
                    visibility=point.visibility,
                    presence=point.presence,
                )
            )
        smoothed.extend(hand[HAND_LANDMARK_COUNT:])
        series[position] = smoothed


# --------------------------------------------------------------------------- #
# 5) 写回 + 统计
# --------------------------------------------------------------------------- #
def _write_back(
    frames: Sequence[PoseFrame],
    series: dict[str, list[list[Landmark] | None]],
    kinds: dict[str, list[ObservedKind]],
    rejected: dict[str, int],
) -> list[HandSideStats]:
    stats: dict[str, HandSideStats] = {
        side: HandSideStats(side=side, rejected_jumps=rejected.get(side, 0)) for side in HAND_SIDES
    }
    for position, frame in enumerate(frames):
        present: list[tuple[str, list[Landmark]]] = []
        for side in HAND_SIDES:
            hand = series[side][position]
            kind = kinds[side][position]
            side_stats = stats[side]
            if hand is None:
                side_stats.frames_missing += 1
                continue
            if kind == "interpolated":
                side_stats.frames_interpolated += 1
            elif kind == "held":
                side_stats.frames_held += 1
            else:
                side_stats.frames_observed += 1
            present.append((side, hand))

        if not present:
            frame.hands = None
            frame.hand_sides = []
            continue
        frame.hands = [[point.model_copy() for point in hand] for _, hand in present]
        frame.hand_sides = [HAND_SIDE_LABELS[side] for side, _ in present]
    return [stats[side] for side in HAND_SIDES]


def _report_notes(report: HandStabilityReport) -> list[str]:
    notes: list[str] = []
    for item in report.sides:
        if not item.frames_present:
            continue
        parts = [f"{item.frames_observed} 帧直采"]
        if item.frames_interpolated:
            parts.append(f"{item.frames_interpolated} 帧插值")
        if item.frames_held:
            parts.append(f"{item.frames_held} 帧沿用上一手型")
        if item.rejected_jumps:
            parts.append(f"拦下 {item.rejected_jumps} 次单帧姿态跳变")
        text = f"{item.side} 手：{'、'.join(parts)}"
        if item.frames_missing:
            text += f"；{item.frames_missing} 帧无观测（长缺口未伪造）"
        notes.append(text)
    return notes


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _distance_2d(a: Landmark, b: Landmark) -> float:
    """手部 z 是相对腕点的深度，跨帧不可比，因此配对/跳变判定只用 x/y。"""
    return math.hypot(a.x - b.x, a.y - b.y)


def _hand_scale(hand: Sequence[Landmark]) -> float:
    """手长：腕点 → 中指 MCP。抓握时该距离几乎不变，适合做姿态差的归一化。"""
    if len(hand) > 9:
        span = _distance_2d(hand[0], hand[9])
        if span > 1e-6:
            return span
    if len(hand) < 2:
        return 0.0
    return max(_distance_2d(hand[0], point) for point in hand[1:])


def _pose_delta(a: Sequence[Landmark], b: Sequence[Landmark]) -> float:
    """相对腕点的姿态差（按手长归一化）。

    只看"位移最大的 5 个点"：握拳 ↔ 张开主要改变指尖，掌心/腕点几乎不动，
    取均值会被 16 个不动的点稀释掉，导致该拦的跳变拦不住。
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    scale = _hand_scale(a)
    if scale <= 1e-6:
        return 0.0
    displacements: list[float] = []
    for index in range(1, min(len(a), len(b), HAND_LANDMARK_COUNT)):
        dx = (a[index].x - a[0].x) - (b[index].x - b[0].x)
        dy = (a[index].y - a[0].y) - (b[index].y - b[0].y)
        displacements.append(math.hypot(dx, dy))
    if not displacements:
        return 0.0
    top = sorted(displacements, reverse=True)[:_TOP_K]
    return (sum(top) / len(top)) / scale


def _copy_hand(hand: Sequence[Landmark], *, anchor: Landmark | None = None) -> list[Landmark]:
    """复制手型；``anchor`` 给定时把腕点平移到该位置（相对手型不变，z 不动）。"""
    dx = 0.0
    dy = 0.0
    if anchor is not None and hand:
        dx = anchor.x - hand[0].x
        dy = anchor.y - hand[0].y
    return [
        Landmark(
            x=point.x + dx,
            y=point.y + dy,
            z=point.z,
            visibility=point.visibility,
            presence=point.presence,
        )
        for point in hand
    ]


def _lerp_hand(a: Sequence[Landmark], b: Sequence[Landmark], ratio: float) -> list[Landmark]:
    """两个 21 点手型按比例线性插值；置信度取两端较小值（补出来的点不算高可信）。"""
    ratio = min(1.0, max(0.0, ratio))
    merged: list[Landmark] = []
    for index in range(min(len(a), len(b))):
        point = a[index].lerp(b[index], ratio)
        point.visibility = min(a[index].visibility, b[index].visibility)
        merged.append(point)
    return merged


def _body_wrist(frame: PoseFrame, side: str) -> Landmark | None:
    point = frame.landmark(_BODY_WRIST_INDEX[side])
    if point is None or point.visibility <= 0.05:
        return None
    return point


__all__ = [
    "HAND_SIDES",
    "HandSideStats",
    "HandStabilityConfig",
    "HandStabilityReport",
    "stabilize_hands",
]
