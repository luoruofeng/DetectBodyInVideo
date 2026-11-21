"""深度符号先验：单目"每根骨段只剩 1 个比特"的解算层。

## 为什么只需要符号

``ik_fit`` 用弱透视投影 ``u = k·X + u0``、``v = k·Y + v0``。骨段真长 ``L`` 由标定给出
且是**硬约束**，2D 观测把 ``(ΔX, ΔY) = (Δu/k, Δv/k)`` 定死，于是

    ΔZ = ± sqrt(L² − ΔX² − ΔY²)

**大小被完全锁死，唯一未知的是一个比特（符号）**。前踢腿之所以解错，不是"深度不够准"，
而是这个比特由 MediaPipe 的**逐帧** ``world_landmarks.z`` 决定，而它没有时序上下文：
同一段动作里符号会来回翻，播放时就是"腿朝反方向踢"。

因此本模块既不输出米制坐标、也不输出骨长，只输出 ``{帧, 骨段} -> ±1``，交给
``ik_fit`` 的 hinge 残差去执行。这样做的直接收益：即使抬升模型在**幅度/尺度**上
完全不可信（跨域时必然如此），只要它的**方向判断**成立，本方案就有效。

## 两种后端

- ``temporal``（默认，零新增依赖）：直接读 ``world_landmarks`` 的 Δz 符号，做
  时间窗多数投票 + 短游程消除。它**不能**纠正整体方向（输入本身就是它要判的东西），
  只负责把帧间翻转压掉——这是"腿突然翻面"这类最刺眼问题的直接解药。
- ``model``：可选 ONNX 抬升器（VideoPose3D / H36M 17 点）只取符号。权重缺失、
  onnxruntime 异常、左右/深度轴约定不符时**自动退回 temporal**，绝不中断分析。

## 为什么必须做自校准

抬升模型的输出坐标系（y 朝上/朝下、z 朝里/朝外、H36M 左右约定）与 MediaPipe 不同，
而这些约定无法只靠"读文档"确认。``auto_calibrate`` 用同段视频里两者的符号一致率判断：
一致率 <0.3 说明整段深度被镜像 → 整体取反；一致率在 0.3~0.55 之间说明模型只是噪声
→ 直接丢弃。这样即使换模型、换权重、换左右约定，也不会把结果搞反。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame
from app.backend.postprocess.ik_fit import (
    BoneLengths,
    estimate_image_scale,
    node_positions,
)
from app.backend.skeleton.topology import BONE_SEGMENTS, node_map

logger = get_logger("postprocess.depth_sign")

# 符号图：``{帧号: {骨段 key: (符号, 置信度)}}``。ik_fit 只依赖这个结构，
# 不 import 本模块 —— 保持"拟合层不认识先验来源"的解耦。
SignMap = dict[int, dict[str, tuple[int, float]]]

# 少于这么多帧就不给符号：时序投票没有样本，等同于乱猜。
MIN_SEQUENCE_FRAMES = 30

ONNX_MODEL_FILENAME = "videopose3d-h36m.onnx"

# H36M 17 关节顺序（VideoPose3D / Human3.6M 通用约定）
# 0=hip(pelvis) 1=rhip 2=rknee 3=rankle 4=lhip 5=lknee 6=lankle 7=spine
# 8=thorax 9=neck/nose 10=head 11=lshoulder 12=lelbow 13=lwrist
# 14=rshoulder 15=relbow 16=rwrist
H36M_TO_NODE: dict[str, int] = {
    "right_hip": 1,
    "right_knee": 2,
    "right_ankle": 3,
    "left_hip": 4,
    "left_knee": 5,
    "left_ankle": 6,
    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}

# 抬升模型能给出符号的骨段（父节点取自 skeleton/topology.py 的唯一真源）
LIFTABLE_KEYS: frozenset[str] = frozenset(H36M_TO_NODE)

# 换左/右时用到的镜像表
_MIRROR_KEY = {
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
}


# --------------------------------------------------------------------------- #
# 配置与报告
# --------------------------------------------------------------------------- #
class DepthSignConfig(BaseModel):
    """深度符号先验的参数。默认关闭：关闭时输出与改动前逐位一致。"""

    enabled: bool = Field(False, description="总开关；关闭后不产生任何符号，ik_fit 行为不变")
    mode: Literal["temporal", "model"] = Field(
        "temporal",
        description="temporal = 用 world z 做时序一致性（零依赖）；"
        "model = 用 ONNX 抬升器判方向（需下载权重，缺失自动退回 temporal）",
    )
    model_path: str | None = Field(None, description=f"抬升模型路径；留空自动搜 models/{ONNX_MODEL_FILENAME}")
    window: int = Field(9, ge=1, le=121, description="多数投票的时间窗（样本数，不是帧数）")
    min_agreement: float = Field(
        0.6, ge=0.0, le=1.0, description="窗内一致率低于该值时不采信该骨段"
    )
    min_run: int = Field(3, ge=1, le=60, description="短于该长度的符号游程被相邻符号吞并（消除尖刺）")
    min_projected_ratio: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="temporal 模式的采样门控：投影比低于它说明骨段几乎正对镜头，"
        "2D 长度已经很小、符号对噪声极敏感（model 模式不受此限，那正是它要解的题）",
    )
    max_projected_ratio: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description="投影比高于它时 Δz≈0，符号没有实际影响，不必占用优化自由度",
    )
    min_depth_meters: float = Field(0.01, gt=0, description="|Δz| 小于该值时符号视为无意义")
    swap_left_right: bool = Field(
        False, description="抬升模型左右约定与本项目相反时置 True（H36M 存在已知的左右互换）"
    )
    depth_axis_inverted: bool = Field(
        False, description="抬升模型 z 轴朝向与本项目相反时置 True；auto_calibrate 会自动判断"
    )
    auto_calibrate: bool = Field(
        True, description="用同段视频的 world z 符号一致率自动判断是否镜像/丢弃模型结果"
    )


class DepthSignReport(BaseModel):
    """符号先验的执行情况，进质检 notes 与任务摘要。"""

    applied: bool = False
    mode: str = "off"
    frames: int = 0
    segments: int = Field(0, description="实际下发的 (帧 × 骨段) 符号数量")
    flips_before: int = Field(0, description="原始符号序列的翻转次数（未做时序平滑）")
    flips_after: int = Field(0, description="平滑后下发符号的翻转次数")
    disagreements: int = Field(0, description="平滑结果与原始观测符号相反的样本数")
    milliseconds: float = 0.0
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _flat_positions(frame: PoseFrame, aspect: float) -> dict[str, np.ndarray]:
    """2D 节点位置（``u = x*aspect``、``v = y``），与 ``ik_fit._normalized_positions`` 同口径。"""
    positions: dict[str, np.ndarray] = {}
    if len(frame.landmarks) < LANDMARK_COUNT:
        return positions
    for node in BONE_SEGMENTS:
        if node.index is None:
            continue
        point = frame.landmarks[node.index]
        positions[node.key] = np.array([point.x * aspect, point.y], dtype=np.float64)
    return positions


def _projected_ratio(
    parent_2d: np.ndarray, child_2d: np.ndarray, length: float, scale: float
) -> float | None:
    """「2D 投影长度 / 骨段正对镜头时该有的投影长度」。

    ≈1：骨段躺在画面平面内（深度分量≈0，符号无意义）；
    ≪1：骨段沿视线方向（深度大小被 2D 定死，只差符号 —— 本模块要解的题）。
    """
    expected = scale * length
    if expected <= 1e-9:
        return None
    projected = float(math.hypot(child_2d[0] - parent_2d[0], child_2d[1] - parent_2d[1]))
    return min(1.0, max(0.0, projected / expected))


# --------------------------------------------------------------------------- #
# 时序平滑
# --------------------------------------------------------------------------- #
def _count_flips(samples: Sequence[tuple[int, int]]) -> int:
    return sum(1 for (_, a), (_, b) in zip(samples, samples[1:], strict=False) if a != b)


def _despike(signs: list[int], min_run: int) -> list[int]:
    """把长度 < ``min_run`` 的游程并入两侧一致的邻居（只在两侧一致时合并，不制造新的翻面）。"""
    if min_run <= 1 or len(signs) < 3:
        return list(signs)
    values = list(signs)
    for _ in range(4):
        changed = False
        index = 0
        while index < len(values):
            end = index
            while end < len(values) and values[end] == values[index]:
                end += 1
            run = end - index
            if run < min_run and 0 < index and end < len(values) and values[index - 1] == values[end]:
                for cursor in range(index, end):
                    values[cursor] = values[index - 1]
                changed = True
            index = end
        if not changed:
            break
    return values


def _vote(
    samples: Sequence[tuple[int, int, float]],
    *,
    window: int,
    min_agreement: float,
    min_run: int,
) -> dict[int, tuple[int, float]]:
    """时间窗多数投票 → 短游程消除 → 一致率过滤。``samples`` 为 ``(帧号, 符号, 权重)``。"""
    if not samples:
        return {}
    half = max(1, window // 2)
    signs: list[int] = []
    confidences: list[float] = []
    for position, (_, raw_sign, _) in enumerate(samples):
        lo = max(0, position - half)
        hi = min(len(samples), position + half + 1)
        total = 0.0
        weight_sum = 0.0
        for _, sign, weight in samples[lo:hi]:
            total += sign * weight
            weight_sum += weight
        if weight_sum <= 1e-9:
            signs.append(raw_sign)
            confidences.append(0.0)
            continue
        ratio = total / weight_sum
        signs.append(1 if ratio > 0 else (-1 if ratio < 0 else raw_sign))
        confidences.append(min(1.0, abs(ratio)))

    smoothed = _despike(signs, min_run)
    result: dict[int, tuple[int, float]] = {}
    for (frame_index, _, _), sign, confidence in zip(samples, smoothed, confidences, strict=True):
        if confidence < min_agreement:
            continue
        result[frame_index] = (sign, round(float(confidence), 4))
    return result


# --------------------------------------------------------------------------- #
# 后端 1：时序一致性（零依赖）
# --------------------------------------------------------------------------- #
def _temporal_signals(
    frames: Sequence[PoseFrame],
    lengths: BoneLengths,
    config: DepthSignConfig,
    aspect: float,
) -> dict[str, list[tuple[int, int, float]]]:
    """从 ``world_landmarks`` 直接取 Δz 符号（未经任何平滑的原始观测）。"""
    signals: dict[str, list[tuple[int, int, float]]] = {node.key: [] for node in BONE_SEGMENTS}
    for frame in frames:
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        positions = node_positions(frame.world_landmarks)
        if not positions:
            continue
        scale, _, _ = estimate_image_scale(frame, aspect)
        flat = _flat_positions(frame, aspect) if scale > 1e-9 else {}
        for node in BONE_SEGMENTS:
            parent = positions.get(node.parent or "")
            child = positions.get(node.key)
            if parent is None or child is None:
                continue
            dz = float(child[2] - parent[2])
            if abs(dz) < config.min_depth_meters:
                continue
            length = lengths.get(node.key, node.mid)
            if flat and scale > 1e-9:
                parent_2d = flat.get(node.parent or "")
                child_2d = flat.get(node.key)
                if parent_2d is not None and child_2d is not None:
                    ratio = _projected_ratio(parent_2d, child_2d, length, scale)
                    if ratio is not None and (
                        ratio < config.min_projected_ratio or ratio > config.max_projected_ratio
                    ):
                        continue
            visibility = min(
                1.0,
                max(0.0, float(frame.landmarks[node.index].visibility))
                if node.index is not None and node.index < len(frame.landmarks)
                else 1.0,
            )
            if visibility <= 0.0:
                continue
            signals[node.key].append(
                (frame.index, 1 if dz > 0 else -1, max(1e-3, visibility))
            )
    return signals


# --------------------------------------------------------------------------- #
# 后端 2：可选 ONNX 抬升器（只取符号）
# --------------------------------------------------------------------------- #
def resolve_model_path(config: DepthSignConfig) -> str | None:
    """显式路径 → ``models_dir`` → 打包资源根 ``models/``。找不到返回 None。"""
    from pathlib import Path

    if config.model_path:
        path = Path(config.model_path)
        return str(path) if path.is_file() else None
    try:
        from app.backend.core.settings import get_settings

        settings = get_settings()
    except Exception:  # pragma: no cover - 配置不可用的极端情况
        return None
    candidates = [
        Path(settings.models_dir) / ONNX_MODEL_FILENAME,
        Path(settings.static_dir).parents[2] / "models" / ONNX_MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _h36m_input(
    frames: Sequence[PoseFrame], aspect: float
) -> tuple[np.ndarray, list[int], list[list[float]]]:
    """把 33 点序列转成 VideoPose3D 期望的 ``[T, 17, 2]``。

    归一化必须复刻其 ``crop_bbox`` + ``normalize_screen_coordinates``：用**1.25 外扩的
    人体框**把坐标映射到 ``[-1, 1]``。这一步与训练时不一致会让输出整体失效，
    是跨检测器复用的第一号坑。
    """
    nodes = node_map()
    landmark_of = {
        h36m_index: nodes[key].index
        for key, h36m_index in H36M_TO_NODE.items()
        if key in nodes and nodes[key].index is not None
    }

    sequence: list[list[list[float]]] = []
    frame_indices: list[int] = []
    confidences: list[list[float]] = []
    for frame in frames:
        if not frame.valid or len(frame.landmarks) < LANDMARK_COUNT:
            continue
        points = [(item.x * aspect, item.y) for item in frame.landmarks]
        visibilities = [max(0.0, min(1.0, float(item.visibility))) for item in frame.landmarks]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        half_width = max(1e-4, (right - left) * 1.25 / 2.0)
        half_height = max(1e-4, (bottom - top) * 1.25 / 2.0)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0

        row: list[list[float]] = []
        confidence_row: list[float] = []
        for h36m_index in range(17):
            landmark_index = landmark_of.get(h36m_index)
            if landmark_index is None or landmark_index >= LANDMARK_COUNT:
                row.append([0.0, 0.0])
                confidence_row.append(0.0)
                continue
            px, py = points[landmark_index]
            # 等价于 VideoPose3D 的 normalize_screen_coordinates：以框中心为原点、半宽高归一化
            row.append([(px - cx) / half_width, (py - cy) / half_height])
            confidence_row.append(visibilities[landmark_index])
        sequence.append(row)
        frame_indices.append(frame.index)
        confidences.append(confidence_row)

    if not sequence:
        return np.zeros((0, 17, 2), dtype=np.float32), [], []
    return np.asarray(sequence, dtype=np.float32), frame_indices, confidences


def _model_signals(
    frames: Sequence[PoseFrame],
    config: DepthSignConfig,
    aspect: float,
    notes: list[str],
) -> tuple[dict[str, list[tuple[int, int, float]]], int]:
    """跑 ONNX 抬升器，返回符号信号与参与帧数；任何失败都返回空并写 note。"""
    path = resolve_model_path(config)
    if not path:
        notes.append(
            f"未找到抬升模型（{ONNX_MODEL_FILENAME}），本次退回时序一致性模式；"
            "可用 scripts/export_videopose3d_onnx.py 离线导出权重"
        )
        return {}, 0
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - 依赖缺失
        notes.append(f"onnxruntime 不可用（{exc}），退回时序一致性模式")
        return {}, 0

    sequence, frame_indices, confidences = _h36m_input(frames, aspect)
    if sequence.shape[0] == 0:
        return {}, 0

    available = set(ort.get_available_providers())
    providers = [
        item
        for item in (
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        )
        if item in available
    ]
    try:
        session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        # 感受野 243 帧：不足时用边缘复制填充，保证每一帧都在"有上下文"的位置上被推断
        receptive = 243
        pad = max(0, receptive - sequence.shape[0])
        padded = (
            np.pad(sequence, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode="edge")
            if pad
            else sequence
        )
        output = session.run(None, {input_name: padded[None, ...]})[0]
        if output.ndim == 4:
            output = output[0]
        if pad:
            output = output[pad // 2 : pad // 2 + sequence.shape[0]]
    except Exception as exc:  # pragma: no cover - 权重损坏 / 形状不符
        notes.append(f"抬升模型推理失败（{exc}），退回时序一致性模式")
        return {}, 0

    notes.append(
        f"抬升模型：{path}（{','.join(providers) or 'CPU'}），只取深度符号，幅度与尺度一律丢弃"
    )

    nodes = node_map()
    signals: dict[str, list[tuple[int, int, float]]] = {key: [] for key in LIFTABLE_KEYS}
    for position, frame_index in enumerate(frame_indices):
        for key, h36m_index in H36M_TO_NODE.items():
            node = nodes.get(key)
            if node is None or node.parent is None:
                continue
            parent_h36m = H36M_TO_NODE.get(node.parent)
            if parent_h36m is None:
                continue
            dz = float(output[position, h36m_index, 2] - output[position, parent_h36m, 2])
            if abs(dz) < 1e-6:
                continue
            visibility = min(
                confidences[position][h36m_index], confidences[position][parent_h36m]
            )
            if visibility <= 0.0:
                continue
            signals[key].append((frame_index, 1 if dz > 0 else -1, max(1e-3, visibility)))
    return signals, len(frame_indices)


# --------------------------------------------------------------------------- #
# 自校准
# --------------------------------------------------------------------------- #
def _calibrate_model_signs(
    model: dict[str, list[tuple[int, int, float]]],
    reference: dict[str, list[tuple[int, int, float]]],
    config: DepthSignConfig,
    notes: list[str],
) -> dict[str, list[tuple[int, int, float]]]:
    """用同段视频的 world z 符号做一致性校准（自动判镜像 / 判无效）。"""
    reference_signs = {
        key: {frame_index: sign for frame_index, sign, _ in values}
        for key, values in reference.items()
    }
    agree = 0
    total = 0
    for key, values in model.items():
        observed = reference_signs.get(key) or {}
        for frame_index, sign, _ in values:
            other = observed.get(frame_index)
            if other is None:
                continue
            total += 1
            agree += 1 if other == sign else 0
    if total < 12:
        return model
    ratio = agree / total
    if not config.auto_calibrate:
        if config.depth_axis_inverted:
            for key, values in model.items():
                model[key] = [(index, -sign, weight) for index, sign, weight in values]
        notes.append(f"未启用自校准：模型与 world z 的符号一致率 {ratio:.0%}")
        return model
    if ratio < 0.3:
        for key, values in model.items():
            model[key] = [(index, -sign, weight) for index, sign, weight in values]
        notes.append(
            f"抬升模型与 world z 的符号一致率仅 {ratio:.0%}，判定为深度轴镜像，已整体取反"
        )
        return model
    if ratio < 0.55:
        notes.append(
            f"抬升模型与 world z 的符号一致率 {ratio:.0%} 接近随机，判定无效，退回时序一致性模式"
        )
        return {}
    notes.append(f"抬升模型与 world z 的符号一致率 {ratio:.0%}，自校准通过")
    return model


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def estimate_depth_signs(
    frames: Sequence[PoseFrame],
    config: DepthSignConfig | None = None,
    *,
    lengths: BoneLengths | None = None,
    aspect: float = 9 / 16.0,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[SignMap, DepthSignReport]:
    """生成 ``{帧号: {骨段: (符号, 置信度)}}``。任何异常都退回"不给符号"，绝不中断分析。"""
    config = config or DepthSignConfig()
    report = DepthSignReport(mode=config.mode if config.enabled else "off")
    ordered = sorted(frames, key=lambda item: item.index)
    report.frames = len(ordered)
    if not config.enabled:
        report.notes.append("深度符号先验未启用")
        return {}, report
    if len(ordered) < MIN_SEQUENCE_FRAMES:
        report.notes.append(
            f"有效帧只有 {len(ordered)} 帧（<{MIN_SEQUENCE_FRAMES}），时序证据不足，本次不修正深度方向"
        )
        return {}, report

    started = time.perf_counter()
    has_world = any(len(frame.world_landmarks) >= LANDMARK_COUNT for frame in ordered)
    if not has_world:
        report.notes.append("当前引擎没有输出米制世界坐标，无法判断深度方向")
        return {}, report

    if lengths is None:
        from app.backend.postprocess.ik_fit import calibrate_bone_lengths

        lengths = calibrate_bone_lengths(ordered)

    notes: list[str] = []
    reference = _temporal_signals(ordered, lengths, config, aspect)
    signals = reference
    used_model = False
    if config.mode == "model":
        model_signals, used_frames = _model_signals(ordered, config, aspect, notes)
        if model_signals:
            model_signals = _calibrate_model_signs(model_signals, reference, config, notes)
            if model_signals:
                signals = model_signals
                used_model = True
                report.mode = "model"
                notes.append(f"抬升覆盖 {used_frames} 帧")
        if not used_model:
            report.mode = "temporal"

    if config.swap_left_right and used_model:
        swapped: dict[str, list[tuple[int, int, float]]] = {
            key: list(values) for key, values in signals.items()
        }
        for key, mirror in _MIRROR_KEY.items():
            if key in swapped and mirror in swapped:
                swapped[key], swapped[mirror] = swapped[mirror], swapped[key]
        signals = swapped
        notes.append("已按配置交换抬升模型的左右约定")

    sign_map: SignMap = {}
    flips_before = 0
    flips_after = 0
    disagreements = 0
    raw_signs = {
        key: {frame_index: sign for frame_index, sign, _ in values}
        for key, values in signals.items()
    }
    per_key_total = max(1, len(signals))
    for order, (key, values) in enumerate(sorted(signals.items())):
        if should_cancel is not None and should_cancel():
            from app.backend.core.errors import AnalysisCancelledError

            raise AnalysisCancelledError("后处理已取消")
        if not values:
            continue
        plain = [(frame_index, sign) for frame_index, sign, _ in values]
        flips_before += _count_flips(plain)
        voted = _vote(
            values,
            window=config.window,
            min_agreement=config.min_agreement,
            min_run=config.min_run,
        )
        if not voted:
            continue
        smoothed = sorted(voted.items())
        flips_after += _count_flips([(index, sign) for index, (sign, _) in smoothed])
        observed = raw_signs.get(key) or {}
        for frame_index, (sign, confidence) in voted.items():
            sign_map.setdefault(frame_index, {})[key] = (int(sign), float(confidence))
            if observed.get(frame_index) not in (None, sign):
                disagreements += 1
        if progress is not None:
            progress(order + 1, per_key_total)

    report.applied = bool(sign_map)
    report.segments = sum(len(value) for value in sign_map.values())
    report.flips_before = flips_before
    report.flips_after = flips_after
    report.disagreements = disagreements
    report.milliseconds = round((time.perf_counter() - started) * 1000.0, 3)
    if report.applied:
        notes.append(
            f"深度方向修正：{len(sign_map)} 帧 / {report.segments} 个骨段符号，"
            f"翻转 {flips_before} → {flips_after}，耗时 {report.milliseconds:.0f} ms"
        )
    else:
        notes.append("没有骨段同时满足投影比与一致率条件，本次不修正深度方向")
    report.notes = [*report.notes, *notes]
    return sign_map, report


__all__ = [
    "H36M_TO_NODE",
    "LIFTABLE_KEYS",
    "MIN_SEQUENCE_FRAMES",
    "ONNX_MODEL_FILENAME",
    "DepthSignConfig",
    "DepthSignReport",
    "SignMap",
    "estimate_depth_signs",
    "resolve_model_path",
]
