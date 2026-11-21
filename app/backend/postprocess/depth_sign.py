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
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

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

if TYPE_CHECKING:  # 仅类型标注：运行期不引入 swing → depth_sign 的反向依赖
    from app.backend.postprocess.swing import KickPhaseReport

logger = get_logger("postprocess.depth_sign")

# 符号图：``{帧号: {骨段 key: (符号, 置信度[, 方向向量])}}``。ik_fit 只依赖这个结构，
# 不 import 本模块 —— 保持"拟合层不认识先验来源"的解耦。
# 第三项（§4 的方向线段）是可选的：两元组仍然完全合法，旧调用方/单测不用改。
SignEntry = tuple[int, float] | tuple[int, float, np.ndarray]
SignMap = dict[int, dict[str, SignEntry]]

# 少于这么多帧就不给符号：时序投票没有样本，等同于乱猜。
MIN_SEQUENCE_FRAMES = 30

LifterName = Literal["videopose3d", "motionbert", "agformer"]

ONNX_MODEL_FILENAME = "videopose3d-h36m.onnx"

# 抬升器 → 权重文件名。三者输入输出契约相同（``[T,17,2] → [T,17,3]``，H36M 17 点），
# 所以后端实现只有一份，差别只在权重与感受野。
LIFTER_FILENAMES: dict[str, str] = {
    "videopose3d": "videopose3d-h36m.onnx",
    "motionbert": "motionbert-h36m.onnx",
    "agformer": "agformer-h36m.onnx",
}

# 感受野（帧）：不足时按边缘复制填充。VideoPose3D 的 243 帧来自 5 层 conv 的叠加；
# MotionBERT 是 transformer，窗口长度由权重决定，128 是官方 H36M 配置的常用值。
LIFTER_RECEPTIVE_FIELDS: dict[str, int] = {
    "videopose3d": 243,
    "motionbert": 128,
    "agformer": 128,
}

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
    emit_direction: bool = Field(
        True,
        description="§4：向 ik_fit 提供「方向线段」（符号校正后的单位骨段方向）。"
        "ik_fit 的 depth_direction_weight=0 时它不产生任何影响，所以默认开启",
    )
    direction_min_confidence: float = Field(
        0.5, ge=0, le=1, description="低于该置信度的骨段不给方向（支撑腿被折扣后通常就落在这里）"
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
    lifter: str = Field("", description="model 模式实际使用的抬升器（videopose3d / motionbert / agformer）")
    auto_calibrate_ratio: float | None = Field(
        None, ge=0, le=1, description="自校准一致率（0~1）；None = 未做自校准（temporal 模式 / 样本不足）"
    )
    calibration: Literal["inactive", "skipped", "passed", "inverted", "discarded"] = Field(
        "inactive",
        description="自校准结论：passed 通过 / inverted 深度轴镜像已取反 / "
        "discarded 判定无效 / skipped 样本不足或未启用",
    )
    direction_segments: int = Field(
        0, description="§4 方向向量先验生效的 (帧 × 骨段) 数量；0 = 只有符号（hinge）"
    )
    swing_leg: str = Field("none", description="§5.2 摆动腿识别结果（left / right / none）")
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
@runtime_checkable
class LifterBackend(Protocol):
    """抬升器后端契约：任何失败都要能被调用方静默降级，不能抛到分析主链。

    输入输出契约固定为 ``[T,17,2] → [T,17,3]``（H36M 17 点、根相对），因此
    VideoPose3D / MotionBERT / AGFormer 可以互换；``depth_sign`` 只读 Δz 的符号。
    """

    name: str
    receptive_field: int

    def availability(self) -> tuple[bool, str]:
        ...

    def load(self) -> None:
        ...

    def infer(self, sequence: np.ndarray) -> np.ndarray:
        ...

    @property
    def providers(self) -> list[str]:
        ...


class OnnxLifter:
    """单个 ONNX 抬升器的通用实现（各 lifter 只差权重与感受野）。"""

    def __init__(self, name: str, path: str | None, *, receptive_field: int | None = None) -> None:
        self.name = name
        self.path = path
        self.receptive_field = int(
            receptive_field or LIFTER_RECEPTIVE_FIELDS.get(name, 243)
        )
        self._session: object | None = None
        self._input_name = "pose2d"
        self._window: int | None = None
        self._providers: list[str] = []

    # ---- 生命周期 ----
    def availability(self) -> tuple[bool, str]:
        if not self.path:
            return False, f"未找到 {self.name} 权重（{LIFTER_FILENAMES.get(self.name, '')}）"
        try:
            import onnxruntime  # noqa: F401, PLC0415 - 仅探测依赖
        except Exception as exc:  # pragma: no cover - 依赖缺失
            return False, f"onnxruntime 不可用（{exc}）"
        return True, ""

    def load(self) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        from app.backend.engine.acceleration import onnx_execution_providers  # noqa: PLC0415

        self._providers = onnx_execution_providers()
        self._session = ort.InferenceSession(self.path, providers=self._providers)
        session = self._session
        if session is None:  # pragma: no cover - 理论不可达
            raise RuntimeError("会话未创建")
        inputs = session.get_inputs()  # type: ignore[attr-defined]
        self._input_name = inputs[0].name
        shape = list(inputs[0].shape)
        # 固定长度输入（部分导出把 T 烘死）→ 按窗口切分；动态维度则整段一次推理
        length = shape[1] if len(shape) > 1 else None
        self._window = int(length) if isinstance(length, int) and length > 0 else None

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    # ---- 推理 ----
    def _run(self, sequence: np.ndarray) -> np.ndarray:
        session = self._session
        if session is None:
            raise RuntimeError("抬升器未 load()")
        output = session.run(None, {self._input_name: sequence[None, ...]})[0]  # type: ignore[attr-defined]
        if output.ndim == 4:
            output = output[0]
        if output.ndim != 3 or output.shape[1] != 17 or output.shape[2] != 3:
            raise ValueError(f"抬升器输出形状不符：{tuple(output.shape)}，期望 [T,17,3]")
        if not np.all(np.isfinite(output)):
            raise ValueError("抬升器输出含 NaN/Inf")
        return np.asarray(output, dtype=np.float64)

    def infer(self, sequence: np.ndarray) -> np.ndarray:
        """``[T,17,2] → [T,17,3]``。不足感受野时边缘复制填充，固定窗口模型按窗切分。"""
        count = int(sequence.shape[0])
        if count == 0:
            return np.zeros((0, 17, 3), dtype=np.float64)
        window = self._window or max(1, self.receptive_field)
        if self._window is None and count <= self.receptive_field:
            pad = self.receptive_field - count
            padded = np.pad(sequence, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode="edge")
            return self._run(padded)[pad // 2 : pad // 2 + count]

        stride = max(1, window // 2)
        outputs = np.zeros((count, 17, 3), dtype=np.float64)
        weights = np.zeros((count, 1, 1), dtype=np.float64)
        start = 0
        while start < count:
            end = min(count, start + window)
            chunk = sequence[start:end]
            pad = window - chunk.shape[0]
            if pad:
                # 最后一窗不足时边缘复制（比零填充更接近训练分布）
                chunk = np.pad(chunk, ((0, pad), (0, 0), (0, 0)), mode="edge")
            result = self._run(chunk)[: end - start]
            outputs[start:end] += result
            weights[start:end] += 1.0
            if end >= count:
                break
            start += stride
        return outputs / np.maximum(weights, 1e-9)


def resolve_lifter_name(config: DepthSignConfig | None = None) -> str:
    """当前抬升器名：环境变量/配置（``DETECT_BODY_LIFTER``）优先，缺配置时用 videopose3d。"""
    try:
        from app.backend.core.settings import get_settings

        return str(get_settings().lifter)
    except Exception:  # pragma: no cover - 配置不可用
        return "videopose3d"


def resolve_model_path(config: DepthSignConfig, lifter: str | None = None) -> str | None:
    """显式路径 → ``models_dir`` → 打包资源根 ``models/``。找不到返回 None。

    查找顺序对所有 lifter 一致，只是文件名随 ``DETECT_BODY_LIFTER`` 变化
    （``videopose3d-h36m.onnx`` / ``motionbert-h36m.onnx`` / ``agformer-h36m.onnx``）。
    """
    from pathlib import Path

    if config.model_path:
        path = Path(config.model_path)
        return str(path) if path.is_file() else None
    name = lifter or resolve_lifter_name(config)
    filename = LIFTER_FILENAMES.get(name, ONNX_MODEL_FILENAME)
    try:
        from app.backend.core.settings import get_settings

        settings = get_settings()
    except Exception:  # pragma: no cover - 配置不可用的极端情况
        return None
    candidates = [
        Path(settings.models_dir) / filename,
        Path(settings.static_dir).parents[2] / "models" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def lifter_catalog(config: DepthSignConfig | None = None) -> list[dict[str, object]]:
    """可选的抬升器与权重就绪情况（供 UI 把 ``model`` 档置灰）。"""
    config = config or DepthSignConfig()
    entries: list[dict[str, object]] = []
    for name, filename in LIFTER_FILENAMES.items():
        path = resolve_model_path(config, name)
        entries.append(
            {
                "name": name,
                "filename": filename,
                "present": path is not None,
                "path": path or "",
                "receptiveField": LIFTER_RECEPTIVE_FIELDS.get(name, 243),
            }
        )
    return entries


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


def build_lifter_backend(
    config: DepthSignConfig, *, lifter: str | None = None
) -> LifterBackend:
    """按 ``DETECT_BODY_LIFTER`` 构造抬升器后端（权重缺失时也返回对象，由调用方降级）。"""
    name = lifter or resolve_lifter_name(config)
    return OnnxLifter(name, resolve_model_path(config, name))


def _model_signals(
    frames: Sequence[PoseFrame],
    config: DepthSignConfig,
    aspect: float,
    notes: list[str],
    *,
    backend: LifterBackend | None = None,
) -> tuple[dict[str, list[tuple[int, int, float]]], int]:
    """跑 ONNX 抬升器，返回符号信号与参与帧数；任何失败都返回空并写 note。"""
    lifter = backend or build_lifter_backend(config)
    available, reason = lifter.availability()
    if not available:
        notes.append(f"{reason}，本次退回时序一致性模式；可用 scripts/export_*_onnx.py 离线导出权重")
        return {}, 0

    sequence, frame_indices, confidences = _h36m_input(frames, aspect)
    if sequence.shape[0] == 0:
        return {}, 0

    try:
        lifter.load()
        output = lifter.infer(sequence)
    except Exception as exc:  # pragma: no cover - 权重损坏 / 形状不符 / EP 异常
        notes.append(f"抬升模型推理失败（{exc}），退回时序一致性模式")
        return {}, 0

    notes.append(
        f"抬升模型：{lifter.name}（{','.join(lifter.providers) or 'CPU'}，"
        f"感受野 {lifter.receptive_field} 帧），只取深度符号，幅度与尺度一律丢弃"
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
) -> tuple[dict[str, list[tuple[int, int, float]]], float | None, str]:
    """用同段视频的 world z 符号做一致性校准（自动判镜像 / 判无效）。

    返回 ``(符号, 一致率, 结论)``；一致率与结论同时进 ``DepthSignReport``，
    这样消费方（UI / 基准脚本）不必去解析 ``notes`` 里的中文句子。
    """
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
        notes.append(f"自校准样本不足（{total} < 12），本次不校核方向")
        return model, None, "skipped"
    ratio = agree / total
    if not config.auto_calibrate:
        if config.depth_axis_inverted:
            for key, values in model.items():
                model[key] = [(index, -sign, weight) for index, sign, weight in values]
        notes.append(f"未启用自校准：模型与 world z 的符号一致率 {ratio:.0%}")
        return model, round(ratio, 4), "skipped"
    if ratio < 0.3:
        for key, values in model.items():
            model[key] = [(index, -sign, weight) for index, sign, weight in values]
        notes.append(
            f"抬升模型与 world z 的符号一致率仅 {ratio:.0%}，判定为深度轴镜像，已整体取反"
        )
        logger.debug("自校准：一致率 %.3f → 整体取反", ratio)
        return model, round(ratio, 4), "inverted"
    if ratio < 0.55:
        notes.append(
            f"抬升模型与 world z 的符号一致率 {ratio:.0%} 接近随机，判定无效，退回时序一致性模式"
        )
        logger.debug("自校准：一致率 %.3f → 判定无效", ratio)
        return {}, round(ratio, 4), "discarded"
    notes.append(f"抬升模型与 world z 的符号一致率 {ratio:.0%}，自校准通过")
    logger.debug("自校准：一致率 %.3f → 通过", ratio)
    return model, round(ratio, 4), "passed"


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
LEG_BONE_SUFFIXES: tuple[str, ...] = ("knee", "ankle", "heel", "foot_index")

# 支撑腿的符号置信度折扣（§5.2）：站直的腿 Δz≈0，"往前还是往后"基本是噪声，
# 强推它只会把腿掰弯。摆动腿保持原值，支撑腿按这个比例削弱。
SUPPORT_LEG_CONFIDENCE_SCALE = 0.45


def _sign_confidence(
    key: str,
    frame_index: int,
    confidence: float,
    swing_leg: str,
    phases: KickPhaseReport | None,
) -> float:
    """按"摆动腿 / 相位"给符号置信度加权（§5.2 / §5.3）。

    只影响**先验的力度**，不影响符号本身；关掉摆动腿识别时这里恒为 1.0，
    因此与改造前的行为逐位一致。
    """
    scale = 1.0
    if swing_leg in ("left", "right") and key.endswith(LEG_BONE_SUFFIXES):
        side = key.split("_", 1)[0]
        if side in ("left", "right") and side != swing_leg:
            scale *= SUPPORT_LEG_CONFIDENCE_SCALE
    if phases is not None and getattr(phases, "detected", False):
        from app.backend.postprocess.swing import direction_weight_scale

        scale *= direction_weight_scale(frame_index, phases)
    return min(1.0, max(0.0, confidence * scale))


def _bone_direction(
    positions: dict[str, np.ndarray] | None, key: str, sign: int
) -> np.ndarray | None:
    """符号校正后的单位骨段方向（"方向线段"）：先按先验定住 z 的正负，再归一化。"""
    if not positions:
        return None
    node = node_map().get(key)
    if node is None or not node.parent:
        return None
    child = positions.get(key)
    parent = positions.get(node.parent)
    if child is None or parent is None:
        return None
    vector = np.array(child, dtype=np.float64) - np.array(parent, dtype=np.float64)
    if sign and float(vector[2]) * sign < 0:
        vector[2] = -vector[2]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return vector / norm


def _smooth_directions(
    directions: dict[int, np.ndarray], window: int
) -> dict[int, np.ndarray]:
    """对方向线段做时间窗平均（向量平均后重新归一化）。

    只改方向、不改长度，也不改变符号先验：符号仍由 ``_vote`` 的多数投票决定。
    """
    if len(directions) < 2 or window <= 1:
        return directions
    indices = sorted(directions)
    half = max(1, int(window) // 2)
    smoothed: dict[int, np.ndarray] = {}
    for position, frame_index in enumerate(indices):
        start = max(0, position - half)
        end = min(len(indices), position + half + 1)
        total = np.zeros(3, dtype=np.float64)
        for neighbor in indices[start:end]:
            total += directions[neighbor]
        norm = float(np.linalg.norm(total))
        if norm > 1e-9:
            smoothed[frame_index] = total / norm
    return smoothed


def estimate_depth_signs(
    frames: Sequence[PoseFrame],
    config: DepthSignConfig | None = None,
    *,
    lengths: BoneLengths | None = None,
    aspect: float = 9 / 16.0,
    swing_leg: str = "none",
    phases: KickPhaseReport | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[SignMap, DepthSignReport]:
    """生成 ``{帧号: {骨段: (符号, 置信度)}}``。任何异常都退回"不给符号"，绝不中断分析。"""
    config = config or DepthSignConfig()
    report = DepthSignReport(mode=config.mode if config.enabled else "off")
    report.swing_leg = swing_leg or "none"
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
        backend = build_lifter_backend(config)
        report.lifter = backend.name
        model_signals, used_frames = _model_signals(
            ordered, config, aspect, notes, backend=backend
        )
        if model_signals:
            model_signals, ratio, verdict = _calibrate_model_signs(
                model_signals, reference, config, notes
            )
            report.auto_calibrate_ratio = ratio
            report.calibration = verdict  # type: ignore[assignment]
            if model_signals:
                signals = model_signals
                used_model = True
                report.mode = "model"
                notes.append(f"抬升覆盖 {used_frames} 帧")
        if not used_model:
            # mode 如实回传 temporal；lifter 保留"尝试过哪个"，便于排查是权重缺失还是判定无效
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
    direction_segments = 0
    # 方向线段只在开启时才取一遍世界坐标（关闭时这一步是纯开销）
    positions_by_frame: dict[int, dict[str, np.ndarray]] = (
        {
            frame.index: node_positions(frame.world_landmarks)
            for frame in ordered
            if frame.valid and len(frame.world_landmarks) >= LANDMARK_COUNT
        }
        if config.emit_direction
        else {}
    )
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
            # §8.3：丢弃必须可诊断 —— 否则"某根骨段没被修正"只能靠猜
            logger.debug(
                "骨段 %s 被丢弃：%d 帧样本的一直率不足以通过 min_agreement=%.2f / min_run=%d",
                key,
                len(values),
                config.min_agreement,
                config.min_run,
            )
            continue
        smoothed = sorted(voted.items())
        flips_after += _count_flips([(index, sign) for index, (sign, _) in smoothed])
        if logger.isEnabledFor(10):  # DEBUG
            agree = np.mean([confidence for _sign, confidence in smoothed])
            flips_key = _count_flips([(index, sign) for index, (sign, _) in smoothed])
            drop = len(values) - len(smoothed)
            logger.debug(
                "骨段 %s：%d/%d 帧通过投票（平均一致率 %.2f，翻转 %d，丢弃 %d）",
                key,
                len(smoothed),
                len(values),
                float(agree),
                flips_key,
                drop,
            )
        observed = raw_signs.get(key) or {}
        # 方向线段是**时序**概念：逐帧把观测方向当目标，会把"朝前"抖成"朝侧"。
        # 所以先按帧取符号校正后的单位方向，再在同一时间窗内做平均（分母 1 归一化）。
        directions: dict[int, np.ndarray] = {}
        if config.emit_direction:
            for frame_index, (sign, confidence) in voted.items():
                weight = _sign_confidence(key, frame_index, confidence, swing_leg, phases)
                if weight < config.direction_min_confidence:
                    continue
                vector = _bone_direction(positions_by_frame.get(frame_index), key, int(sign))
                if vector is not None:
                    directions[frame_index] = vector
            directions = _smooth_directions(directions, config.window)
        for frame_index, (sign, confidence) in voted.items():
            weight = _sign_confidence(key, frame_index, confidence, swing_leg, phases)
            entry: tuple[object, ...] = (int(sign), float(weight))
            direction = directions.get(frame_index)
            if direction is not None:
                entry = (int(sign), float(weight), direction)
                direction_segments += 1
            sign_map.setdefault(frame_index, {})[key] = entry  # type: ignore[index]
            if observed.get(frame_index) not in (None, sign):
                disagreements += 1
        if progress is not None:
            progress(order + 1, per_key_total)

    report.applied = bool(sign_map)
    report.direction_segments = direction_segments
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
    "LIFTER_FILENAMES",
    "LIFTER_RECEPTIVE_FIELDS",
    "MIN_SEQUENCE_FRAMES",
    "ONNX_MODEL_FILENAME",
    "DepthSignConfig",
    "DepthSignReport",
    "LifterBackend",
    "LifterName",
    "OnnxLifter",
    "SignMap",
    "build_lifter_backend",
    "estimate_depth_signs",
    "lifter_catalog",
    "resolve_lifter_name",
    "resolve_model_path",
]
