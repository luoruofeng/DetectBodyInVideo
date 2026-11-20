"""推理引擎抽象层（任务 4.1）。

约定：
- 统一数据结构 ``PoseFrame``：``index`` / ``timestamp_ms`` / ``landmarks[33]`` /
  ``world_landmarks[33]`` / ``bbox`` / ``score``，多人时每帧每人一个 ``PoseFrame``。
- ``landmarks`` 为归一化坐标（相对图像宽高，``y`` 向下，``z`` 与 ``x`` 同量级）；
  ``world_landmarks`` 为米制（原点≈两髋中点，``y`` 向下）。
- 引擎只负责"单帧 → 骨骼"，不碰视频 IO、不平滑、不做坐标翻转。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.errors import AppError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings

logger = get_logger("engine.base")

LANDMARK_COUNT = 33

# MediaPipe Pose 关键点顺序（BlazePose 33 点），全项目统一使用该索引
LANDMARK_NAMES: tuple[str, ...] = (
    "nose",  # 0
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",  # 11
    "right_shoulder",  # 12
    "left_elbow",  # 13
    "right_elbow",  # 14
    "left_wrist",  # 15
    "right_wrist",  # 16
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",  # 23
    "right_hip",  # 24
    "left_knee",  # 25
    "right_knee",  # 26
    "left_ankle",  # 27
    "right_ankle",  # 28
    "left_heel",  # 29
    "right_heel",  # 30
    "left_foot_index",  # 31
    "right_foot_index",  # 32
)

LEFT_INDICES: tuple[int, ...] = (1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
RIGHT_INDICES: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


class EngineError(AppError):
    """引擎运行期失败（推理异常、模型损坏等）。"""

    status_code = 500
    code = "engine_error"


class EngineUnavailableError(AppError):
    """引擎不可用（缺依赖 / 缺模型 / EP 不可用），调用方应降级到其它引擎。"""

    status_code = 503
    code = "engine_unavailable"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
class Landmark(BaseModel):
    """单个关键点。``visibility`` 为置信度，``presence`` 仅部分引擎提供。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0
    presence: float | None = None

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.visibility)

    def lerp(self, other: Landmark, t: float) -> Landmark:
        t = min(1.0, max(0.0, t))
        return Landmark(
            x=self.x + (other.x - self.x) * t,
            y=self.y + (other.y - self.y) * t,
            z=self.z + (other.z - self.z) * t,
            visibility=self.visibility + (other.visibility - self.visibility) * t,
            presence=None
            if self.presence is None or other.presence is None
            else self.presence + (other.presence - self.presence) * t,
        )


class BoundingBox(BaseModel):
    """轴对齐包围盒，单位为归一化坐标（相对图像宽高）。"""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def iou(self, other: BoundingBox) -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.area + other.area - inter
        return float(inter / union) if union > 1e-9 else 0.0

    def expanded(self, factor: float = 1.25, *, clip: bool = True) -> BoundingBox:
        """按比例外扩（RTMPose 需要含完整人体的上下文）。"""
        cx, cy = self.center
        half_w = self.width * factor / 2.0
        half_h = self.height * factor / 2.0
        x, y = cx - half_w, cy - half_h
        width, height = half_w * 2.0, half_h * 2.0
        if clip:
            x = max(0.0, x)
            y = max(0.0, y)
            width = min(1.0 - x, width)
            height = min(1.0 - y, height)
        return BoundingBox(x=x, y=y, width=max(0.0, width), height=max(0.0, height))

    @classmethod
    def from_points(
        cls, points: Sequence[tuple[float, float]], *, pad: float = 0.0
    ) -> BoundingBox | None:
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = min(xs) - pad, min(ys) - pad
        return cls(
            x=x,
            y=y,
            width=max(0.0, max(xs) + pad - x),
            height=max(0.0, max(ys) + pad - y),
        )


class PoseFrame(BaseModel):
    """一帧（或一帧中的一个人）的姿态结果。"""

    index: int = 0
    timestamp_ms: float = 0.0
    landmarks: list[Landmark] = Field(default_factory=list, description="归一化坐标 33 点")
    world_landmarks: list[Landmark] = Field(
        default_factory=list, description="米制世界坐标 33 点，引擎不支持时为空"
    )
    bbox: BoundingBox | None = None
    score: float = 0.0
    valid: bool = True
    interpolated: bool = False
    track_id: int | None = None
    hands: list[list[Landmark]] | None = Field(default=None, description="每手 21 点（可选增强）")
    hand_sides: list[str] = Field(default_factory=list, description="与 hands 对齐的 Left/Right")
    face: list[Landmark] | None = Field(default=None, description="精简面部点（可选增强）")

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0

    def landmark(self, index: int) -> Landmark | None:
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def mean_visibility(self, indices: Iterable[int] | None = None) -> float:
        pool = list(indices) if indices is not None else range(len(self.landmarks))
        values = [
            self.landmarks[i].visibility
            for i in pool
            if 0 <= i < len(self.landmarks) and self.landmarks[i].visibility > 0
        ]
        return float(sum(values) / len(values)) if values else 0.0

    def as_array(self, *, use_world: bool = False) -> np.ndarray:
        """转为 ``(33, 4)`` 的 ``[x, y, z, visibility]`` 数组，缺失点补 0。"""
        source = self.world_landmarks if use_world and self.world_landmarks else self.landmarks
        array = np.zeros((LANDMARK_COUNT, 4), dtype=np.float32)
        for i, item in enumerate(source[:LANDMARK_COUNT]):
            array[i] = (item.x, item.y, item.z, item.visibility)
        return array

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        index: int = 0,
        timestamp_ms: float = 0.0,
        score: float = 0.0,
        bbox: BoundingBox | None = None,
        valid: bool = True,
        track_id: int | None = None,
    ) -> PoseFrame:
        points = [
            Landmark(
                x=float(array[i, 0]),
                y=float(array[i, 1]),
                z=float(array[i, 2]),
                visibility=float(array[i, 3]) if array.shape[1] > 3 else 1.0,
            )
            for i in range(min(LANDMARK_COUNT, array.shape[0]))
        ]
        return cls(
            index=index,
            timestamp_ms=timestamp_ms,
            landmarks=points,
            bbox=bbox,
            score=score,
            valid=valid,
            track_id=track_id,
        )

    def bbox_from_landmarks(self, *, min_visibility: float = 0.3) -> BoundingBox | None:
        points = [
            (item.x, item.y)
            for item in self.landmarks
            if item.visibility >= min_visibility
        ]
        return BoundingBox.from_points(points)


class EngineCapabilities(BaseModel):
    name: str
    display_name: str
    multi_person: bool = False
    world_landmarks: bool = False
    hands: bool = False
    face: bool = False
    supports_batching: bool = Field(False, description="是否真正支持批量推理（任务 9.1）")
    providers: list[str] = Field(default_factory=list)
    model_tier: str = "full"
    device: str | None = None
    notes: list[str] = Field(default_factory=list)


class EngineSelection(BaseModel):
    """引擎选型结果：最终用谁、为什么降级，全部可回传给 UI。"""

    engine: str
    display_name: str
    requested: str = "auto"
    degraded: bool = False
    reasons: list[str] = Field(default_factory=list)
    capabilities: EngineCapabilities
    model_tier: str = "full"


# --------------------------------------------------------------------------- #
# 引擎接口
# --------------------------------------------------------------------------- #
class PoseEngine(ABC):
    """所有姿态引擎的统一接口。子类必须可"构造 → prepare → infer → close"。"""

    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base Engine"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        plan: object | None = None,
        model_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.plan = plan
        self.model_path = model_path
        self.runtime: object | None = None
        self._prepared = False
        self._notes: list[str] = []

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def notes(self) -> list[str]:
        return list(self._notes)

    @classmethod
    def availability(cls, settings: Settings | None = None) -> tuple[bool, str]:
        """返回 ``(是否可用, 原因)``；不抛异常，供选型阶段静默降级。"""
        return True, ""

    def apply_runtime(self, config: object | None) -> list[str]:
        """按 ``RuntimeConfig`` 调优（线程 / FP16 / 批处理）。

        必须在 ``prepare()`` 之前调用；返回给用户看的说明（降级原因等）。
        """
        self.runtime = config
        return []

    @abstractmethod
    def prepare(self, device: str | None = None) -> None:
        """加载模型、创建会话。失败抛 ``EngineUnavailableError``。"""

    @abstractmethod
    def infer(
        self, image: np.ndarray, *, index: int = 0, timestamp_ms: float = 0.0
    ) -> list[PoseFrame]:
        """单帧推理。返回 0..N 个 ``PoseFrame``（0 = 本帧未检测到人）。"""

    def infer_batch(
        self,
        frames: Iterable[tuple[int, float, np.ndarray]],
    ) -> list[list[PoseFrame]]:
        """默认逐帧调用；支持批处理的子类（ONNX）可覆写。"""
        return [
            self.infer(image, index=index, timestamp_ms=timestamp)
            for index, timestamp, image in frames
        ]

    @abstractmethod
    def close(self) -> None:
        """释放模型与会话，可重复调用。"""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        ...

    def __enter__(self) -> PoseEngine:
        self.prepare()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def engine_tier(settings: Settings | None, plan: object | None) -> str:
    """按 ``AccelerationPlan.model_tier``（优先）或配置决定模型档位。"""
    tier = getattr(plan, "model_tier", None)
    if isinstance(tier, str) and tier in {"lite", "full", "heavy"}:
        return tier
    if settings is not None and settings.prefer_model_tier in {"lite", "full", "heavy"}:
        return settings.prefer_model_tier
    return "full"


def providers_from_plan(plan: object | None) -> list[str]:
    """把 ``AccelerationPlan`` 的 EP 链转成 onnxruntime 期望的列表。"""
    providers = getattr(plan, "execution_providers", None)
    if isinstance(providers, (list, tuple)) and providers:
        return [str(item) for item in providers]
    return ["CPUExecutionProvider"]


EngineKind = Literal["mediapipe", "onnx", "auto"]
