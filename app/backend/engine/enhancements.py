"""手指 / 面部增强（任务 4.5）。

- 每手 21 点（MediaPipe HandLandmarker 标准顺序），面部只保留驱动
  AppleModel.usda 所需的语义子集（下颌、下巴、鼻、眼睑、虹膜、嘴角、耳），
  避免把 468 点全量塞进导出 JSON。
- 默认开启：``RefinementConfig(hands=True, face=True)``。模型缺失或 mediapipe
  不可用时静默降级，只写 ``notes``，绝不影响主链路（姿态识别）。
- handedness 直接采用 MediaPipe 输出（其定义假设输入为镜像/自拍画面，与本项目
  ``mirroredReference=true`` 的约定一致）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_WRIST,
    RIGHT_WRIST,
    EngineUnavailableError,
    Landmark,
    PoseFrame,
)
from app.backend.engine.mediapipe_engine import _import_mediapipe
from app.backend.engine.models import MEDIAPIPE_MODEL_URLS, download_model, model_path

logger = get_logger("engine.enhancements")

HAND_LANDMARK_COUNT = 21

# 手部腕点与人体腕点的最大容许距离（归一化坐标，≈画面宽度的 12%）。
# 超过这个距离说明这只手不属于当前这个人（或姿态的腕点已经飘了），
# 硬挂上去会让手指关节"长"在别人的手腕上，播放时表现为手指突然乱跳。
HAND_WRIST_MATCH_MAX = 0.12
_SIDE_WRIST: dict[str, tuple[int, ...]] = {
    "left": (LEFT_WRIST,),
    "right": (RIGHT_WRIST,),
    "": (LEFT_WRIST, RIGHT_WRIST),
}

# MediaPipe Face Mesh 语义点。左右按**人物自身**命名（与 ARKit / AppleModel 一致）；
# 索引取自 MediaPipe canonical face model（非镜像输入下的解剖学左右）。
# 478 点（refine_landmarks=True）额外提供虹膜中心 468/473，用于 eyeball 关节。
FACE_SUBSET: dict[str, int] = {
    "forehead": 10,
    "jaw": 175,
    "chin": 152,
    "nose_tip": 1,
    "right_eye_outer": 33,
    "right_eye_inner": 133,
    "right_eye_upper_lid": 159,
    "right_eye_lower_lid": 145,
    "right_iris": 468,
    "left_eye_inner": 362,
    "left_eye_outer": 263,
    "left_eye_upper_lid": 386,
    "left_eye_lower_lid": 374,
    "left_iris": 473,
    "left_mouth": 61,
    "right_mouth": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_ear": 454,
    "right_ear": 234,
}

# 与 FACE_SUBSET 的 key 顺序一致；map_frame 依赖该稳定顺序解析 face 点。
FACE_SUBSET_NAMES: tuple[str, ...] = tuple(FACE_SUBSET)


class RefinementConfig(BaseModel):
    hands: bool = True
    face: bool = True
    max_hands: int = Field(2, ge=1, le=4)
    num_faces: int = Field(1, ge=1, le=2)
    min_detection_confidence: float = Field(0.5, gt=0, lt=1, description="面部检测阈值")
    hand_detection_confidence: float = Field(
        0.4, gt=0, lt=1, description="手部检测阈值（刻意低于姿态：宁可多检，交给后处理去抖）"
    )
    hand_presence_confidence: float = Field(
        0.35, gt=0, lt=1, description="手部存在性阈值；越低越不容易在握拳/遮挡时整只手消失"
    )
    refine_landmarks: bool = Field(True, description="FaceMesh 虹膜精修，提供 468/473 虹膜中心")
    hand_model: str | None = None
    face_model: str | None = None


class RefinementResult(BaseModel):
    hands: list[list[Landmark]] = Field(default_factory=list)
    hand_sides: list[str] = Field(default_factory=list, description="与 hands 对齐：Left/Right")
    face: list[Landmark] | None = None
    face_names: list[str] = Field(default_factory=list)


class MediaPipeRefiner:
    """手 + 脸的关键点补全器（复用 mediapipe tasks，IMAGE 模式即可）。"""

    def __init__(self, config: RefinementConfig, *, settings: Settings | None = None) -> None:
        self.config = config
        self.settings = settings
        self._hand: Any = None
        self._face: Any = None
        self._mp: Any = None
        self._prepared = False
        self.notes: list[str] = []

    @classmethod
    def availability(cls, config: RefinementConfig, settings: Settings | None = None) -> tuple[bool, str]:
        try:
            _import_mediapipe()
        except EngineUnavailableError as exc:
            return False, exc.message
        for filename in cls._filenames(config):
            if filename and not model_path(filename, settings).is_file():
                return False, f"缺少模型 {filename}（可设置 auto_download_models=true 自动下载）"
        return True, ""

    @staticmethod
    def _filenames(config: RefinementConfig) -> tuple[str | None, str | None]:
        return (
            config.hand_model or ("hand_landmarker.task" if config.hands else None),
            config.face_model or ("face_landmarker.task" if config.face else None),
        )

    def _resolve(self, filename: str | None, explicit: str | None) -> str | None:
        if not explicit and not filename:
            return None
        candidate = model_path(explicit or filename or "", self.settings)
        if candidate.is_file():
            return str(candidate)
        url = MEDIAPIPE_MODEL_URLS.get(explicit or filename or "")
        if url and self.settings is not None and self.settings.auto_download_models:
            return str(download_model(url, candidate))
        raise EngineUnavailableError(
            f"缺少增强模型 {candidate.name}",
            details={"expectedPath": str(candidate), "downloadUrl": url},
        )

    def prepare(self) -> None:
        if self._prepared:
            return
        mp, mp_python, mp_vision = _import_mediapipe()
        self._mp = mp

        hand_file = None
        face_file = None
        try:
            if self.config.hands:
                hand_file = self._resolve("hand_landmarker.task", self.config.hand_model)
                base = mp_python.BaseOptions(model_asset_path=hand_file)
                hand_kwargs: dict[str, object] = {
                    "base_options": base,
                    "running_mode": mp_vision.RunningMode.IMAGE,
                    "num_hands": self.config.max_hands,
                    "min_hand_detection_confidence": self.config.hand_detection_confidence,
                    "min_tracking_confidence": self.config.hand_presence_confidence,
                }
                try:
                    # 部分 mediapipe 版本没有 min_hand_presence_confidence；不支持则回退。
                    options = mp_vision.HandLandmarkerOptions(
                        **hand_kwargs,
                        min_hand_presence_confidence=self.config.hand_presence_confidence,
                    )
                except TypeError:
                    options = mp_vision.HandLandmarkerOptions(**hand_kwargs)
                self._hand = mp_vision.HandLandmarker.create_from_options(options)
            if self.config.face:
                face_file = self._resolve("face_landmarker.task", self.config.face_model)
                base = mp_python.BaseOptions(model_asset_path=face_file)
                face_kwargs: dict[str, object] = {
                    "base_options": base,
                    "running_mode": mp_vision.RunningMode.IMAGE,
                    "num_faces": self.config.num_faces,
                    "min_face_detection_confidence": self.config.min_detection_confidence,
                    "output_face_blendshapes": False,
                    "output_facial_transformation_matrixes": False,
                }
                if self.config.refine_landmarks:
                    # 部分 mediapipe 版本没有 refine_landmarks 参数；不支持则回退 468 点。
                    try:
                        options = mp_vision.FaceLandmarkerOptions(
                            **face_kwargs, refine_landmarks=True
                        )
                    except TypeError:
                        self.notes.append("当前 mediapipe 不支持 refine_landmarks，虹膜点将用眼中心回退")
                        options = mp_vision.FaceLandmarkerOptions(**face_kwargs)
                else:
                    options = mp_vision.FaceLandmarkerOptions(**face_kwargs)
                self._face = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - 增强失败必须静默
            self.close()
            self.notes.append(f"增强模块不可用：{exc}")
            logger.warning("手/脸增强初始化失败，已禁用", exc_info=True)
            return
        self._prepared = True
        logger.info("增强模块就绪：hands=%s face=%s", bool(self._hand), bool(self._face))

    def infer(self, image: np.ndarray) -> RefinementResult:
        if not self._prepared or self._mp is None:
            return RefinementResult()
        rgb = np.ascontiguousarray(image[:, :, ::-1]) if image.ndim == 3 else image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = RefinementResult()

        if self._hand is not None:
            try:
                hands = self._hand.detect(mp_image)
                result.hands = [
                    [_to_landmark(item) for item in hand]
                    for hand in (getattr(hands, "hand_landmarks", None) or [])
                ]
                for entry in getattr(hands, "handedness", None) or []:
                    # MediaPipe 返回 List[List[Category]]；个别版本外层带 .categories。
                    categories = getattr(entry, "categories", None) or list(entry or [])
                    result.hand_sides.append(
                        str(getattr(categories[0], "category_name", "")) if categories else ""
                    )
            except Exception:  # noqa: BLE001
                logger.debug("手部推理失败", exc_info=True)

        if self._face is not None:
            try:
                faces = self._face.detect(mp_image)
                raw = getattr(faces, "face_landmarks", None) or []
                if raw:
                    points = raw[0]
                    subset: list[Landmark] = []
                    names: list[str] = []
                    for name, index in FACE_SUBSET.items():
                        if 0 <= index < len(points):
                            subset.append(_to_landmark(points[index]))
                            names.append(name)
                    result.face = subset
                    result.face_names = names
            except Exception:  # noqa: BLE001
                logger.debug("面部推理失败", exc_info=True)
        return result

    def close(self) -> None:
        for name in ("_hand", "_face"):
            module = getattr(self, name, None)
            if module is not None:
                try:
                    module.close()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭增强模块失败", exc_info=True)
            setattr(self, name, None)
        self._prepared = False


def _to_landmark(item: Any, *, default_visibility: float = 1.0) -> Landmark:
    # HandLandmarker / FaceLandmarker 的 NormalizedLandmark 没有 visibility 字段；
    # 检测到的点按"可见"处理，否则下游渲染/质检会把全部手指/面部点当成无效。
    visibility = float(getattr(item, "visibility", 0.0) or 0.0)
    return Landmark(
        x=float(getattr(item, "x", 0.0)),
        y=float(getattr(item, "y", 0.0)),
        z=float(getattr(item, "z", 0.0)),
        visibility=visibility if visibility > 0 else default_visibility,
    )


def apply_refinements(
    frames: list[PoseFrame], result: RefinementResult
) -> list[PoseFrame]:
    """把增强结果挂到对应的人身上。

    手/脸检测是整帧级别的，而姿态可能是多人。旧实现一律挂给 ``frames[0]``，
    多人画面里跟踪到的人往往不是第一个人 —— 结果是把 A 的手指姿态锚在 B 的腕点上，
    播放时手指忽大忽小、忽握忽松。这里改成按"手部腕点 ↔ 人体腕点"的距离分配：

    - 距离最近的一个人承接这只手；超过 ``HAND_WRIST_MATCH_MAX`` 视为"不属于已识别的
      任何人"（多半是画面里没被姿态模型检出的人），直接丢弃而不是硬挂；
    - 画面里只有一个人时保留兜底：即便腕点对不上也挂给他，避免因为姿态腕点抖动而丢手。
    """
    if not frames:
        return frames
    if result.hands:
        _attach_hands(frames, result)
    if result.face:
        frames[0].face = list(result.face)
    return frames


def _attach_hands(frames: list[PoseFrame], result: RefinementResult) -> None:
    buckets: dict[int, tuple[list[list[Landmark]], list[str]]] = {}
    dropped = 0
    for index, hand in enumerate(result.hands):
        if len(hand) < HAND_LANDMARK_COUNT:
            continue
        label = result.hand_sides[index] if index < len(result.hand_sides) else ""
        distances = [_wrist_distance(frame, hand, label) for frame in frames]
        best = min(range(len(frames)), key=lambda position: distances[position])
        if distances[best] > HAND_WRIST_MATCH_MAX and len(frames) > 1:
            dropped += 1
            continue
        hands, sides = buckets.setdefault(best, ([], []))
        hands.append(list(hand))
        sides.append(label)

    for position, (hands, sides) in buckets.items():
        frames[position].hands = hands
        frames[position].hand_sides = sides
    if dropped:
        logger.debug("%s 只手离所有已识别腕点过远，未挂到骨骼上", dropped)


def _wrist_distance(frame: PoseFrame, hand: list[Landmark], label: str) -> float:
    """手部腕点与该人物对应侧（标签缺失时取两侧较小值）腕点的距离。"""
    if not hand or len(frame.landmarks) < LANDMARK_COUNT:
        return float("inf")
    candidates = _SIDE_WRIST.get(label.strip().lower(), _SIDE_WRIST[""])
    best = float("inf")
    for index in candidates:
        point = frame.landmarks[index]
        if point.visibility <= 0.05:
            continue
        best = min(best, math.hypot(hand[0].x - point.x, hand[0].y - point.y))
    return best
