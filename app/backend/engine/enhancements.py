"""手指 / 面部增强（任务 4.5）。

- 每手 21 点（MediaPipe HandLandmarker 标准顺序），面部只保留驱动
  AppleModel.usda 所需的语义子集（下颌、下巴、鼻、眼睑、虹膜、嘴角、耳），
  避免把 468 点全量塞进导出 JSON。
- 默认开启：``RefinementConfig(hands=True, face=True)``。模型缺失或 mediapipe
  不可用时静默降级，只写 ``notes``，绝不影响主链路（姿态识别）。
- handedness 直接采用 MediaPipe 输出（其定义假设输入为镜像/自拍画面，与本项目
  ``mirroredReference=true`` 的约定一致）。

**手腕 ROI 裁剪检测（默认开启）**：整帧是"远景全身"最常见的输入——人物站姿、
双手在身侧或胸前，1080p 下一只手只有 40~60 px。HandLandmarker 的掌部检测器在
这个尺度上极不稳定（实测整帧检出率约 35%），而 21 点缺失就等于白膜手指整段
不动。因此优先按姿态腕点/肘点裁一块「手部 ROI」（尺寸由前臂投影长度估出的手长
决定，必要时放大到 ≈256 px）再检测，检出率实测可到 98%。裁剪框由姿态给出，所以
左右手归属可以直接由 ROI 决定，不再依赖会翻转的 handedness 分类。

ROI 用 IMAGE 模式逐帧独立检测，不用 VIDEO 模式：ROI 每帧跟着腕点移动，跨帧
"跟踪"会把上一帧的手型带到新的裁剪框里，实测指关节抖动反而放大 6 倍。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    EngineUnavailableError,
    Landmark,
    PoseFrame,
)
from app.backend.engine.mediapipe_engine import _import_mediapipe
from app.backend.engine.models import MEDIAPIPE_MODEL_URLS, download_model, model_path

logger = get_logger("engine.enhancements")

HAND_LANDMARK_COUNT = 21

# --------------------------------------------------------------------------- #
# 手腕 ROI 参数
# --------------------------------------------------------------------------- #
# 手长（腕→中指尖）约等于前臂长（肘→腕）的 0.75 倍，用于从姿态估出 ROI 尺寸。
_ROI_HAND_LENGTH_RATIO = 0.75
# ROI 中心从腕点沿前臂方向外移 0.45 倍手长：手掌在腕点之前，这样五指不会顶到框边。
_ROI_CENTER_BIAS = 0.45
# 放大上限：ROI 太小的时候再放大也没有新信息，只会浪费算力。
_ROI_MAX_UPSCALE = 4.0
# 检出腕点相对 ROI 边长的最大偏移；超过说明这只手不是该侧腕点附近的手。
_ROI_WRIST_MATCH_MAX = 0.4
# 单块 ROI 边长不超过画面短边的这个比例；再大就等于整帧检测，白白多跑一次推理。
_ROI_MAX_FRAME_SHARE = 0.6
# 姿态腕点低于该可见度就不裁 ROI（裁出来多半是躯干，只会造成误检）。
_ROI_WRIST_MIN_VISIBILITY = 0.25

_ROI_SIDES: tuple[tuple[str, int, int], ...] = (
    ("Left", LEFT_WRIST, LEFT_ELBOW),
    ("Right", RIGHT_WRIST, RIGHT_ELBOW),
)

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
        0.3,
        gt=0,
        lt=1,
        description="手部检测阈值（刻意低于姿态：握拳正对镜头时掌部得分常在 0.2~0.4，宁可多检交给后处理去抖）",
    )
    hand_presence_confidence: float = Field(
        0.25, gt=0, lt=1, description="手部存在性阈值；越低越不容易在握拳/遮挡时整只手消失"
    )
    refine_landmarks: bool = Field(True, description="FaceMesh 虹膜精修，提供 468/473 虹膜中心")
    hand_model: str | None = None
    face_model: str | None = None
    hand_roi: bool = Field(
        True,
        description="按姿态腕点裁 ROI 再检测手；远景全身视频整帧基本检不到手，关掉会退回整帧检测",
    )
    hand_roi_scales: tuple[float, ...] = Field(
        default=(2.6, 4.5, 1.3),
        description="ROI 半宽 = 系数 × 估计手长，按顺序尝试、命中即停；握拳正对镜头需要更大的框",
    )
    hand_roi_min_box_px: int = Field(64, ge=24, le=480, description="ROI 边长下限（像素，防手长估计过小）")
    hand_roi_target_px: int = Field(256, ge=96, le=768, description="ROI 放大后的目标边长（像素）")
    hand_roi_wrist_visibility: float = Field(
        _ROI_WRIST_MIN_VISIBILITY,
        gt=0.0,
        le=1.0,
        description="姿态腕点可见度低于该值时不裁 ROI（只见半个身体时避免误检）",
    )
    hand_roi_full_fallback: bool = Field(
        True, description="ROI 一只都没检到时，再整帧检测一次作为兜底"
    )


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
        # 逐帧统计：手部检出率是"手指动不动"的唯一直接指标，写进 notes 便于排查。
        self.stats: dict[str, int] = {
            "roiHands": 0,  # 尝试定位的手次数（左右腕各算一次）
            "roiHits": 0,
            "roiCrops": 0,  # 实际裁剪推理次数（尺度阶梯会大于手次数）
            "fullFallbacks": 0,
            "fullHits": 0,
            "wristSkipped": 0,
        }

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

    def infer(
        self, image: np.ndarray, pose_frames: Sequence[PoseFrame] | None = None
    ) -> RefinementResult:
        """``pose_frames`` 用于手腕 ROI 定位；不传则退回整帧检测（旧行为）。"""
        if not self._prepared or self._mp is None:
            return RefinementResult()
        result = RefinementResult()

        if self._hand is not None:
            try:
                result.hands, result.hand_sides = self._detect_hands(image, pose_frames)
            except Exception:  # noqa: BLE001
                logger.debug("手部推理失败", exc_info=True)

        if self._face is not None:
            try:
                mp_image = self._to_mp_image(image)
                faces = self._face.detect(mp_image) if mp_image is not None else None
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

    # ------------------------------------------------------------------ #
    # 手部：手腕 ROI 优先，整帧兜底
    # ------------------------------------------------------------------ #
    def _detect_hands(
        self, image: np.ndarray, pose_frames: Sequence[PoseFrame] | None
    ) -> tuple[list[list[Landmark]], list[str]]:
        hands: list[list[Landmark]] = []
        sides: list[str] = []
        if self.config.hand_roi and pose_frames:
            for hand, side in self._iter_roi_hands(image, pose_frames):
                if len(hands) >= self.config.max_hands:
                    break
                hands.append(hand)
                sides.append(side)
        if not hands and self.config.hand_roi_full_fallback:
            self.stats["fullFallbacks"] += 1
            hands, sides = self._detect_hands_full(image)
        return hands[: self.config.max_hands], sides[: self.config.max_hands]

    def _detect_hands_full(self, image: np.ndarray) -> tuple[list[list[Landmark]], list[str]]:
        mp_image = self._to_mp_image(image)
        if mp_image is None:
            return [], []
        detected = self._hand.detect(mp_image)
        hands = [
            [_to_landmark(item) for item in hand]
            for hand in (getattr(detected, "hand_landmarks", None) or [])
        ]
        sides: list[str] = []
        for entry in getattr(detected, "handedness", None) or []:
            # MediaPipe 返回 List[List[Category]]；个别版本外层带 .categories。
            categories = getattr(entry, "categories", None) or list(entry or [])
            sides.append(str(getattr(categories[0], "category_name", "")) if categories else "")
        self.stats["fullHits"] += len(hands)
        return hands, sides

    def _iter_roi_hands(
        self, image: np.ndarray, pose_frames: Sequence[PoseFrame]
    ) -> Iterator[tuple[list[Landmark], str]]:
        """按每个已识别的人的左右腕点各裁 ROI 检测；产出 (21 点, Left/Right)。

        ROI 尺寸从小到大试（``hand_roi_scales``）：小框精度最高、贴得最紧，但当手是
        **握拳正对镜头**（看不到掌面与五指轮廓，本项目的主场景）时掌部检测器常常整块
        漏检；放大到包含前臂与小臂一半的上下文后召回明显回升。命中即停，所以平均
        每只手只多跑零点几次推理。
        """
        height, width = image.shape[:2]
        for frame in pose_frames:
            if len(frame.landmarks) < LANDMARK_COUNT:
                continue
            for side, wrist_index, elbow_index in _ROI_SIDES:
                wrist = frame.landmarks[wrist_index]
                if wrist.visibility < self.config.hand_roi_wrist_visibility:
                    self.stats["wristSkipped"] += 1
                    continue
                self.stats["roiHands"] += 1
                found = self._detect_hand_roi(
                    image,
                    wrist,
                    frame.landmarks[elbow_index],
                    width,
                    height,
                    side,
                )
                if found is not None:
                    yield found, side

    def _detect_hand_roi(
        self,
        image: np.ndarray,
        wrist: Landmark,
        elbow: Landmark,
        width: int,
        height: int,
        side: str,
    ) -> list[Landmark] | None:
        """按尺度阶梯在腕点附近找这只手；找到就返回整帧归一化后的 21 点。"""
        boxes = [
            box
            for box in (
                _hand_roi_box(
                    wrist,
                    elbow,
                    width,
                    height,
                    scale=float(scale),
                    min_box_px=self.config.hand_roi_min_box_px,
                    side=side,
                )
                for scale in self.config.hand_roi_scales
            )
            if box is not None
        ]
        if not boxes:
            return None
        # 近景素材（手本身就很大）下大尺度的框会铺满整幅画面，退化成"整帧检测"还多花算力：
        # 这类框只在没有更小的框可选时才用。
        limit = _ROI_MAX_FRAME_SHARE * min(width, height)
        usable = [box for box in boxes if max(box.width, box.height) <= limit]
        for box in usable or [min(boxes, key=lambda item: max(item.width, item.height))]:
            crop = image[box.y0 : box.y0 + box.height, box.x0 : box.x0 + box.width]
            if crop.size == 0:
                continue
            factor = min(
                max(self.config.hand_roi_target_px / max(box.height, box.width), 1.0),
                _ROI_MAX_UPSCALE,
            )
            if factor > 1.0:
                # 小目标放大后再检测：掌部检测器的尺度金字塔对小手的召回明显更好。
                crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            mp_image = self._to_mp_image(crop)
            if mp_image is None:
                continue
            self.stats["roiCrops"] += 1
            detected = self._hand.detect(mp_image)
            best = _closest_hand(getattr(detected, "hand_landmarks", None) or [], box)
            if best is None:
                continue
            self.stats["roiHits"] += 1
            return [_roi_to_landmark(item, box, width, height) for item in best]
        return None

    def _to_mp_image(self, image: np.ndarray) -> Any:
        if not self._prepared or self._mp is None or image.size == 0:
            return None
        rgb = np.ascontiguousarray(image[:, :, ::-1]) if image.ndim == 3 else image
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def stats_note(self) -> list[str]:
        """手部检出率的可读摘要；没有任何检测尝试时返回空列表。"""
        if not self.stats["roiHands"] and not self.stats["fullFallbacks"]:
            return []
        parts: list[str] = []
        if self.stats["roiHands"]:
            parts.append(
                f"腕点 ROI 命中 {self.stats['roiHits']}/{self.stats['roiHands']} 只手"
                f"（{self.stats['roiCrops']} 次裁剪推理）"
            )
        if self.stats["fullFallbacks"]:
            parts.append(
                f"整帧兜底 {self.stats['fullFallbacks']} 次（命中 {self.stats['fullHits']} 只手）"
            )
        if self.stats["wristSkipped"]:
            parts.append(f"{self.stats['wristSkipped']} 次腕点可见度不足跳过")
        return ["手部检测：" + "；".join(parts)]

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


class _RoiBox(NamedTuple):
    """一块手腕 ROI：像素框 + 该侧的期望腕点（ROI 内归一化坐标）。"""

    x0: int
    y0: int
    width: int
    height: int
    side: str
    wrist_uv: tuple[float, float]
    hand_length_px: float


def _hand_roi_box(
    wrist: Landmark,
    elbow: Landmark,
    image_width: int,
    image_height: int,
    *,
    scale: float,
    min_box_px: int,
    side: str,
) -> _RoiBox | None:
    """按「腕点 + 前臂方向」估一块手部 ROI。

    手长由前臂投影长度按 0.75 倍估算，中心沿前臂方向从腕点外移 0.45 倍手长——
    手掌本身在腕点之前，这样五指（尤其伸直的食指）不会顶到框边被截断。
    越界时**平移**而不是裁小：手贴画面边缘时宁可带进一点背景，也不能把手指切掉。
    """
    wx, wy = wrist.x * image_width, wrist.y * image_height
    ex, ey = elbow.x * image_width, elbow.y * image_height
    vx, vy = wx - ex, wy - ey
    forearm = math.hypot(vx, vy)
    if forearm < 1e-3:
        # 前臂退化（手臂正对镜头时 2D 投影可能塌成一点）：用最小框兜底，方向取"向下"。
        hand_length = min_box_px / 2.0 / max(scale, 1e-3) * 0.5
        ux, uy = 0.0, 1.0
    else:
        hand_length = _ROI_HAND_LENGTH_RATIO * forearm
        ux, uy = vx / forearm, vy / forearm

    half = max(scale * hand_length, min_box_px / 2.0)
    size = max(2.0, half * 2.0)
    cx = wx + _ROI_CENTER_BIAS * hand_length * ux
    cy = wy + _ROI_CENTER_BIAS * hand_length * uy

    x0 = min(max(0.0, cx - size / 2.0), max(0.0, image_width - size))
    y0 = min(max(0.0, cy - size / 2.0), max(0.0, image_height - size))
    x0i, y0i = int(round(x0)), int(round(y0))
    box_width = min(int(size), image_width - x0i)
    box_height = min(int(size), image_height - y0i)
    if box_width < 24 or box_height < 24:
        return None
    return _RoiBox(
        x0=x0i,
        y0=y0i,
        width=box_width,
        height=box_height,
        side=side,
        wrist_uv=((wx - x0i) / box_width, (wy - y0i) / box_height),
        hand_length_px=hand_length,
    )


def _closest_hand(hands: Sequence[Any], box: _RoiBox) -> Any:
    """ROI 里可能检到多只手，取腕点离期望腕点最近、尺寸也对得上的那只。

    两道门槛缺一不可：腕点距离挡住"半身/另一只手"，尺寸挡住把远处的头发、鞋、
    衣角当成手——放大 ROI 时这两类误检都会变多。
    """
    best: Any = None
    best_distance = _ROI_WRIST_MATCH_MAX
    expected_span = 0.45 * box.hand_length_px / max(min(box.width, box.height), 1)
    for hand in hands:
        if len(hand) < HAND_LANDMARK_COUNT:
            continue
        distance = math.hypot(
            float(getattr(hand[0], "x", 0.0)) - box.wrist_uv[0],
            float(getattr(hand[0], "y", 0.0)) - box.wrist_uv[1],
        )
        if distance > best_distance:
            continue
        span = math.hypot(
            float(getattr(hand[9], "x", 0.0)) - float(getattr(hand[0], "x", 0.0)),
            float(getattr(hand[9], "y", 0.0)) - float(getattr(hand[0], "y", 0.0)),
        )
        if expected_span > 1e-6 and not (0.3 * expected_span <= span <= 3.0 * expected_span):
            continue
        best, best_distance = hand, distance
    return best


def _roi_to_landmark(
    item: Any, box: _RoiBox, image_width: int, image_height: int
) -> Landmark:
    """ROI 内归一化坐标 → 整帧归一化坐标；z 是相对腕点的深度，按宽度比例换算。"""
    x = (box.x0 + float(getattr(item, "x", 0.0)) * box.width) / max(image_width, 1)
    y = (box.y0 + float(getattr(item, "y", 0.0)) * box.height) / max(image_height, 1)
    z = float(getattr(item, "z", 0.0)) * box.width / max(image_width, 1)
    return Landmark(x=x, y=y, z=z, visibility=1.0)


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
