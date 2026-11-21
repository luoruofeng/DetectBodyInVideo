"""MediaPipe PoseLandmarker 引擎（任务 4.2，兜底路径，必做）。

设计：
- 依赖延迟导入：未安装 mediapipe 时只在选型阶段返回不可用，不影响服务启动。
- ``RunningMode.VIDEO`` + ``num_poses`` 提供跨帧跟踪；``smooth_landmarks=True``
  时启用 VIDEO 模式（MediaPipe 的时间维度跟踪即其平滑来源），否则退化为 IMAGE 模式。
- 模型档位 lite / full / heavy 与 ``AccelerationPlan.model_tier`` 对齐。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import (
    EngineCapabilities,
    EngineError,
    EngineUnavailableError,
    Landmark,
    PoseEngine,
    PoseFrame,
    engine_tier,
)
from app.backend.engine.models import POSE_MODEL_FILENAMES, resolve_mediapipe_model

logger = get_logger("engine.mediapipe")

# 下肢关键点：髋 / 膝 / 踝 / 脚跟 / 脚尖。ROI 重解只替换这一段。
LOWER_BODY_INDICES: tuple[int, ...] = (23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
# 触发 ROI 重解的可见度阈值：低于它说明整帧推理"看不见腿"
LOWER_BODY_REFINE_VISIBILITY = 0.55
# 裁剪框外扩比例：给"腿还没检测到"留出上下余量
LOWER_BODY_PADDING = 0.35


def _fallback_note(tier: str, resolved: Path | str) -> str:
    """实际加载的模型与请求档位不同时生成一条可展示的说明。"""
    expected = POSE_MODEL_FILENAMES.get(tier)
    name = Path(resolved).name
    if not expected or name == expected:
        return ""
    actual = next((key for key, value in POSE_MODEL_FILENAMES.items() if value == name), None)
    return f"{tier} 档模型缺失，已回退 {actual or name} 档（{name}）"


def _import_mediapipe() -> tuple[Any, Any, Any]:
    """返回 ``(mp, mp_python, mp_vision)``；缺失时抛 EngineUnavailableError。"""
    try:
        import mediapipe as mp  # noqa: PLC0415 - 延迟导入
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - 缺包 / 二进制不兼容
        raise EngineUnavailableError(
            "mediapipe 不可用，无法使用 MediaPipe 姿态引擎",
            details={
                "reason": str(exc),
                "hint": "pip install 'mediapipe>=0.10.14'（需 Python 3.9+）",
            },
        ) from exc
    return mp, mp_python, mp_vision


class MediaPipePoseEngine(PoseEngine):
    name = "mediapipe"
    display_name = "MediaPipe PoseLandmarker"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        plan: object | None = None,
        model_path: str | None = None,
        num_poses: int | None = None,
        smooth_landmarks: bool | None = None,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        delegate: str | None = None,
        lower_body_refine: bool | None = None,
    ) -> None:
        super().__init__(settings=settings, plan=plan, model_path=model_path)
        self.tier = engine_tier(settings, plan)
        self.lower_body_refine = bool(
            lower_body_refine
            if lower_body_refine is not None
            else (settings.lower_body_refine if settings else False)
        )
        self._crop_landmarker: Any = None
        self._refined_frames = 0
        self._refined_landmarks = 0
        self.num_poses = max(1, int(num_poses or (settings.max_poses if settings else 1)))
        if smooth_landmarks is None:
            smooth_landmarks = settings.smooth_landmarks if settings else True
        self.smooth_landmarks = bool(smooth_landmarks)
        self.min_detection_confidence = float(
            min_detection_confidence
            if min_detection_confidence is not None
            else (settings.min_detection_confidence if settings else 0.5)
        )
        self.min_tracking_confidence = float(
            min_tracking_confidence
            if min_tracking_confidence is not None
            else (settings.min_tracking_confidence if settings else 0.5)
        )
        self.delegate = delegate or (
            "GPU" if settings and settings.mediapipe_gpu_delegate else "CPU"
        )
        self._landmarker: Any = None
        self._mp: Any = None
        self._mp_image_class: Any = None
        self._mp_image_format: Any = None
        self._mode = "IMAGE"
        self._last_timestamp_ms = -1
        self._resolved_model: str | None = None

    # ---- 选型 ----
    @classmethod
    def availability(
        cls, settings: Settings | None = None, plan: object | None = None
    ) -> tuple[bool, str]:
        try:
            _import_mediapipe()
        except EngineUnavailableError as exc:
            return False, exc.message
        tier = engine_tier(settings, plan)
        try:
            model_file = resolve_mediapipe_model(
                "pose",
                settings=settings,
                tier=tier,
                allow_download=bool(settings and settings.auto_download_models),
            )
        except EngineUnavailableError as exc:
            return False, exc.message
        return True, _fallback_note(tier, model_file)

    # ---- 生命周期 ----
    def prepare(self, device: str | None = None) -> None:
        if self._prepared:
            return
        mp, mp_python, mp_vision = _import_mediapipe()
        model_file = resolve_mediapipe_model(
            "pose",
            settings=self.settings,
            tier=self.tier,
            explicit=self.model_path,
        )
        note = _fallback_note(self.tier, model_file)
        if note:
            self._notes.append(note)
        self._resolved_model = str(model_file)
        self._mode = "VIDEO" if self.smooth_landmarks else "IMAGE"
        running_mode = mp_vision.RunningMode.VIDEO if self._mode == "VIDEO" else mp_vision.RunningMode.IMAGE

        delegate = mp_python.BaseOptions.Delegate.GPU if (device or self.delegate).upper() == "GPU" else mp_python.BaseOptions.Delegate.CPU
        try:
            self._landmarker = self._create(mp_python, mp_vision, model_file, running_mode, delegate)
        except Exception as exc:  # noqa: BLE001 - GPU delegate 在多数轮子上不可用
            if delegate == mp_python.BaseOptions.Delegate.GPU:
                self._notes.append(f"GPU delegate 创建失败，已回退 CPU：{exc}")
                logger.warning("MediaPipe GPU delegate 不可用，回退 CPU", exc_info=True)
                delegate = mp_python.BaseOptions.Delegate.CPU
                try:
                    self._landmarker = self._create(
                        mp_python, mp_vision, model_file, running_mode, delegate
                    )
                except Exception as exc2:  # noqa: BLE001
                    raise EngineError(
                        "MediaPipe PoseLandmarker 初始化失败", details={"reason": str(exc2)}
                    ) from exc2
            else:
                raise EngineError(
                    "MediaPipe PoseLandmarker 初始化失败", details={"reason": str(exc)}
                ) from exc

        self._mp = mp
        self._mp_image_class = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB
        if self.lower_body_refine:
            # ROI 重解用独立 landmarker（IMAGE 模式）：不污染主 landmarker 的跨帧跟踪，
            # 否则同一帧被喂两次会打乱 VIDEO 模式的时间戳单调性。
            try:
                self._crop_landmarker = self._create(
                    mp_python,
                    mp_vision,
                    model_file,
                    mp_vision.RunningMode.IMAGE,
                    delegate,
                    num_poses=1,
                )
                self._notes.append("下肢 ROI 重解已启用（坏帧裁剪下半身再推理一次）")
            except Exception as exc:  # noqa: BLE001 - 可选增强，失败只降级
                self._crop_landmarker = None
                self._notes.append(f"下肢 ROI 重解创建失败，已跳过：{exc}")
                logger.warning("下肢 ROI landmarker 创建失败", exc_info=True)
        self._prepared = True
        logger.info(
            "MediaPipe 引擎就绪：tier=%s mode=%s poses=%s model=%s",
            self.tier,
            self._mode,
            self.num_poses,
            model_file.name,
        )

    def _create(
        self,
        mp_python: Any,
        mp_vision: Any,
        model_file: Any,
        running_mode: Any,
        delegate: Any,
        num_poses: int | None = None,
    ) -> Any:
        base_options = mp_python.BaseOptions(model_asset_path=str(model_file), delegate=delegate)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=num_poses or self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:  # noqa: BLE001 - 关闭失败不阻塞退出
                logger.debug("关闭 MediaPipe landmarker 失败", exc_info=True)
        crop, self._crop_landmarker = self._crop_landmarker, None
        if crop is not None:
            try:
                crop.close()
            except Exception:  # noqa: BLE001
                logger.debug("关闭下肢 ROI landmarker 失败", exc_info=True)
        self._prepared = False

    # ---- 推理 ----
    def _to_landmarks(self, raw: Any) -> list[Landmark]:
        return [
            Landmark(
                x=float(getattr(item, "x", 0.0)),
                y=float(getattr(item, "y", 0.0)),
                z=float(getattr(item, "z", 0.0)),
                visibility=float(getattr(item, "visibility", 0.0) or 0.0),
                presence=(
                    float(item.presence)
                    if getattr(item, "presence", None) is not None
                    else None
                ),
            )
            for item in raw
        ]

    def infer(
        self, image: np.ndarray, *, index: int = 0, timestamp_ms: float = 0.0
    ) -> list[PoseFrame]:
        if not self._prepared or self._landmarker is None:
            raise EngineError("引擎尚未 prepare()，不能推理")

        rgb = np.ascontiguousarray(image[:, :, ::-1]) if image.ndim == 3 else image
        mp_image = self._mp_image_class(image_format=self._mp_image_format, data=rgb)

        try:
            if self._mode == "VIDEO":
                # detect_for_video 要求时间戳严格递增
                stamp = max(self._last_timestamp_ms + 1, int(round(timestamp_ms)))
                self._last_timestamp_ms = stamp
                result = self._landmarker.detect_for_video(mp_image, stamp)
            else:
                result = self._landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001 - 单帧失败不应打断整段分析
            raise EngineError(
                "MediaPipe 推理失败", details={"index": index, "reason": str(exc)}
            ) from exc

        frames: list[PoseFrame] = []
        for person_index, raw in enumerate(getattr(result, "pose_landmarks", None) or []):
            landmarks = self._to_landmarks(raw)
            world_raw = getattr(result, "pose_world_landmarks", None) or []
            world = (
                self._to_landmarks(world_raw[person_index])
                if person_index < len(world_raw)
                else []
            )
            frame = PoseFrame(
                index=index,
                timestamp_ms=float(timestamp_ms),
                landmarks=landmarks,
                world_landmarks=world,
                score=_score_of(landmarks),
                valid=bool(landmarks),
            )
            if self._crop_landmarker is not None:
                self._refine_lower_body(frame, rgb, image.shape)
            frame.bbox = frame.bbox_from_landmarks()
            frames.append(frame)
        return frames

    # ---- 下肢 ROI 重解（§5.1） ----
    def _lower_body_visibility(self, frame: PoseFrame) -> float:
        values = [
            frame.landmarks[index].visibility
            for index in LOWER_BODY_INDICES
            if index < len(frame.landmarks)
        ]
        return float(sum(values) / len(values)) if values else 0.0

    def _lower_body_box(
        self, frame: PoseFrame, shape: tuple[int, ...]
    ) -> tuple[int, int, int, int] | None:
        """由髋/膝/踝/脚跟/脚尖的 2D 位置算裁剪框（像素，含外扩）。"""
        height, width = shape[0], shape[1]
        xs: list[float] = []
        ys: list[float] = []
        for index in LOWER_BODY_INDICES:
            if index >= len(frame.landmarks):
                continue
            point = frame.landmarks[index]
            if point.visibility <= 0.05:
                continue
            xs.append(point.x)
            ys.append(point.y)
        if len(xs) < 2:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        box_w = max(0.12, (x1 - x0) * (1 + LOWER_BODY_PADDING))
        box_h = max(0.20, (y1 - y0) * (1 + LOWER_BODY_PADDING))
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2 + box_h * 0.1  # 略向下偏：腿在髋以下
        left = int(max(0.0, center_x - box_w / 2) * width)
        right = int(min(1.0, center_x + box_w / 2) * width)
        top = int(max(0.0, center_y - box_h / 2) * height)
        bottom = int(min(1.0, center_y + box_h / 2) * height)
        if right - left < 16 or bottom - top < 16:
            return None
        return left, top, right, bottom

    def _refine_lower_body(
        self, frame: PoseFrame, rgb: np.ndarray, shape: tuple[int, ...]
    ) -> None:
        """下半身裁剪后再推理一次，只替换更可信的下肢点（2D 与 z 同步换回全图坐标）。

        为什么这样做：整帧推理时下肢在画面里只占十几行像素，MediaPipe 会给出低
        visibility 的点，z 基本是外推值；裁到下半身再推理等于给了一个"等效长焦"，
        同一帧的腿部站位能重新解一次。**不做时序平均**，避免与后续平滑冲突。
        """
        if not frame.valid or len(frame.landmarks) < 33:
            return
        if self._lower_body_visibility(frame) >= LOWER_BODY_REFINE_VISIBILITY:
            return
        box = self._lower_body_box(frame, shape)
        if box is None:
            return
        left, top, right, bottom = box
        crop = np.ascontiguousarray(rgb[top:bottom, left:right])
        if crop.size == 0:
            return
        try:
            mp_image = self._mp_image_class(image_format=self._mp_image_format, data=crop)
            result = self._crop_landmarker.detect(mp_image)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - 单帧增强失败不影响主链
            logger.debug("下肢 ROI 推理失败（帧 %s），保留整帧结果", frame.index, exc_info=True)
            return
        raw = getattr(result, "pose_landmarks", None) or []
        if not raw:
            return
        refined = self._to_landmarks(raw[0])
        height, width = shape[0], shape[1]
        crop_w = right - left
        crop_h = bottom - top
        replaced = 0
        for index in LOWER_BODY_INDICES:
            if index >= len(refined) or index >= len(frame.landmarks):
                continue
            candidate = refined[index]
            current = frame.landmarks[index]
            if candidate.visibility <= current.visibility:
                continue
            frame.landmarks[index] = Landmark(
                x=(left + candidate.x * crop_w) / width,
                y=(top + candidate.y * crop_h) / height,
                # MediaPipe 的 z 与 x 同尺度（归一化图宽），裁剪后要按裁剪宽度缩放回来
                z=candidate.z * crop_w / width,
                visibility=candidate.visibility,
                presence=candidate.presence,
            )
            replaced += 1
        if replaced:
            self._refined_frames += 1
            self._refined_landmarks += replaced

    # ---- 能力 ----
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            display_name=self.display_name,
            multi_person=self.num_poses > 1,
            world_landmarks=True,
            hands=False,
            face=False,
            providers=["mediapipe-tasks"],
            model_tier=self.tier,
            device=self.delegate,
            notes=self._notes
            + [
                f"tracking_mode={self._mode}",
                f"model={self._resolved_model or '(未加载)'}",
                *(
                    [f"下肢 ROI 重解：{self._refined_frames} 帧 / {self._refined_landmarks} 个点"]
                    if self._refined_frames
                    else []
                ),
            ],
        )


def _score_of(landmarks: list[Landmark]) -> float:
    values = [item.visibility for item in landmarks if item.visibility > 0]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
