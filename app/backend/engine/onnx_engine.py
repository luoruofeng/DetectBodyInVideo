"""ONNX Runtime 加速引擎（任务 4.3，RTMPose / RTMO）。

链路：HOG 行人粗检 → 人体框裁剪（等比外扩 + 仿射对齐到模型输入）→
ONNX 推理 → SimCC / 回归解码 → 反变换回原图 → 映射到 MediaPipe 33 点 → IoU 跟踪。

说明：
- RTMPose 输出 17 个 COCO 关键点，本项目对外统一 33 点，缺失点 ``visibility=0``，
  由任务 6 的映射层推导（如 spine / toes）。
- RTMPose 不提供 ``z``，``world_landmarks`` 为空；没有世界坐标就没有米制关节，
  3D 预览 / 导出请改用 MediaPipe 引擎。
- 会话创建失败会按 EP 链自动降级到 CPU，并把原因写进 ``notes``。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import (
    LANDMARK_COUNT,
    BoundingBox,
    EngineCapabilities,
    EngineError,
    EngineUnavailableError,
    Landmark,
    PoseEngine,
    PoseFrame,
    engine_tier,
    providers_from_plan,
)
from app.backend.engine.models import RTMPOSE_HINT, resolve_onnx_model
from app.backend.engine.tracking import IouTracker

logger = get_logger("engine.onnx")

DEFAULT_INPUT_SIZE = (192, 256)  # (width, height)，RTMPose-m 默认
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)

# COCO 17 关键点 → MediaPipe 33 点索引
COCO17_TO_MP33: dict[int, int] = {
    0: 0,  # nose
    1: 2,  # left_eye
    2: 5,  # right_eye
    3: 7,  # left_ear
    4: 8,  # right_ear
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    9: 15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}
SUPPORTED_KEYPOINT_COUNTS = (17,)


class OnnxPoseConfig(BaseModel):
    simcc_split_ratio: float = Field(2.0, gt=0, le=4)
    bbox_padding: float = Field(1.25, ge=1.0, le=2.0, description="人体框外扩倍数")
    detector_score: float = Field(0.35, ge=0, le=1, description="HOG 行人置信度阈值")
    max_persons: int = Field(1, ge=1, le=10)
    pad_value: int = 0
    normalize: bool = True


# --------------------------------------------------------------------------- #
# 图像预处理 / 解码（纯函数，便于单测）
# --------------------------------------------------------------------------- #
def crop_to_input(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    out_size: tuple[int, int],
    *,
    padding: float = 1.25,
    pad_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """按归一化 bbox 裁剪并仿射对齐到 ``out_size``。

    返回 ``(crop_bgr, inverse_matrix)``；``inverse_matrix`` 用于把裁剪坐标系
    的像素点映射回原图（``crop_xy @ M_inv.T``）。
    """
    height, width = image.shape[:2]
    out_w, out_h = out_size
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) * width
    cy = (y + bh / 2.0) * height
    box_w = max(1.0, bw * width) * padding
    box_h = max(1.0, bh * height) * padding

    aspect = out_w / out_h
    if box_w / box_h > aspect:
        box_h = box_w / aspect
    else:
        box_w = box_h * aspect

    x1, y1 = cx - box_w / 2.0, cy - box_h / 2.0
    x2, y2 = cx + box_w / 2.0, cy + box_h / 2.0

    source = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    target = np.float32([[0.0, 0.0], [0.0, float(out_h)], [float(out_w), 0.0]])
    matrix = cv2.getAffineTransform(source, target)
    crop = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value),
    )
    return crop, cv2.invertAffineTransform(matrix)


def preprocess(crop_bgr: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """HWC BGR uint8 → NCHW float32（ImageNet 均值方差）。"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if normalize:
        rgb = (rgb - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
            IMAGENET_STD, dtype=np.float32
        )
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
    return tensor


def as_persons(array: np.ndarray) -> np.ndarray:
    """把 ``(K,S)`` / ``(1,K,S)`` / ``(1,N,K,S)`` 统一成 ``(P,K,S)``。"""
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return array[0]
    raise EngineError("不认识的模型输出形状", details={"shape": list(array.shape)})


def _row_probs(values: np.ndarray) -> np.ndarray:
    """SimCC 输出可能是概率也可能是 logits：非概率时按行做 softmax。"""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
        return array
    shifted = array - array.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-9)


def decode_simcc(
    x_logits: np.ndarray, y_logits: np.ndarray, *, split_ratio: float = 2.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """SimCC 解码：每个 person 返回 ``(coords(K,2) 输入像素坐标, scores(K,))``。"""
    persons_x = as_persons(np.asarray(x_logits))
    persons_y = as_persons(np.asarray(y_logits))
    count = min(persons_x.shape[0], persons_y.shape[0])
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for p in range(count):
        logits_x = persons_x[p]
        logits_y = persons_y[p]
        if logits_x.shape[0] != logits_y.shape[0]:
            raise EngineError(
                "SimCC 的 x/y 关键点数量不一致",
                details={"x": list(logits_x.shape), "y": list(logits_y.shape)},
            )
        probs_x = _row_probs(logits_x)
        probs_y = _row_probs(logits_y)
        index_x = np.argmax(probs_x, axis=-1).astype(np.float32)
        index_y = np.argmax(probs_y, axis=-1).astype(np.float32)
        coords = np.stack([index_x / split_ratio, index_y / split_ratio], axis=-1)
        scores = np.minimum(
            probs_x.max(axis=-1).astype(np.float32),
            probs_y.max(axis=-1).astype(np.float32),
        )
        results.append((coords, scores))
    return results


def decode_regression(output: np.ndarray, input_size: tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """直接回归坐标的模型：``(K,2)`` / ``(N,K,2)``，值 ≤1.5 视为归一化。"""
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 2:
        persons = array[None, ...]
    elif array.ndim == 3:
        persons = array
    else:
        raise EngineError("不认识的回归输出形状", details={"shape": list(array.shape)})
    out_w, out_h = input_size
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for person in persons:
        coords = person[:, :2].copy()
        if float(np.nanmax(np.abs(coords))) <= 1.5:
            coords[:, 0] *= out_w
            coords[:, 1] *= out_h
        scores = (
            person[:, 2].copy() if person.shape[1] > 2 else np.ones(person.shape[0], dtype=np.float32)
        )
        np.nan_to_num(coords, copy=False, nan=0.0)
        results.append((coords, scores.astype(np.float32)))
    return results


def decode_outputs(
    outputs: list[np.ndarray],
    input_size: tuple[int, int],
    *,
    names: list[str] | None = None,
    split_ratio: float = 2.0,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """自动识别输出格式（SimCC 双输出 / 单输出回归）。"""
    if not outputs:
        raise EngineError("模型没有任何输出")
    arrays = [np.asarray(item) for item in outputs]

    # 1) 优先按输出名判定（RTMPose 导出通常为 simcc_x / simcc_y）
    if names and len(names) == len(arrays):
        lowered = [name.lower() for name in names]
        x_index = next((i for i, n in enumerate(lowered) if "simcc" in n and n.endswith("_x")), None)
        y_index = next((i for i, n in enumerate(lowered) if "simcc" in n and n.endswith("_y")), None)
        if x_index is not None and y_index is not None:
            return "simcc", decode_simcc(
                arrays[x_index], arrays[y_index], split_ratio=split_ratio
            )

    # 2) 按形状推断：x 轴长度 ≈ 输入宽 × split_ratio，y 轴 ≈ 输入高 × split_ratio
    out_w, out_h = input_size
    expect_x = out_w * split_ratio
    expect_y = out_h * split_ratio
    simcc_candidates = [item for item in arrays if item.ndim >= 3 and item.shape[-1] > 32]
    if len(simcc_candidates) >= 2:
        def _distance(item: np.ndarray, expected: float) -> float:
            return abs(float(item.shape[-1]) - expected)

        pool = sorted(simcc_candidates, key=lambda item: item.shape[-1])
        best_pair: tuple[np.ndarray, np.ndarray, float] | None = None
        for i, first in enumerate(pool):
            for second in pool[i + 1 :]:
                cost = min(
                    _distance(first, expect_x) + _distance(second, expect_y),
                    _distance(first, expect_y) + _distance(second, expect_x),
                )
                if best_pair is None or cost < best_pair[2]:
                    best_pair = (first, second, cost)
        if best_pair is not None and best_pair[2] < max(expect_x, expect_y):
            first, second, _cost = best_pair
            if _distance(first, expect_x) + _distance(second, expect_y) <= (
                _distance(first, expect_y) + _distance(second, expect_x)
            ):
                return "simcc", decode_simcc(first, second, split_ratio=split_ratio)
            return "simcc", decode_simcc(second, first, split_ratio=split_ratio)

    return "regression", decode_regression(arrays[0], input_size)


# --------------------------------------------------------------------------- #
# 检测器
# --------------------------------------------------------------------------- #
class HogPersonDetector:
    """OpenCV HOG 行人检测；不可用时退回整帧（单人场景足够）。"""

    def __init__(self, *, score_threshold: float = 0.35, max_persons: int = 1) -> None:
        self.score_threshold = score_threshold
        self.max_persons = max_persons
        self._hog: Any = None
        self._unavailable_reason = ""
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        except Exception as exc:  # noqa: BLE001 - 精简版 OpenCV 可能缺 HOG
            self._unavailable_reason = str(exc)
            logger.warning("HOG 行人检测不可用，退化为整帧框：%s", exc)

    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        """返回归一化 bbox 列表，按面积降序；空列表表示未检测到。"""
        if self._hog is None:
            return []
        height, width = image.shape[:2]
        scale = 1.0
        work = image
        if width > 512:
            scale = 512.0 / width
            work = cv2.resize(image, (512, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        try:
            rects, weights = self._hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
        except cv2.error:
            logger.warning("HOG 检测失败，退化为整帧框", exc_info=True)
            return []
        boxes: list[tuple[float, float, float, float, float]] = []
        for (x, y, w, h), weight in zip(rects, weights, strict=False):
            if float(weight) < self.score_threshold:
                continue
            boxes.append(
                (
                    x / scale / width,
                    y / scale / height,
                    w / scale / width,
                    h / scale / height,
                    float(weight),
                )
            )
        boxes.sort(key=lambda item: item[2] * item[3], reverse=True)
        return [(b[0], b[1], b[2], b[3]) for b in boxes[: self.max_persons]]


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class OnnxPoseEngine(PoseEngine):
    name = "onnx"
    display_name = "ONNX Runtime (RTMPose/RTMO)"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        plan: object | None = None,
        model_path: str | None = None,
        config: OnnxPoseConfig | None = None,
        multi_person: bool | None = None,
    ) -> None:
        super().__init__(settings=settings, plan=plan, model_path=model_path)
        self.tier = engine_tier(settings, plan)
        self.config = config or OnnxPoseConfig(
            max_persons=max(1, settings.max_poses if settings else 1)
        )
        if settings is not None:
            self.config.max_persons = max(1, min(self.config.max_persons, settings.max_poses))
        self._multi_person = multi_person
        self._session: Any = None
        self._input_name = "input"
        self._input_size = DEFAULT_INPUT_SIZE
        self._providers: list[str] = []
        self._tracker = IouTracker(
            iou_threshold=settings.track_iou_threshold if settings else 0.3,
            max_age=settings.track_max_age if settings else 15,
        )
        self._resolved_model: Path | None = None
        self._output_kind = "unknown"
        self._detector: HogPersonDetector | None = None
        self._thread_override = 0
        self._fp16_policy = "auto"
        self._fp16_active = False

    # ---- 选型 ----
    @classmethod
    def availability(
        cls, settings: Settings | None = None, plan: object | None = None
    ) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401, PLC0415 - 探测用
        except Exception as exc:  # noqa: BLE001
            return False, f"onnxruntime 未安装：{exc}"
        try:
            resolve_onnx_model(settings=settings, tier=engine_tier(settings, plan))
        except EngineUnavailableError as exc:
            return False, exc.message
        return True, ""

    @property
    def multi_person(self) -> bool:
        if self._multi_person is not None:
            return self._multi_person
        name = (self._resolved_model.name if self._resolved_model else "").lower()
        return "rtmo" in name

    # ---- 生命周期 ----
    def apply_runtime(self, config: object | None) -> list[str]:
        """任务 9.1：线程数与 FP16 策略在 ``prepare()`` 前生效。"""
        self.runtime = config
        notes: list[str] = []
        threads = int(getattr(config, "worker_threads", 0) or 0)
        if threads > 0:
            self._thread_override = threads
            notes.append(f"推理线程固定为 {threads}（运行时配置）")
        policy = str(getattr(config, "fp16", "auto") or "auto")
        if policy in {"auto", "on", "off"}:
            self._fp16_policy = policy
        self._notes.extend(notes)
        return notes

    def prepare(self, device: str | None = None) -> None:
        if self._prepared:
            return
        # 非 create_app 入口（脚本 / 基准测试）也要保证 GPU 运行时先注入 sys.path
        try:
            from app.backend.engine.accelerator_install import activate as _activate_runtime  # noqa: PLC0415

            _activate_runtime(self.settings)
        except Exception:  # noqa: BLE001 - 激活失败不应阻断引擎
            logger.debug("加速运行时激活失败", exc_info=True)
        try:
            import onnxruntime as ort  # noqa: PLC0415 - 延迟导入
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(
                "onnxruntime 未安装，无法使用 ONNX 加速引擎",
                details={"reason": str(exc), "hint": "pip install onnxruntime（GPU 机器用 onnxruntime-gpu）"},
            ) from exc

        wanted = providers_from_plan(self.plan)
        available = set(ort.get_available_providers())
        providers = [item for item in wanted if item in available] or ["CPUExecutionProvider"]

        model_file = resolve_onnx_model(
            settings=self.settings,
            tier=self.tier,
            explicit=self.model_path or (self.settings.onnx_pose_model if self.settings else None),
        )
        fp16_model = self._pick_fp16_model(model_file, providers)
        if fp16_model is not None:
            model_file = fp16_model
            self._fp16_active = True
            self._notes.append(f"使用 FP16 权重 {fp16_model.name}")
        elif self._fp16_policy != "off":
            self._notes.append("未找到同名 .fp16.onnx 权重，保持 FP32")
        self._resolved_model = model_file

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = self._thread_override or int(getattr(self.plan, "threads", 0) or 0)
        if threads > 0:
            session_options.intra_op_num_threads = threads
            session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3

        session = self._create_session(ort, model_file, session_options, providers)
        self._session = session
        self._providers = list(session.get_providers())

        inputs = session.get_inputs()
        if not inputs:
            raise EngineUnavailableError("ONNX 模型没有输入节点", details={"model": str(model_file)})
        self._input_name = inputs[0].name
        self._input_size = _resolve_input_size(inputs[0].shape)

        if "CUDAExecutionProvider" in self._providers or "TensorrtExecutionProvider" in self._providers:
            logger.info("ONNX 引擎使用 GPU 加速：%s", ", ".join(self._providers))
        else:
            logger.info("ONNX 引擎运行在 CPU：%s", ", ".join(self._providers))

        self._detector = HogPersonDetector(
            score_threshold=self.config.detector_score,
            max_persons=1 if not self.multi_person else self.config.max_persons,
        )
        self._prepared = True

    def _pick_fp16_model(self, model_file: Path, providers: list[str]) -> Path | None:
        """找同名 FP16 权重；``auto`` 只在真正有加速 EP 时尝试。"""
        if self._fp16_policy == "off":
            return None
        accelerated = {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CoreMLExecutionProvider",
        }
        if self._fp16_policy == "auto" and not accelerated & set(providers):
            return None
        candidates = (
            model_file.with_name(f"{model_file.stem}.fp16{model_file.suffix}"),
            model_file.with_name(f"{model_file.stem}_fp16{model_file.suffix}"),
        )
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _create_session(
        self, ort: Any, model_file: Path, options: Any, providers: list[str]
    ) -> Any:
        try:
            return ort.InferenceSession(str(model_file), sess_options=options, providers=providers)
        except Exception as exc:  # noqa: BLE001 - GPU EP 初始化失败很常见
            if providers == ["CPUExecutionProvider"]:
                raise EngineUnavailableError(
                    "ONNX 会话创建失败",
                    details={"model": str(model_file), "reason": str(exc), "hint": RTMPOSE_HINT},
                ) from exc
            self._notes.append(f"EP {providers} 初始化失败，已回退 CPU：{exc}")
            logger.warning("ONNX EP %s 初始化失败，回退 CPU", providers, exc_info=True)
            try:
                return ort.InferenceSession(
                    str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception as exc2:  # noqa: BLE001
                raise EngineUnavailableError(
                    "ONNX 会话创建失败（CPU 兜底也失败）",
                    details={"model": str(model_file), "reason": str(exc2)},
                ) from exc2

    def close(self) -> None:
        self._session = None
        self._prepared = False

    # ---- 推理 ----
    def _detect_boxes(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        if self._detector is None:
            self._detector = HogPersonDetector(
                score_threshold=self.config.detector_score,
                max_persons=1 if not self.multi_person else self.config.max_persons,
            )
        boxes = self._detector.detect(image)
        if not boxes:
            boxes = [(0.0, 0.0, 1.0, 1.0)]
        return boxes

    def infer(
        self, image: np.ndarray, *, index: int = 0, timestamp_ms: float = 0.0
    ) -> list[PoseFrame]:
        if not self._prepared or self._session is None:
            raise EngineError("引擎尚未 prepare()，不能推理")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return []

        frames: list[PoseFrame] = []
        for bbox in self._detect_boxes(image):
            crop, matrix_inv = crop_to_input(
                image,
                bbox,
                self._input_size,
                padding=self.config.bbox_padding,
                pad_value=self.config.pad_value,
            )
            tensor = preprocess(crop, normalize=self.config.normalize)
            try:
                outputs = self._session.run(None, {self._input_name: tensor})
            except Exception as exc:  # noqa: BLE001 - 单帧失败不打断整段
                raise EngineError(
                    "ONNX 推理失败",
                    details={"index": index, "model": str(self._resolved_model), "reason": str(exc)},
                ) from exc

            kind, persons = decode_outputs(
                outputs,
                self._input_size,
                names=[item.name for item in self._session.get_outputs()],
                split_ratio=self.config.simcc_split_ratio,
            )
            self._output_kind = kind
            for coords, scores in persons[: self.config.max_persons]:
                frame = _to_pose_frame(
                    coords,
                    scores,
                    matrix_inv,
                    width=width,
                    height=height,
                    index=index,
                    timestamp_ms=timestamp_ms,
                    bbox=bbox,
                )
                if frame is not None:
                    frames.append(frame)

        if self.multi_person and frames:
            self._tracker.update(frames, index)
        elif frames:
            frames[0].track_id = 1
        return frames

    def infer_batch(
        self,
        frames: Iterable[tuple[int, float, np.ndarray]],
    ) -> list[list[PoseFrame]]:
        """任务 9.1：单人模型把 N 帧堆成一个 batch 一次推理。

        RTMO（多人）需要逐帧跑检测 + 跟踪，保持顺序调用。
        """
        items = [(int(index), float(ts), image) for index, ts, image in frames]
        if len(items) <= 1 or self.multi_person:
            return [
                self.infer(image, index=index, timestamp_ms=timestamp)
                for index, timestamp, image in items
            ]
        if not self._prepared or self._session is None:
            raise EngineError("引擎尚未 prepare()，不能推理")

        results: list[list[PoseFrame]] = [[] for _ in items]
        tensors: list[np.ndarray] = []
        metas: list[tuple[int, float, int, int, tuple[float, float, float, float], np.ndarray]] = []

        for slot, (_index, timestamp, image) in enumerate(items):
            height, width = image.shape[:2]
            if height <= 0 or width <= 0:
                continue
            bbox = self._detect_boxes(image)[0]
            crop, matrix_inv = crop_to_input(
                image,
                bbox,
                self._input_size,
                padding=self.config.bbox_padding,
                pad_value=self.config.pad_value,
            )
            tensors.append(preprocess(crop, normalize=self.config.normalize))
            metas.append((slot, timestamp, width, height, bbox, matrix_inv))

        if not tensors:
            return results

        batch_input = np.concatenate(tensors, axis=0)
        try:
            outputs = self._session.run(None, {self._input_name: batch_input})
        except Exception as exc:  # noqa: BLE001
            raise EngineError(
                "ONNX 批量推理失败",
                details={"batch": len(tensors), "reason": str(exc)},
            ) from exc

        kind, persons = decode_outputs(
            outputs,
            self._input_size,
            names=[item.name for item in self._session.get_outputs()],
            split_ratio=self.config.simcc_split_ratio,
        )
        self._output_kind = f"{kind}-batch"
        for position, (slot, timestamp, width, height, bbox, matrix_inv) in enumerate(metas):
            if position >= len(persons):
                break
            coords, scores = persons[position]
            index = items[slot][0]
            frame = _to_pose_frame(
                coords,
                scores,
                matrix_inv,
                width=width,
                height=height,
                index=index,
                timestamp_ms=timestamp,
                bbox=bbox,
            )
            if frame is not None:
                frame.track_id = 1
                results[slot] = [frame]
        return results

    # ---- 能力 ----
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            display_name=self.display_name,
            multi_person=self.multi_person,
            world_landmarks=False,
            hands=False,
            face=False,
            supports_batching=not self.multi_person,
            providers=self._providers or ["CPUExecutionProvider"],
            model_tier=self.tier,
            device=getattr(self.plan, "device_name", None),
            notes=self._notes
            + [
                f"model={self._resolved_model.name if self._resolved_model else '(未加载)'}",
                f"input={self._input_size[0]}x{self._input_size[1]}",
                f"output={self._output_kind}",
                f"fp16={'on' if self._fp16_active else 'off'}",
                "RTMPose 不提供 z/world landmarks，没有米制 3D 关节（请改用 MediaPipe）",
            ],
        )


def _resolve_input_size(shape: Any) -> tuple[int, int]:
    """ONNX 输入形状 ``[N, 3, H, W]``，动态维回退默认值。"""
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return DEFAULT_INPUT_SIZE
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or height <= 0:
        height = DEFAULT_INPUT_SIZE[1]
    if not isinstance(width, int) or width <= 0:
        width = DEFAULT_INPUT_SIZE[0]
    return (int(width), int(height))


def _to_pose_frame(
    coords: np.ndarray,
    scores: np.ndarray,
    matrix_inv: np.ndarray,
    *,
    width: int,
    height: int,
    index: int,
    timestamp_ms: float,
    bbox: tuple[float, float, float, float],
) -> PoseFrame | None:
    """裁剪坐标 → 原图归一化坐标 → MediaPipe 33 点。"""
    if coords.shape[0] not in SUPPORTED_KEYPOINT_COUNTS:
        raise EngineError(
            "不支持的模型关键点数量（当前仅支持 COCO 17 点模型）",
            details={"keypoints": int(coords.shape[0]), "supported": list(SUPPORTED_KEYPOINT_COUNTS)},
        )

    homogeneous = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)
    image_points = homogeneous @ matrix_inv.T
    landmarks = [Landmark() for _ in range(LANDMARK_COUNT)]
    for coco_index, mp_index in COCO17_TO_MP33.items():
        if coco_index >= len(image_points):
            continue
        x, y = float(image_points[coco_index, 0]), float(image_points[coco_index, 1])
        landmarks[mp_index] = Landmark(
            x=x / max(1, width),
            y=y / max(1, height),
            z=0.0,
            visibility=float(max(0.0, min(1.0, scores[coco_index]))),
        )

    visible = sum(1 for item in landmarks if item.visibility >= 0.3)
    if visible < 4:
        return None
    score = float(np.mean([landmarks[i].visibility for i in COCO17_TO_MP33.values()]))
    frame = PoseFrame(
        index=index,
        timestamp_ms=float(timestamp_ms),
        landmarks=landmarks,
        world_landmarks=[],
        score=score,
        valid=True,
        bbox=None,
    )
    # 检测框比关键点外接框更稳（关键点可能漏掉脚部），优先用检测框
    detection = BoundingBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
    frame.bbox = detection if detection.area > 0 else frame.bbox_from_landmarks()
    return frame
