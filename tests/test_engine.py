"""推理引擎抽象层 / 跟踪器 / ONNX 解码单测（任务 4.1、4.3、4.4）。"""

from __future__ import annotations

import numpy as np
import pytest

from app.backend.core.errors import AppError
from app.backend.core.settings import Settings
from app.backend.engine import registry
from app.backend.engine.base import (
    LANDMARK_COUNT,
    BoundingBox,
    EngineCapabilities,
    EngineUnavailableError,
    Landmark,
    PoseEngine,
    PoseFrame,
)
from app.backend.engine.consistency import compare_frames
from app.backend.engine.onnx_engine import (
    COCO17_TO_MP33,
    crop_to_input,
    decode_outputs,
    decode_simcc,
    preprocess,
)
from app.backend.engine.tracking import (
    IouTracker,
    drop_short_tracks,
    filter_track,
    primary_track_id,
    summarize_tracks,
)


# --------------------------------------------------------------------------- #
# 假引擎
# --------------------------------------------------------------------------- #
def make_frame(index: int, bbox: BoundingBox, *, score: float = 0.9) -> PoseFrame:
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)]
    return PoseFrame(
        index=index,
        timestamp_ms=index * 33.0,
        landmarks=landmarks,
        bbox=bbox,
        score=score,
    )


class _OkEngine(PoseEngine):
    name = "ok"
    display_name = "OK Engine"

    @classmethod
    def availability(cls, settings: Settings | None = None) -> tuple[bool, str]:
        return True, ""

    def prepare(self, device: str | None = None) -> None:
        self._prepared = True

    def infer(self, image, *, index: int = 0, timestamp_ms: float = 0.0):
        return [make_frame(index, BoundingBox(x=0.1, y=0.1, width=0.2, height=0.4))]

    def close(self) -> None:
        self._prepared = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(name=self.name, display_name=self.display_name, model_tier="lite")


class _BrokenEngine(_OkEngine):
    name = "broken"
    display_name = "Broken Engine"

    def prepare(self, device: str | None = None) -> None:
        raise EngineUnavailableError("故意失败", details={"test": True})


class _MissingEngine(_OkEngine):
    name = "missing"
    display_name = "Missing Engine"

    @classmethod
    def availability(cls, settings: Settings | None = None) -> tuple[bool, str]:
        return False, "缺模型"


def _factory_for(engine_class: type[PoseEngine]):
    def factory(**kwargs):
        return engine_class(**kwargs)

    factory.engine_class = engine_class
    return factory


@pytest.fixture()
def patched_registry(monkeypatch: pytest.MonkeyPatch):
    """把内置引擎换成可控假引擎，避免单测依赖 mediapipe / onnx 权重。"""
    monkeypatch.setattr(registry, "_FACTORIES", {}, raising=True)
    monkeypatch.setattr(registry, "_ORDER", {}, raising=True)
    registry.register_engine("onnx", _factory_for(_BrokenEngine), priority=10)
    registry.register_engine("mediapipe", _factory_for(_OkEngine), priority=20)
    yield
    registry._FACTORIES.clear()
    registry._ORDER.clear()


class _Plan:
    gpu_accelerated = True
    model_tier = "full"
    device_name = "fake-gpu"
    threads = 2
    execution_providers = ["CPUExecutionProvider"]


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models", log_level="ERROR")


# --------------------------------------------------------------------------- #
# 4.1 抽象层与选型
# --------------------------------------------------------------------------- #
def test_pose_frame_array_roundtrip() -> None:
    array = np.arange(LANDMARK_COUNT * 4, dtype=np.float32).reshape(LANDMARK_COUNT, 4)
    frame = PoseFrame.from_array(array, index=3, timestamp_ms=99.0)
    assert len(frame.landmarks) == LANDMARK_COUNT
    assert np.allclose(frame.as_array(), array)
    assert frame.timestamp_s == pytest.approx(0.099)


def test_bbox_iou_and_expand() -> None:
    a = BoundingBox(x=0.0, y=0.0, width=0.4, height=0.4)
    b = BoundingBox(x=0.2, y=0.0, width=0.4, height=0.4)
    assert a.iou(b) == pytest.approx(1 / 3, rel=1e-6)
    expanded = a.expanded(2.0)
    assert expanded.width == pytest.approx(0.8)
    assert expanded.area >= a.area


def test_select_engine_degrades_when_preferred_fails(patched_registry, tmp_path) -> None:
    engine, selection = registry.select_engine(
        _Plan(), settings=_settings(tmp_path), preferred="onnx"
    )
    try:
        assert isinstance(engine, _OkEngine)
        assert selection.engine == "ok"
        assert selection.degraded is True
        assert any("onnx" in reason for reason in selection.reasons)
    finally:
        engine.close()


def test_select_engine_reports_all_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(registry, "_FACTORIES", {}, raising=True)
    monkeypatch.setattr(registry, "_ORDER", {}, raising=True)
    registry.register_engine("onnx", _factory_for(_BrokenEngine), priority=10)
    registry.register_engine("mediapipe", _factory_for(_MissingEngine), priority=20)
    with pytest.raises(EngineUnavailableError) as info:
        registry.select_engine(_Plan(), settings=_settings(tmp_path), preferred="auto")
    reasons = info.value.details["reasons"]
    assert len(reasons) == 2
    assert any("缺模型" in item for item in reasons)


def test_select_engine_uses_onnx_first_on_gpu(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(registry, "_FACTORIES", {}, raising=True)
    monkeypatch.setattr(registry, "_ORDER", {}, raising=True)
    registry.register_engine("onnx", _factory_for(_OkEngine), priority=10)
    registry.register_engine("mediapipe", _factory_for(_MissingEngine), priority=20)
    engine, selection = registry.select_engine(_Plan(), settings=_settings(tmp_path))
    engine.close()
    assert selection.engine == "ok"
    assert selection.degraded is False


def test_unavailable_engines_raise_app_error() -> None:
    from app.backend.engine.mediapipe_engine import MediaPipePoseEngine

    available, reason = MediaPipePoseEngine.availability()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    with pytest.raises(AppError):
        MediaPipePoseEngine().prepare()


# --------------------------------------------------------------------------- #
# 4.3 ONNX 预处理 / 解码
# --------------------------------------------------------------------------- #
def test_crop_to_input_maps_back_to_original_pixels() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    crop, matrix_inv = crop_to_input(image, (0.25, 0.25, 0.5, 0.5), (192, 256))
    assert crop.shape == (256, 192, 3)

    # 裁剪图中心 → 原图中心（人物框中心 320, 240）
    center = np.array([[96.0, 128.0, 1.0]]) @ matrix_inv.T
    assert center[0, 0] == pytest.approx(320.0, abs=1.0)
    assert center[0, 1] == pytest.approx(240.0, abs=1.0)


def test_preprocess_shape_and_normalization() -> None:
    crop = np.full((256, 192, 3), 128, dtype=np.uint8)
    tensor = preprocess(crop)
    assert tensor.shape == (1, 3, 256, 192)
    assert tensor.dtype == np.float32


def test_decode_simcc_returns_peak_coordinates() -> None:
    x_logits = np.zeros((1, 17, 384), dtype=np.float32)
    y_logits = np.zeros((1, 17, 512), dtype=np.float32)
    x_logits[0, :, 100] = 0.9
    y_logits[0, :, 200] = 0.8
    (coords, scores), = decode_simcc(x_logits, y_logits)
    assert coords.shape == (17, 2)
    assert coords[0, 0] == pytest.approx(50.0)  # 100 / split_ratio 2
    assert coords[0, 1] == pytest.approx(100.0)
    assert scores[0] == pytest.approx(0.8)


def test_decode_outputs_by_names_and_shapes() -> None:
    x_logits = np.random.default_rng(0).random((1, 17, 384)).astype(np.float32)
    y_logits = np.random.default_rng(1).random((1, 17, 512)).astype(np.float32)
    kind, persons = decode_outputs(
        [y_logits, x_logits], (192, 256), names=["simcc_y", "simcc_x"]
    )
    assert kind == "simcc"
    assert len(persons) == 1

    kind, persons = decode_outputs([y_logits, x_logits], (192, 256))
    assert kind == "simcc"
    assert len(persons) == 1


def test_coco17_mapping_covers_core_joints() -> None:
    assert COCO17_TO_MP33[0] == 0
    assert COCO17_TO_MP33[5] == 11
    assert COCO17_TO_MP33[16] == 28
    assert len(set(COCO17_TO_MP33.values())) == 17


# --------------------------------------------------------------------------- #
# 4.4 多人跟踪
# --------------------------------------------------------------------------- #
def test_tracker_keeps_ids_for_two_people() -> None:
    tracker = IouTracker(iou_threshold=0.2, max_age=5)
    left = BoundingBox(x=0.05, y=0.2, width=0.25, height=0.6)
    right = BoundingBox(x=0.6, y=0.2, width=0.25, height=0.6)

    tracker.update([make_frame(0, left), make_frame(0, right)], 0)
    moved_left = BoundingBox(x=0.1, y=0.2, width=0.25, height=0.6)
    moved_right = BoundingBox(x=0.65, y=0.2, width=0.25, height=0.6)
    frames = tracker.update([make_frame(1, moved_left), make_frame(1, moved_right)], 1)

    ids = sorted(frame.track_id for frame in frames)
    assert ids == [1, 2]


def test_tracker_drops_track_after_max_age() -> None:
    tracker = IouTracker(iou_threshold=0.2, max_age=1)
    box = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.4)
    tracker.update([make_frame(0, box)], 0)
    tracker.update([], 1)
    tracker.update([], 2)
    assert tracker.active_tracks == []


def test_primary_track_picks_longest_and_summaries() -> None:
    frames = []
    for index in range(10):
        frames.append(make_frame(index, BoundingBox(x=0.1, y=0.2, width=0.2, height=0.5)))
        if index < 4:
            frames.append(make_frame(index, BoundingBox(x=0.7, y=0.2, width=0.1, height=0.3)))
    # 手写 track_id，模拟 tracker 输出
    for frame in frames:
        frame.track_id = 1 if frame.bbox.width == pytest.approx(0.2) else 2

    summaries = summarize_tracks(frames)
    assert [item.id for item in summaries] == [1, 2]
    assert summaries[0].frames == 10
    assert primary_track_id(frames) == 1
    assert len(filter_track(frames, 2)) == 4
    assert len(drop_short_tracks(frames, min_frames=5)) == 10


# --------------------------------------------------------------------------- #
# 4.3 一致性对比
# --------------------------------------------------------------------------- #
def test_compare_frames_detects_shift() -> None:
    base = [make_frame(index, BoundingBox()) for index in range(3)]
    same = compare_frames(base, [frame.model_copy(deep=True) for frame in base])
    assert same.equivalent is True
    assert same.mean_error == pytest.approx(0.0)

    shifted = []
    for frame in base:
        copy = frame.model_copy(deep=True)
        for landmark in copy.landmarks:
            landmark.x += 0.2
        shifted.append(copy)
    report = compare_frames(base, shifted)
    assert report.equivalent is False
    assert report.mean_error == pytest.approx(0.2, abs=1e-6)
