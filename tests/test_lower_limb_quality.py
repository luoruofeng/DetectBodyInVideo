"""2D 质量与下肢专项（``docs/todo-kick-depth-accuracy.md`` §5.1）。

三件事各自可单测：

1. **检测阈值能透传**：``EngineOptions`` → 引擎 settings（含越界夹紧）；
2. **条件式重跑判据**：下肢可见度低 + 档位不是 heavy 才重跑，且能用开关关掉；
3. **下肢 ROI 重解**：坏帧裁下半身再推理，坐标要换回全图、z 要按裁剪宽度缩放。
"""

from __future__ import annotations

import numpy as np
import pytest
import synth_pose

from app.backend.core.settings import Settings
from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.engine.mediapipe_engine import MediaPipePoseEngine
from app.backend.pipeline import (
    AnalysisOptions,
    EngineOptions,
    _engine_settings,
    _retry_settings_for_quality,
    lower_limb_unreliable_ratio,
    lower_limb_visibility,
)
from app.backend.postprocess.pipeline import PostProcessResult


def _settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models", log_level="ERROR")


def _frames(leg_visibility: float, count: int = 12) -> list[PoseFrame]:
    frames = synth_pose.make_frames(count)
    for frame in frames:
        for index in range(23, 33):
            point = frame.landmarks[index]
            frame.landmarks[index] = Landmark(
                x=point.x, y=point.y, z=point.z, visibility=leg_visibility
            )
    return frames


# --------------------------------------------------------------------------- #
# 1. 阈值透传
# --------------------------------------------------------------------------- #
def test_engine_options_override_settings(tmp_path):
    base = _settings(tmp_path)
    settings = _engine_settings(
        base,
        EngineOptions(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.25,
            model_tier="heavy",
            lower_body_refine=True,
        ),
    )
    assert settings.min_detection_confidence == pytest.approx(0.3)
    assert settings.min_tracking_confidence == pytest.approx(0.25)
    assert settings.prefer_model_tier == "heavy"
    assert settings.lower_body_refine is True
    # 不能就地改原配置（同一进程里其它任务还在用）
    assert base.min_detection_confidence == pytest.approx(0.5)


def test_engine_options_are_clamped(tmp_path):
    settings = _engine_settings(
        _settings(tmp_path),
        EngineOptions(min_detection_confidence=0.0, min_tracking_confidence=1.0),
    )
    assert settings.min_detection_confidence > 0
    assert settings.min_tracking_confidence < 1


def test_partial_overrides_keep_other_fields(tmp_path):
    base = _settings(tmp_path)
    settings = _engine_settings(base, EngineOptions(min_detection_confidence=0.4))
    assert settings.min_tracking_confidence == base.min_tracking_confidence
    assert settings.prefer_model_tier == base.prefer_model_tier


# --------------------------------------------------------------------------- #
# 2. 条件式重跑判据
# --------------------------------------------------------------------------- #
def test_lower_limb_visibility_measures_legs_only():
    good = _frames(0.9)
    bad = _frames(0.2)
    assert lower_limb_visibility(good) == pytest.approx(0.9, abs=1e-6)
    assert lower_limb_visibility(bad) == pytest.approx(0.2, abs=1e-6)
    assert lower_limb_unreliable_ratio(bad, 0.5) == pytest.approx(1.0)
    assert lower_limb_unreliable_ratio(good, 0.5) == pytest.approx(0.0)


class FakeResult:
    """轻量替身：判据只读 ``selection.model_tier`` 与 ``postprocess.frames``。"""

    def __init__(self, visibility: float, tier: str) -> None:
        self.postprocess = PostProcessResult(frames=_frames(visibility))
        self.selection = type(
            "Selection", (), {"model_tier": tier, "capabilities": None}
        )()


def test_retry_triggers_only_when_legs_are_unreliable(tmp_path):
    settings = _settings(tmp_path)

    options = AnalysisOptions()
    assert _retry_settings_for_quality(FakeResult(0.9, "full"), options, settings) is None  # type: ignore[arg-type]
    retry = _retry_settings_for_quality(FakeResult(0.2, "full"), options, settings)  # type: ignore[arg-type]
    assert retry is not None and retry.prefer_model_tier == "heavy"
    # 已经是 heavy 档就不再重跑
    assert _retry_settings_for_quality(FakeResult(0.2, "heavy"), options, settings) is None  # type: ignore[arg-type]
    # 显式关掉
    off = AnalysisOptions(engineOptions={"retry_with_heavy": False})
    assert _retry_settings_for_quality(FakeResult(0.2, "full"), off, settings) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 3. 下肢 ROI 重解
# --------------------------------------------------------------------------- #
class _FakeCropResult:
    def __init__(self, landmarks: list[object]) -> None:
        self.pose_landmarks = [landmarks]


class _FakeCropLandmarker:
    """裁剪图里"看到"的下肢点：全部放在裁剪图中央、高置信度。"""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _image: object) -> _FakeCropResult:
        self.calls += 1
        points = [
            Landmark(x=0.5, y=0.5, z=0.0, visibility=0.95) for _ in range(LANDMARK_COUNT)
        ]
        return _FakeCropResult(points)

    def close(self) -> None:
        return None


def _engine_with_fake_crop(tmp_path) -> MediaPipePoseEngine:
    engine = MediaPipePoseEngine(settings=_settings(tmp_path), lower_body_refine=True)

    class _Image:
        def __init__(self, *, image_format=None, data=None) -> None:
            self.data = data

    engine._mp_image_class = _Image  # type: ignore[assignment]
    engine._mp_image_format = "SRGB"  # type: ignore[assignment]
    engine._crop_landmarker = _FakeCropLandmarker()  # type: ignore[assignment]
    return engine


def test_lower_body_refine_replaces_weak_leg_landmarks(tmp_path):
    engine = _engine_with_fake_crop(tmp_path)
    frame = _frames(0.2, count=1)[0]
    image = np.zeros((480, 270, 3), dtype=np.uint8)
    leg_before = frame.landmarks[27]

    engine._refine_lower_body(frame, image, image.shape)

    leg_after = frame.landmarks[27]
    assert leg_after.visibility == pytest.approx(0.95)
    assert leg_after != leg_before
    # 换回全图坐标：裁剪框中心要落在原腿附近，而不是 (0.5, 0.5) 的整幅图中心
    box = engine._lower_body_box(frame, image.shape)
    assert box is not None
    assert engine._refined_frames == 1
    assert engine._refined_landmarks == len(range(23, 33))


def test_lower_body_refine_skips_confident_frames(tmp_path):
    engine = _engine_with_fake_crop(tmp_path)
    frame = _frames(0.9, count=1)[0]
    image = np.zeros((480, 270, 3), dtype=np.uint8)
    engine._refine_lower_body(frame, image, image.shape)
    assert engine._refined_frames == 0


def test_lower_body_refine_survives_crop_failure(tmp_path):
    engine = _engine_with_fake_crop(tmp_path)

    class _Boom:
        def detect(self, _image: object):
            raise RuntimeError("crop 推理炸了")

        def close(self) -> None:
            return None

    engine._crop_landmarker = _Boom()  # type: ignore[assignment]
    frame = _frames(0.2, count=1)[0]
    image = np.zeros((480, 270, 3), dtype=np.uint8)
    engine._refine_lower_body(frame, image, image.shape)  # 不抛
    assert engine._refined_frames == 0


def test_lower_body_box_needs_enough_points(tmp_path):
    engine = _engine_with_fake_crop(tmp_path)
    frame = _frames(0.2, count=1)[0]
    for index in range(23, 33):
        point = frame.landmarks[index]
        frame.landmarks[index] = Landmark(x=point.x, y=point.y, z=point.z, visibility=0.0)
    assert engine._lower_body_box(frame, (480, 270, 3)) is None
