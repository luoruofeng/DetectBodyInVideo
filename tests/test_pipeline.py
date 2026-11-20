"""流水线集成测试：采样 →（假）引擎 → 后处理 → 骨骼契约（任务 4/5/6）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_skeleton import STANDARD_POINTS

from app.backend.core.settings import Settings
from app.backend.engine.base import (
    LANDMARK_COUNT,
    EngineCapabilities,
    Landmark,
    PoseEngine,
    PoseFrame,
)
from app.backend.media.sampling import SamplingConfig
from app.backend.pipeline import AnalysisCancelledError, AnalysisOptions, analyze_video, summarize
from app.backend.postprocess.pipeline import PostProcessConfig
from app.backend.postprocess.smoothing import SmoothingConfig
from app.backend.skeleton.bundle import SkeletonConfig


class FakePoseEngine(PoseEngine):
    """按帧号给出稳定的站立姿态，用来验证装配层而不是模型本身。"""

    name = "fake"
    display_name = "Fake Engine"

    @classmethod
    def availability(cls, settings: Settings | None = None) -> tuple[bool, str]:
        return True, ""

    def prepare(self, device: str | None = None) -> None:
        self._prepared = True

    def infer(self, image, *, index: int = 0, timestamp_ms: float = 0.0):
        landmarks = [
            Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)
        ]
        world = [Landmark() for _ in range(LANDMARK_COUNT)]
        drift = (index % 5) * 0.002
        for landmark_index, (x, y) in STANDARD_POINTS.items():
            landmarks[landmark_index] = Landmark(
                x=x + drift, y=y, z=0.0, visibility=0.95
            )
            # 米制世界坐标（髋为原点、y 向下），与 MediaPipe 输出同构
            world[landmark_index] = Landmark(
                x=(x - 0.5) * 0.62 + drift, y=(y - 0.56) * 0.92, z=0.0, visibility=0.95
            )
        return [
            PoseFrame(
                index=index,
                timestamp_ms=timestamp_ms,
                landmarks=landmarks,
                world_landmarks=world,
                score=0.95,
                valid=True,
            )
        ]

    def close(self) -> None:
        self._prepared = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(name=self.name, display_name=self.display_name, model_tier="lite")


def test_analyze_video_end_to_end(sample_video: Path, video_settings: Settings) -> None:
    events: list[str] = []
    engine = FakePoseEngine()
    result = analyze_video(
        sample_video,
        settings=video_settings,
        engine=engine,
        options=AnalysisOptions(
            sampling=SamplingConfig(every_n_frames=2),
            postprocess=PostProcessConfig(smoothing=SmoothingConfig(method="none")),
            skeleton=SkeletonConfig(),
        ),
        progress=lambda item: events.append(item.message),
    )

    assert result.selection.engine == "fake"
    assert len(result.frames) > 0
    assert len(result.skeleton.frames) == len(result.postprocess.frames)
    assert result.skeleton.root_motion.frames
    assert result.postprocess.quality.valid_frame_ratio > 0.9
    assert events and events[-1] == "完成"

    # 采样步长 2：帧号应为 0,2,4...
    indexes = [frame.index for frame in result.postprocess.frames]
    assert indexes == sorted(indexes)
    assert indexes[1] - indexes[0] == 2

    summary = summarize(result)
    assert summary["frames"] == len(result.postprocess.frames)
    assert summary["engine"] == "fake"
    assert isinstance(summary["quality"], dict)


def test_analyze_video_can_be_cancelled(sample_video: Path, video_settings: Settings) -> None:
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    with pytest.raises(AnalysisCancelledError):
        analyze_video(
            sample_video,
            settings=video_settings,
            engine=FakePoseEngine(),
            options=AnalysisOptions(max_frames=100),
            should_cancel=should_cancel,
        )


def test_analyze_video_respects_max_frames(sample_video: Path, video_settings: Settings) -> None:
    result = analyze_video(
        sample_video,
        settings=video_settings,
        engine=FakePoseEngine(),
        options=AnalysisOptions(max_frames=4, skeleton=SkeletonConfig()),
    )
    assert len(result.frames) <= 4
    assert any("max_frames" in note for note in result.notes)


def test_analyze_video_emits_preview_and_stage_progress(
    sample_video: Path, video_settings: Settings
) -> None:
    previews: list[tuple[int, int]] = []
    progress: list = []
    analyze_video(
        sample_video,
        settings=video_settings,
        engine=FakePoseEngine(),
        options=AnalysisOptions(
            sampling=SamplingConfig(every_n_frames=2),
            postprocess=PostProcessConfig(smoothing=SmoothingConfig(method="none")),
            skeleton=SkeletonConfig(),
        ),
        progress=progress.append,
        on_inference_done=lambda frames, skeleton: previews.append(
            (len(frames), len(skeleton.frames))
        ),
    )

    assert previews and previews[0][0] == previews[0][1] > 0
    # 推理阶段的 frames_done 已满，但后处理阶段必须单独上报，不能停在 "后处理 100%"
    assert any(item.stage == "ik_fit" for item in progress)
    assert any(item.stage == "done" for item in progress)
    assert all(item.eta_seconds is None for item in progress if item.stage not in {"inference", "done"})
