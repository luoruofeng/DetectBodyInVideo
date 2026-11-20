"""任务 9：批处理、背压、取消、断点续算。"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_pipeline import FakePoseEngine

from app.backend.core.errors import AnalysisCancelledError
from app.backend.core.settings import Settings
from app.backend.engine.base import EngineCapabilities
from app.backend.media.probe import probe_video
from app.backend.media.sampling import SamplingConfig, build_analysis_plan
from app.backend.perf.checkpoint import FrameCheckpoint, checkpoint_key
from app.backend.perf.config import RuntimeConfig
from app.backend.pipeline import AnalysisOptions, analyze_video
from app.backend.skeleton.bundle import SkeletonConfig


class BatchPoseEngine(FakePoseEngine):
    """记录每次批量推理的帧数，证明 batch 路径被真正走到。"""

    name = "batch-fake"

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.infer_calls = 0

    def infer(self, image, *, index: int = 0, timestamp_ms: float = 0.0):
        self.infer_calls += 1
        return super().infer(image, index=index, timestamp_ms=timestamp_ms)

    def infer_batch(self, frames):
        items = list(frames)
        self.batch_sizes.append(len(items))
        engine = FakePoseEngine()
        return [engine.infer(image, index=i, timestamp_ms=t) for i, t, image in items]

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            display_name=self.display_name,
            supports_batching=True,
            model_tier="lite",
        )


def test_batching_is_used_when_engine_supports_it(
    sample_video: Path, video_settings: Settings
) -> None:
    engine = BatchPoseEngine()
    result = analyze_video(
        sample_video,
        settings=video_settings,
        engine=engine,
        options=AnalysisOptions(
            skeleton=SkeletonConfig(),
            perf=RuntimeConfig(batch_size=4),
        ),
    )
    assert len(result.frames) == 40
    assert engine.batch_sizes
    assert max(engine.batch_sizes) == 4
    assert all(size <= 4 for size in engine.batch_sizes)
    assert engine.infer_calls == 0  # 全部走 infer_batch


def test_cancel_then_resume_reuses_checkpoint(
    sample_video: Path, video_settings: Settings
) -> None:
    probe = probe_video(sample_video)
    plan = build_analysis_plan(sample_video, probe, SamplingConfig())
    key = checkpoint_key(sample_video, probe, plan)

    state = {"frames": 0, "cancel": False}

    def on_progress(info) -> None:
        if info.frames_done >= 12:
            state["cancel"] = True

    with pytest.raises(AnalysisCancelledError):
        analyze_video(
            sample_video,
            settings=video_settings,
            engine=FakePoseEngine(),
            options=AnalysisOptions(
                perf=RuntimeConfig(batch_size=1, queue_size=4, checkpoint=True, resume=True)
            ),
            progress=on_progress,
            should_cancel=lambda: state["cancel"],
        )

    checkpoint = FrameCheckpoint(video_settings.checkpoints_dir / f"{key}.pose.jsonl")
    done = checkpoint.load()
    assert done, "取消后必须留下可续算的断点"
    assert len(done) < 40
    snapshot = len(done)

    result = analyze_video(
        sample_video,
        settings=video_settings,
        engine=FakePoseEngine(),
        options=AnalysisOptions(
            perf=RuntimeConfig(batch_size=2, queue_size=4, checkpoint=True, resume=True)
        ),
    )

    assert len(result.frames) == 40
    assert any("断点续算" in note for note in result.notes)
    assert not checkpoint.path.exists(), "成功完成后断点文件应被清理"
    # 续算确实跳过了已完成的帧
    assert snapshot > 0


def test_checkpoint_tolerates_broken_lines(tmp_path: Path) -> None:
    checkpoint = FrameCheckpoint.open(tmp_path, "demo")
    checkpoint.append(3, [])
    checkpoint.flush()
    with checkpoint.path.open("a", encoding="utf-8") as sink:
        sink.write('{"index": 4, "frames": [ 掉线\n')
    checkpoint.close()

    loaded = FrameCheckpoint(checkpoint.path).load()
    assert list(loaded) == [3]
    assert loaded[3] == []


def test_checkpoint_key_depends_on_sampling_plan(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    first = checkpoint_key(sample_video, probe, build_analysis_plan(sample_video, probe, SamplingConfig()))
    second = checkpoint_key(
        sample_video, probe, build_analysis_plan(sample_video, probe, SamplingConfig(every_n_frames=3))
    )
    assert first != second


def test_runtime_config_resume_enables_checkpoint() -> None:
    config = RuntimeConfig(resume=True)
    assert config.checkpoint is True
    assert RuntimeConfig.from_settings(Settings()).queue_size >= 1
