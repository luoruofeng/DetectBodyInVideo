"""视频探测 / 逐帧读取 / 采样策略测试（任务 3.2 - 3.4）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.backend.core.errors import UnsupportedMediaError, ValidationFailedError
from app.backend.media.probe import generate_thumbnail, probe_video
from app.backend.media.reader import VideoFrameReader
from app.backend.media.sampling import (
    SamplingConfig,
    build_analysis_plan,
    coarse_active_ranges,
    select_indices,
)

FPS = 10.0
FRAMES = 40


def test_probe_metadata(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    assert probe.width == 160
    assert probe.height == 120
    assert probe.display_width == 160
    assert probe.display_height == 120
    assert abs(probe.fps - FPS) < 0.5
    assert 35 <= probe.frame_count <= 41
    assert abs(probe.duration_s - probe.frame_count / probe.fps) < 0.01
    assert probe.rotation == 0
    assert probe.codec is not None
    assert probe.is_variable_frame_rate is False


def test_probe_rejects_corrupt_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is definitely not a video")
    with pytest.raises(UnsupportedMediaError):
        probe_video(broken)


def test_thumbnail_generation(sample_video: Path, tmp_path: Path) -> None:
    destination = tmp_path / "thumb.jpg"
    probe = probe_video(sample_video)
    assert generate_thumbnail(sample_video, destination, probe=probe, max_width=80) is True
    assert destination.stat().st_size > 0


def test_reader_streams_all_frames(sample_video: Path) -> None:
    with VideoFrameReader(sample_video) as reader:
        packets = list(reader.frames())
    assert 35 <= len(packets) <= 41
    assert packets[0].index == 0
    assert packets[-1].image.shape == (120, 160, 3)
    assert packets[1].timestamp_ms > packets[0].timestamp_ms


def test_reader_stride(sample_video: Path) -> None:
    with VideoFrameReader(sample_video) as reader:
        indices = [packet.index for packet in reader.frames(stride=5)]
    assert indices == [0, 5, 10, 15, 20, 25, 30, 35]


def test_reader_time_range(sample_video: Path) -> None:
    with VideoFrameReader(sample_video) as reader:
        packets = list(reader.frames(start_s=1.0, end_s=2.0))
    assert packets
    assert all(900 <= packet.timestamp_ms < 2100 for packet in packets)
    assert packets[0].index >= 9


def test_reader_target_fps_resample(sample_video: Path) -> None:
    with VideoFrameReader(sample_video) as reader:
        packets = list(reader.frames(target_fps=5.0))
    assert 17 <= len(packets) <= 22
    gaps = [b.timestamp_ms - a.timestamp_ms for a, b in zip(packets, packets[1:])]
    assert gaps and min(gaps) > 100  # 10fps → 5fps 后间隔约 200ms


def test_reader_random_access(sample_video: Path) -> None:
    with VideoFrameReader(sample_video) as reader:
        packet = reader.read_frame(5)
        assert packet is not None
        assert packet.index == 5
        assert reader.read_frame(-1) is None
        collected = [item.index for item in reader.frames_at([0, 6, 12])]
    assert collected == [0, 6, 12]


def test_select_indices_stride_and_bounds(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    baseline = select_indices(SamplingConfig(), probe)
    assert len(baseline) == probe.frame_count

    every_two = select_indices(SamplingConfig(every_n_frames=2), probe)
    assert every_two == baseline[::2]

    low_fps = select_indices(SamplingConfig(target_fps=5.0), probe)
    assert low_fps == baseline[::2]

    window = select_indices(SamplingConfig(start_s=1.0, end_s=2.0), probe)
    assert window[0] >= 9 and window[-1] < 21

    upscaled = select_indices(SamplingConfig(target_fps=60.0), probe)
    assert len(upscaled) == len(baseline)  # 不做帧复制


def test_analysis_plan_counts_frames(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    plan = build_analysis_plan(sample_video, probe, SamplingConfig(every_n_frames=4))
    assert plan.stride == 4
    assert plan.selected_frames == len(select_indices(SamplingConfig(every_n_frames=4), probe))
    assert plan.effective_fps == pytest.approx(probe.fps / 4, rel=0.05)
    assert plan.trimmed is False


def test_analysis_plan_rejects_reversed_range(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    with pytest.raises(ValidationFailedError):
        build_analysis_plan(sample_video, probe, SamplingConfig(start_s=2.0, end_s=1.0))


def test_coarse_active_ranges_runs(sample_video: Path) -> None:
    """粗筛要么返回区间列表，要么返回 None（缺少 HOG），不能抛异常。"""
    probe = probe_video(sample_video)
    result = coarse_active_ranges(sample_video, probe, sample_fps=2.0)
    assert result is None or isinstance(result, list)


def test_analysis_plan_with_trim(sample_video: Path) -> None:
    probe = probe_video(sample_video)
    plan = build_analysis_plan(sample_video, probe, SamplingConfig(trim_empty=True))
    assert plan.selected_frames >= 0
    assert isinstance(plan.warnings, list)
