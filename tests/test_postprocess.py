"""后处理单测：平滑、补帧、角度、质检（任务 5）。"""

from __future__ import annotations

import numpy as np
import pytest

from app.backend.core.errors import AnalysisCancelledError
from app.backend.engine.base import (
    LANDMARK_COUNT,
    Landmark,
    PoseFrame,
)
from app.backend.postprocess.angles import angle_deg, compute_angles, torso_lean
from app.backend.postprocess.gaps import GapFillConfig, densify, process_gaps
from app.backend.postprocess.one_euro import OneEuroFilter1D, OneEuroParams, smooth_array
from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess
from app.backend.postprocess.quality import QualityConfig, build_quality_report
from app.backend.postprocess.smoothing import (
    SmoothingConfig,
    moving_average,
    savitzky_golay,
    smooth_frames,
)


def make_frame(
    index: int,
    *,
    points: dict[int, tuple[float, float, float]] | None = None,
    score: float = 0.9,
    visibility: float = 0.9,
    track_id: int | None = None,
) -> PoseFrame:
    landmarks = [
        Landmark(x=0.5, y=0.5, z=0.0, visibility=visibility) for _ in range(LANDMARK_COUNT)
    ]
    for landmark_index, (x, y, z) in (points or {}).items():
        landmarks[landmark_index] = Landmark(x=x, y=y, z=z, visibility=visibility)
    return PoseFrame(
        index=index,
        timestamp_ms=index * 33.0,
        landmarks=landmarks,
        score=score,
        track_id=track_id,
    )


# --------------------------------------------------------------------------- #
# 5.1 平滑
# --------------------------------------------------------------------------- #
def test_one_euro_keeps_constant_signal() -> None:
    flt = OneEuroFilter1D(OneEuroParams())
    outputs = [flt.filter(0.42, 1 / 30) for _ in range(20)]
    assert outputs[0] == pytest.approx(0.42)
    assert outputs[-1] == pytest.approx(0.42, abs=1e-6)


def test_one_euro_reduces_jitter() -> None:
    rng = np.random.default_rng(7)
    values = 0.5 + rng.normal(0, 0.02, size=60)
    flt = OneEuroFilter1D(OneEuroParams(min_cutoff=1.0, beta=0.5))
    smoothed = [flt.filter(float(v), 1 / 30) for v in values]
    raw_jitter = np.mean(np.abs(np.diff(values)))
    smooth_jitter = np.mean(np.abs(np.diff(smoothed)))
    assert smooth_jitter < raw_jitter


def test_smooth_array_keeps_visibility_column() -> None:
    array = np.zeros((5, LANDMARK_COUNT, 4), dtype=np.float32)
    array[:, :, 0] = np.linspace(0.4, 0.6, 5)[:, None]
    array[:, :, 3] = 0.8
    smoothed = smooth_array(array, [i * 33.0 for i in range(5)], OneEuroParams())
    assert np.allclose(smoothed[:, :, 3], 0.8)
    assert smoothed.shape == array.shape


def test_moving_average_and_savgol() -> None:
    constant = np.full(10, 3.0)
    assert np.allclose(moving_average(constant, 5), 3.0)

    ramp = np.arange(10, dtype=np.float64)
    # 二次多项式在 polyorder=2 的 SG 内部应被精确还原
    quadratic = ramp**2
    result = savitzky_golay(quadratic, 5, 2)
    assert np.allclose(result[2:-2], quadratic[2:-2], atol=1e-3)


def test_smooth_frames_records_filter_params() -> None:
    frames = [make_frame(index) for index in range(8)]
    smoothed, info = smooth_frames(frames, SmoothingConfig(method="one-euro"))
    assert info.applied is True
    assert info.method == "one-euro"
    assert info.params["minCutoff"] == pytest.approx(1.0)
    assert len(smoothed) == len(frames)
    # 原始帧不被就地修改
    assert frames[0].landmarks[0].x == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 5.2 缺失帧
# --------------------------------------------------------------------------- #
def test_short_gap_is_interpolated_long_gap_is_not() -> None:
    frames = []
    for index in range(12):
        if index in {3, 4} or index >= 9:
            frames.append(make_frame(index, score=0.1))
        else:
            frames.append(make_frame(index))

    filled, report = process_gaps(frames, GapFillConfig(max_gap_frames=3))
    assert report.frames_filled == 2
    assert filled[3].interpolated is True
    assert filled[3].valid is True
    assert filled[3].landmarks and filled[3].landmarks[0].x == pytest.approx(0.5)
    # 9~11 是长缺口（3 帧 > max 2），保持无效且不写坐标
    assert report.frames_missing >= 3
    assert filled[10].valid is False


def test_short_gap_also_interpolates_world_landmarks() -> None:
    """补帧必须同时补 world_landmarks。

    只要有一个有效帧缺世界坐标，导出链路就会把**整条人物轨迹**从 world 模式降级到
    归一化标定模式（``skeleton/transform.py``），骨长与尺度因此换一套算法，
    播放时表现为"某个肢体突然变长变短"。
    """

    def frame(index: int, world_z: float) -> PoseFrame:
        return PoseFrame(
            index=index,
            timestamp_ms=index * 33.0,
            landmarks=[
                Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(LANDMARK_COUNT)
            ],
            world_landmarks=[
                Landmark(x=0.1, y=0.2, z=world_z, visibility=0.9)
                for _ in range(LANDMARK_COUNT)
            ],
            score=0.9,
        )

    frames = [frame(0, 0.0), frame(1, 0.0), frame(2, 0.2)]
    frames[1].landmarks = []  # 第 1 帧观测丢失（只剩两端有效）
    frames[1].world_landmarks = []

    filled, report = process_gaps(frames, GapFillConfig(max_gap_frames=2))
    assert report.frames_filled == 1
    assert filled[1].valid is True
    assert len(filled[1].world_landmarks) == LANDMARK_COUNT
    assert filled[1].world_landmarks[0].z == pytest.approx(0.1, abs=1e-6)


def test_densify_inserts_placeholders() -> None:
    frames = [make_frame(0), make_frame(2), make_frame(4)]
    dense = densify(frames, stride=2, fps=30.0)
    assert [frame.index for frame in dense] == [0, 2, 4]
    assert all(frame.valid for frame in dense)

    sparse = [make_frame(0), make_frame(4)]
    dense = densify(sparse, stride=2, fps=30.0)
    assert [frame.index for frame in dense] == [0, 2, 4]
    assert dense[1].landmarks == []
    assert dense[1].timestamp_ms == pytest.approx(66.0)


# --------------------------------------------------------------------------- #
# 5.3 角度
# --------------------------------------------------------------------------- #
def test_angle_deg_handles_right_and_degenerate_cases() -> None:
    vertex = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)
    up = Landmark(x=0.5, y=0.3, z=0.0, visibility=0.9)
    right = Landmark(x=0.7, y=0.5, z=0.0, visibility=0.9)
    assert angle_deg(up, vertex, right) == pytest.approx(90.0)

    opposite = Landmark(x=0.5, y=0.7, z=0.0, visibility=0.9)
    assert angle_deg(up, vertex, opposite) == pytest.approx(180.0)

    # 零长度向量不产生 NaN，而是 None
    assert angle_deg(vertex, vertex, right) is None
    # 非有限坐标
    bad = Landmark(x=float("nan"), y=0.1, z=0.0, visibility=0.9)
    assert angle_deg(bad, vertex, right) is None
    # 置信度不足
    assert angle_deg(up, vertex, right, min_visibility=0.95) is None


def test_compute_angles_for_straight_arm() -> None:
    frame = make_frame(
        0,
        points={
            11: (0.5, 0.3, 0.0),  # left_shoulder
            13: (0.5, 0.5, 0.0),  # left_elbow
            15: (0.7, 0.5, 0.0),  # left_wrist
        },
    )
    angles = compute_angles(frame)
    assert angles["left_elbow"] == pytest.approx(90.0, abs=1e-6)


def test_torso_lean_upright_vs_horizontal() -> None:
    hips = Landmark(x=0.5, y=0.7, z=0.0, visibility=0.9)
    shoulders = Landmark(x=0.5, y=0.4, z=0.0, visibility=0.9)
    assert torso_lean(shoulders, hips) == pytest.approx(0.0, abs=1e-6)

    horizontal = Landmark(x=0.8, y=0.7, z=0.0, visibility=0.9)
    assert torso_lean(horizontal, hips) == pytest.approx(90.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 5.4 质检
# --------------------------------------------------------------------------- #
def test_quality_verdict_ready_and_rerecord() -> None:
    good = [make_frame(index) for index in range(10)]
    report = build_quality_report(good, QualityConfig())
    assert report.valid_frame_ratio == pytest.approx(1.0)
    assert report.verdict == "ready"
    assert report.verdict_label == "可直接使用"

    bad = [make_frame(index, score=0.1) for index in range(10)]
    for frame in bad:
        frame.valid = False
    report = build_quality_report(bad, QualityConfig())
    assert report.verdict == "rerecord"
    assert report.issues


def test_quality_detects_big_jump() -> None:
    frames = [make_frame(index) for index in range(6)]
    moved = make_frame(6, points={23: (0.95, 0.5, 0.0), 24: (0.95, 0.5, 0.0)})
    frames.append(moved)
    report = build_quality_report(frames, QualityConfig())
    assert report.max_jump > 0.4
    assert report.max_jump_frame == 6
    assert report.verdict != "ready"


# --------------------------------------------------------------------------- #
# 深度专项指标（任务 5.3）
# --------------------------------------------------------------------------- #
def _world_frames(z_sigma: float, count: int = 24) -> list[PoseFrame]:
    """合成一段带 z 噪声的序列（2D 不动，只有世界坐标抖动）。"""
    import synth_pose

    return synth_pose.add_depth_noise(
        synth_pose.make_frames(count), z_sigma=z_sigma, xy_sigma=0.002
    )


def test_quality_reports_depth_metrics() -> None:
    report = build_quality_report(_world_frames(0.002))
    assert report.z_jitter_rms >= 0
    assert 0.0 <= report.bone_length_cv <= 1.0
    assert 0.0 <= report.depth_confidence <= 1.0
    assert any("深度可信度" in note for note in report.notes)


def test_dirty_depth_lowers_confidence_and_verdict() -> None:
    """z 抖动很大时可信度下降，并把 ready 降级为 review + 给出重拍建议。"""
    clean = build_quality_report(_world_frames(0.001))
    dirty = build_quality_report(_world_frames(0.05))
    assert dirty.depth_confidence < clean.depth_confidence
    assert dirty.depth_confidence < 0.6
    assert dirty.z_jitter_rms > clean.z_jitter_rms
    assert dirty.verdict != "ready"
    assert any("深度" in issue for issue in dirty.issues)
    assert any("侧面" in issue for issue in dirty.issues)


def test_no_world_landmarks_does_not_downgrade_verdict() -> None:
    """引擎没有世界坐标时不做深度降级（否则所有结果都会变成"需复核"）。"""
    report = build_quality_report([make_frame(index) for index in range(10)])
    assert report.depth_confidence == 0.0
    assert report.verdict == "ready"
    assert any("世界坐标" in issue for issue in report.issues)


# --------------------------------------------------------------------------- #
# 拍摄质量规则（任务 6.2）
# --------------------------------------------------------------------------- #
def _clean_world_frames(count: int = 12) -> list[PoseFrame]:
    """直立人物的 33 点（归一化 y 向下 + 米制世界坐标），用于拍摄质量规则。

    没有复用 `synth_pose`：那套合成骨架的四肢/腿部 y 方向是"反的"（脚比头还高），
    对骨长/重投影那些只看距离的指标没影响，但"人物占画面多高"会被算成 4%。
    """
    # 归一化 x、y（y 向下）；世界宽度（米）用于把像素宽度换算成尺度
    layout: dict[int, tuple[float, float]] = {}
    layout[0] = (0.50, 0.14)  # nose
    for index, dx in ((1, 0.012), (2, 0.016), (3, 0.020), (4, -0.012), (5, -0.016), (6, -0.020)):
        layout[index] = (0.50 + dx, 0.135)
    layout[7] = (0.534, 0.145)  # left ear
    layout[8] = (0.466, 0.145)
    layout[9] = (0.51, 0.175)
    layout[10] = (0.49, 0.175)
    layout[11] = (0.569, 0.32)  # left shoulder
    layout[12] = (0.431, 0.32)
    layout[13] = (0.592, 0.44)  # left elbow
    layout[14] = (0.408, 0.44)
    layout[15] = (0.598, 0.56)  # left wrist
    layout[16] = (0.402, 0.56)
    for point, dx in ((17, 0.004), (19, 0.006), (21, -0.004)):
        layout[point] = (0.598 + dx, 0.60)
    for point, dx in ((18, -0.004), (20, -0.006), (22, 0.004)):
        layout[point] = (0.402 + dx, 0.60)
    layout[23] = (0.541, 0.55)  # left hip
    layout[24] = (0.459, 0.55)
    layout[25] = (0.541, 0.74)  # left knee
    layout[26] = (0.459, 0.74)
    layout[27] = (0.541, 0.90)  # left ankle
    layout[28] = (0.459, 0.90)
    layout[29] = (0.541, 0.925)
    layout[30] = (0.459, 0.925)
    layout[31] = (0.541, 0.945)
    layout[32] = (0.459, 0.945)

    frames: list[PoseFrame] = []
    for index in range(count):
        landmarks: list[Landmark] = []
        world: list[Landmark] = []
        for point in range(LANDMARK_COUNT):
            x, y = layout[point]
            landmarks.append(Landmark(x=x, y=y, z=0.0, visibility=0.95))
            # 世界坐标：按"1 米 ≈ 0.45 归一化高度"反推，y 向下
            world.append(Landmark(x=(x - 0.5) / 0.45, y=(y - 0.5) / 0.45, z=0.0, visibility=0.95))
        frames.append(
            PoseFrame(
                index=index,
                timestamp_ms=index * 33.0,
                landmarks=landmarks,
                world_landmarks=world,
                score=0.95,
                valid=True,
            )
        )
    return frames


def _scale_about_center(frames: list[PoseFrame], factor: float) -> list[PoseFrame]:
    """以画面中心为原点缩放归一化坐标（模拟走近 / 远离）。"""
    for frame in frames:
        for landmark in frame.landmarks:
            landmark.x = 0.5 + (landmark.x - 0.5) * factor
            landmark.y = 0.5 + (landmark.y - 0.5) * factor
    return frames


def test_framing_reports_clean_shot_as_ok() -> None:
    report = build_quality_report(_clean_world_frames())
    assert report.framing.subject_height > 0.5
    assert 0.8 < report.framing.perspective_ratio < 1.25
    assert not any("机位" in issue or "离镜头" in issue for issue in report.issues)


def test_far_subject_triggers_too_far_hint() -> None:
    frames = _scale_about_center(_clean_world_frames(), 0.45)
    report = build_quality_report(frames)
    assert report.framing.subject_height < 0.5
    assert any("离镜头太远" in issue for issue in report.issues)


def test_high_camera_triggers_perspective_hint() -> None:
    """把头部宽度放大 = 头比脚更靠近镜头，等价于机位偏高/俯拍。"""
    frames = _clean_world_frames()
    for frame in frames:
        left, right = frame.landmarks[7], frame.landmarks[8]
        center = 0.5 * (left.x + right.x)
        for landmark in (left, right):
            landmark.x = center + (landmark.x - center) * 1.8
    report = build_quality_report(frames)
    assert report.framing.perspective_ratio > 1.2
    assert any("机位偏高" in issue for issue in report.issues)


def test_approaching_subject_triggers_depth_hint() -> None:
    """每帧放大 3% = 快速朝镜头走近，应提示前后位移不可信。"""
    frames = _clean_world_frames()
    for index, frame in enumerate(frames):
        factor = 1.03**index
        for landmark in frame.landmarks:
            landmark.x = 0.5 + (landmark.x - 0.5) * factor
            landmark.y = 0.5 + (landmark.y - 0.5) * factor
    report = build_quality_report(frames)
    assert report.framing.area_change_per_frame > 0.02
    assert any("走近" in issue for issue in report.issues)


# --------------------------------------------------------------------------- #
# 流水线
# --------------------------------------------------------------------------- #
def test_run_postprocess_end_to_end() -> None:
    frames = []
    for index in range(20):
        track = 1 if index < 15 else 2
        score = 0.1 if index == 7 else 0.9
        frames.append(make_frame(index, score=score, track_id=track))

    result = run_postprocess(
        frames,
        PostProcessConfig(
            smoothing=SmoothingConfig(method="moving-average", window=5),
        ),
    )
    # 默认跟随最长 track（id=1），2 号 track 被丢弃
    assert result.target_track_id == 1
    assert all(frame.track_id == 1 for frame in result.frames)
    assert len(result.frames) == 15
    assert len(result.angles) == len(result.frames)
    assert result.gaps.frames_filled == 1
    assert result.frames[7].interpolated is True
    assert result.quality.frames_total == 15
    assert result.smoothing.method == "moving-average"
    assert result.quality.smoothing["method"] == "moving-average"


def test_run_postprocess_reports_real_stage_progress() -> None:
    """后处理不再只报一次假的 100%，每个阶段都要给出可显示的进度。"""
    from app.backend.postprocess.ik_fit import FitConfig

    events: list[tuple[str, str, int, int]] = []
    result = run_postprocess(
        [make_frame(index) for index in range(8)],
        PostProcessConfig(
            smoothing=SmoothingConfig(method="none"),
            ik_fit=FitConfig(enabled=False),
        ),
        progress=lambda *event: events.append(event),
    )

    assert len(result.frames) == 8
    stages = {event[0] for event in events}
    assert {"densify", "gaps", "smooth", "angles", "quality"} <= stages
    assert {"ik_fit"} & stages == set()  # 关闭拟合时不应上报 IK 进度
    assert all(total > 0 and 0 <= done <= total for _, _, done, total in events)
    assert ("quality", "生成质检报告", 8, 8) in events


def test_run_postprocess_honours_cancel() -> None:
    with pytest.raises(AnalysisCancelledError):
        run_postprocess(
            [make_frame(index) for index in range(5)],
            PostProcessConfig(),
            should_cancel=lambda: True,
        )
