"""动作 JSON（iOS App 格式）转换：字段契约、重采样、骨盆位移、写盘自校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backend.export.motion_clip import (
    BODY_JOINTS,
    MOTION_SCHEMA_VERSION,
    MotionClipConfig,
    MotionClipDocument,
    build_motion_clip,
    motion_filename,
    read_motion_clip,
    write_motion_clip,
)
from app.backend.export.samples import build_sample_document
from app.backend.export.schema import (
    COORDINATE_SYSTEM,
    GENERATOR,
    SCHEMA_VERSION,
    UNITS,
    ExportDocument,
    ExportFrame,
    JointOut,
    QualityOut,
    RootMotionFrameOut,
    RootMotionOut,
    SourceInfo,
)
from app.backend.skeleton.mapping import ARKIT_JOINTS
from app.backend.skeleton.root_motion import Quaternion
from app.backend.skeleton.transform import Vector3

RESOURCES = Path(__file__).resolve().parents[1] / "resources" / "air-squat-motion.json"
REFERENCE_MODEL = "t-pose_humanbody_2048.usdz"


# --------------------------------------------------------------------------- #
# 手工构造文档（不依赖流水线）
# --------------------------------------------------------------------------- #
def _layout(
    *,
    ankle_y: float = -0.82,
    knee_y: float = -0.45,
    hip_y: float = -0.06,
) -> dict[str, tuple[float, float, float]]:
    """简化骨架：骨盆在 (0,0)，其余关节按给定高度摆放。"""
    return {
        "root": (0.0, 0.0, 0.0),
        "hips_joint": (0.0, 0.0, 0.0),
        "spine_3_joint": (0.0, 0.22, 0.0),
        "spine_7_joint": (0.0, 0.47, 0.0),
        "neck_1_joint": (0.0, 0.50, 0.0),
        "head_joint": (0.0, 0.62, 0.0),
        "left_shoulder_1_joint": (0.17, 0.32, 0.0),
        "left_arm_joint": (0.17, 0.10, 0.0),
        "left_hand_joint": (0.17, -0.05, 0.0),
        "right_shoulder_1_joint": (-0.17, 0.32, 0.0),
        "right_arm_joint": (-0.17, 0.10, 0.0),
        "right_hand_joint": (-0.17, -0.05, 0.0),
        "left_upLeg_joint": (0.10, hip_y, 0.0),
        "left_leg_joint": (0.12, knee_y, 0.0),
        "left_foot_joint": (0.15, ankle_y, 0.0),
        "right_upLeg_joint": (-0.10, hip_y, 0.0),
        "right_leg_joint": (-0.12, knee_y, 0.0),
        "right_foot_joint": (-0.15, ankle_y, 0.0),
    }


def _frame(
    index: int,
    t: float,
    layout: dict[str, tuple[float, float, float]],
    *,
    valid: bool = True,
    visibility: float = 0.9,
) -> ExportFrame:
    return ExportFrame(
        index=index,
        t=t,
        valid=valid,
        joints={
            name: JointOut(x=x, y=y, z=z, visibility=visibility)
            for name, (x, y, z) in layout.items()
        },
    )


def _document(
    frames: list[ExportFrame],
    *,
    root_motion: RootMotionOut | None = None,
    file: str = "squat.mp4",
) -> ExportDocument:
    return ExportDocument(
        version=SCHEMA_VERSION,
        generator=GENERATOR,
        source=SourceInfo(file=file, fps=30, width=1080, height=1920, frames=len(frames)),
        coordinateSystem=COORDINATE_SYSTEM,
        units=UNITS,
        personHeightMeters=1.70,
        jointNames=list(ARKIT_JOINTS),
        frames=frames,
        quality=QualityOut(),
        rootMotion=root_motion,
    )


# --------------------------------------------------------------------------- #
# 字段契约
# --------------------------------------------------------------------------- #
def test_motion_clip_top_level_matches_app_format() -> None:
    clip = build_motion_clip(build_sample_document(), MotionClipConfig())

    assert clip.schemaVersion == MOTION_SCHEMA_VERSION == 1
    assert clip.coordinateSpace == "hip-local-meters-yup"
    assert clip.bodyOrientation == "upright"
    assert clip.referenceModel == REFERENCE_MODEL
    assert clip.mirroredReference is True
    assert clip.sampleIntervalMs == 100
    assert clip.skeleton == list(BODY_JOINTS)
    assert clip.playbackHolds == []
    assert clip.frames and clip.durationMs == clip.frames[-1].t
    assert clip.keyframesMs == [0, clip.durationMs]


def test_motion_clip_joints_are_pelvis_local_and_complete() -> None:
    clip = build_motion_clip(build_sample_document(), MotionClipConfig())

    for frame in clip.frames:
        assert list(frame.joints) == list(BODY_JOINTS)
        assert frame.joints["root"] == [0.0, 0.0, 0.0]
        assert frame.joints["pelvis"] == [0.0, 0.0, 0.0]
        # 髋局部 + Y 向上：膝盖必须在髋下方
        assert frame.joints["leftKnee"][1] < frame.joints["leftHip"][1]
        assert len(frame.joints["leftWrist"]) == 3


def test_motion_clip_resamples_to_10hz() -> None:
    clip = build_motion_clip(build_sample_document(), MotionClipConfig(sample_interval_ms=100))
    stamps = [frame.t for frame in clip.frames]
    assert stamps[0] == 0
    assert all(b - a == 100 for a, b in zip(stamps, stamps[1:], strict=False))
    # 样例是 30 帧 @30fps ≈ 0.967s → 0/100/.../900
    assert clip.durationMs == 900


def test_joint_visibility_only_lists_low_confidence() -> None:
    clip = build_motion_clip(build_sample_document(), MotionClipConfig())
    for frame in clip.frames:
        assert all(value < 1.0 for value in frame.jointVisibility.values())
        assert set(frame.jointVisibility) <= set(BODY_JOINTS)


def test_origin_is_subtracted_even_if_source_is_offset() -> None:
    layout = _layout()
    offset_layout = {
        name: (x + 0.5, y + 1.2, z - 0.3) for name, (x, y, z) in layout.items()
    }
    document = _document(
        [
            _frame(0, 0.0, offset_layout),
            _frame(1, 0.1, offset_layout),
        ]
    )
    clip = build_motion_clip(document, MotionClipConfig(sample_interval_ms=100))
    first = clip.frames[0]

    assert first.joints["pelvis"] == [0.0, 0.0, 0.0]
    assert first.joints["leftShoulder"] == pytest.approx([0.17, 0.32, 0.0], abs=1e-4)
    assert first.joints["leftAnkle"][1] == pytest.approx(-0.82, abs=1e-4)


# --------------------------------------------------------------------------- #
# 骨盆升降
# --------------------------------------------------------------------------- #
def test_pelvis_offset_uses_leg_span_when_only_world_landmarks() -> None:
    standing = _layout(ankle_y=-0.82)
    squat = _layout(ankle_y=-0.44, knee_y=-0.16, hip_y=-0.05)
    document = _document(
        [
            _frame(0, 0.0, standing),
            _frame(1, 0.1, squat),
            _frame(2, 0.2, squat),
        ]
    )
    clip = build_motion_clip(document, MotionClipConfig(sample_interval_ms=100))

    assert clip.frames[0].pelvisOffsetMeters == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert clip.frames[1].pelvisOffsetMeters[1] == pytest.approx(-0.38, abs=1e-3)
    assert clip.frames[1].pelvisOffsetMeters[0] == 0.0


def test_pelvis_offset_prefers_root_motion_when_available() -> None:
    layout = _layout()
    root_motion = RootMotionOut(
        mode="world",
        totalTranslation=0.44,
        frames=[
            RootMotionFrameOut(
                index=0, t=0.0, translation=Vector3(x=0.0, y=0.0, z=0.0), rotation=Quaternion()
            ),
            RootMotionFrameOut(
                index=1,
                t=0.1,
                translation=Vector3(x=0.01, y=-0.38, z=0.22),
                rotation=Quaternion(),
            ),
        ],
    )
    document = _document(
        [_frame(0, 0.0, layout), _frame(1, 0.1, layout)],
        root_motion=root_motion,
    )
    clip = build_motion_clip(document, MotionClipConfig(sample_interval_ms=100))

    assert clip.frames[0].pelvisOffsetMeters == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert clip.frames[1].pelvisOffsetMeters == pytest.approx([0.01, -0.38, 0.22], abs=1e-4)


# --------------------------------------------------------------------------- #
# 插值与无效帧
# --------------------------------------------------------------------------- #
def test_invalid_frames_are_not_interpolated() -> None:
    low = _layout(ankle_y=-0.82)
    high = _layout(ankle_y=-0.44, knee_y=-0.16, hip_y=-0.05)
    document = _document(
        [
            _frame(0, 0.0, low),
            _frame(1, 0.1, high, valid=False),
            _frame(2, 0.2, low),
        ]
    )
    clip = build_motion_clip(document, MotionClipConfig(sample_interval_ms=100))

    # 无效帧被跳过，t=100ms 处仍是从 low 插值到 low 的结果
    assert len(clip.frames) == 3
    assert clip.frames[1].pelvisOffsetMeters[1] == pytest.approx(0.0, abs=1e-4)
    assert clip.frames[1].joints["leftKnee"][1] == pytest.approx(-0.45, abs=1e-4)


def test_long_gap_holds_previous_pose_instead_of_blending() -> None:
    low = _layout(ankle_y=-0.82)
    high = _layout(ankle_y=-0.44, knee_y=-0.16, hip_y=-0.05)
    document = _document(
        [
            _frame(0, 0.0, low),
            _frame(1, 1.0, high),
        ]
    )
    clip = build_motion_clip(
        document, MotionClipConfig(sample_interval_ms=100, hold_gap_ms=400)
    )

    # 间隔 1000ms > hold_gap_ms=400：中间采样保持前一帧，不做线性混合
    assert clip.frames[4].joints["leftKnee"] == pytest.approx([0.12, -0.45, 0.0], abs=1e-4)
    assert clip.frames[-1].joints["leftKnee"] == pytest.approx([0.12, -0.16, 0.0], abs=1e-4)
    assert clip.frames[-1].t == 1000


# --------------------------------------------------------------------------- #
# 写盘 / 回读 / CLI
# --------------------------------------------------------------------------- #
def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    clip = build_motion_clip(build_sample_document(), MotionClipConfig(loop=True, name_zh="挥手动"))
    target = tmp_path / motion_filename("我的 视频.mp4")
    artifact = write_motion_clip(clip, target)

    assert artifact.filename == target.name
    assert artifact.mode == "motion"
    assert artifact.validated is True
    assert artifact.frames == len(clip.frames)

    reloaded = read_motion_clip(target)
    assert isinstance(reloaded, MotionClipDocument)
    assert reloaded.loop is True
    assert reloaded.nameZh == "挥手动"
    assert len(reloaded.frames) == len(clip.frames)


def test_invalid_motion_file_raises(tmp_path: Path) -> None:
    from app.backend.export.writer import ExportValidationError

    broken = tmp_path / "broken_motion.json"
    broken.write_text('{"schemaVersion": 1, "frames": []}', encoding="utf-8")
    with pytest.raises(ExportValidationError):
        read_motion_clip(broken)


def test_cli_converts_pose_json(tmp_path: Path) -> None:
    from app.backend.export.motion_clip import main
    from app.backend.export.writer import write_document

    pose_path = tmp_path / "clip_posefull.json"
    write_document(build_sample_document(), pose_path, mode="full")
    output = tmp_path / "clip_motion.json"

    assert main([str(pose_path), "-o", str(output), "--name", "Wave"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coordinateSpace"] == "hip-local-meters-yup"
    assert payload["exerciseID"] == "wave"
    assert len(payload["frames"]) >= 1


# --------------------------------------------------------------------------- #
# 与 App 自带样例对齐
# --------------------------------------------------------------------------- #
def test_fields_are_subset_of_app_reference_sample() -> None:
    if not RESOURCES.is_file():
        pytest.skip("resources/air-squat-motion.json 未放置（已在 .gitignore 中忽略）")
    reference = json.loads(RESOURCES.read_text(encoding="utf-8"))
    clip = build_motion_clip(build_sample_document(), MotionClipConfig(name_zh="测试动作"))
    payload = clip.model_dump(mode="json")

    missing = set(payload) - set(reference)
    assert not missing, f"动作 JSON 出现了 App 样例里没有的顶层字段：{sorted(missing)}"
    unknown = set(payload["skeleton"]) - set(reference["skeleton"])
    assert not unknown, f"关节名不在 App 骨架里：{sorted(unknown)}"
    assert set(reference["frames"][0]["joints"]) >= set(payload["frames"][0]["joints"])
