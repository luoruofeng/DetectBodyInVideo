"""任务 7：导出 Schema / 粒度 / 流式写盘 / 自校验 / 回归样例。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.backend.export import (
    SCHEMA_FILE,
    SCHEMA_VERSION,
    ExportConfig,
    ExportDocument,
    ExportValidationError,
    build_export_document,
    export_document,
    export_filename,
    read_document,
    schema_payload,
    write_document,
)
from app.backend.export.samples import SAMPLE_PATH, build_sample_document, synthetic_pose_frames
from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess
from app.backend.skeleton.bundle import SkeletonConfig, build_skeleton_bundle


def _bundle():
    postprocess = run_postprocess(synthetic_pose_frames(12), PostProcessConfig(stride=1, fps=30))
    skeleton = build_skeleton_bundle(
        postprocess.frames, SkeletonConfig(include_optional=True, mode="world")
    )
    angles = {
        frame.index: values
        for frame, values in zip(postprocess.frames, postprocess.angles, strict=False)
    }
    return postprocess, skeleton, angles


def _document(granularity: str = "full", **kwargs) -> ExportDocument:
    postprocess, skeleton, angles = _bundle()
    from app.backend.export.schema import SourceInfo

    return build_export_document(
        source=SourceInfo(file="clip.mp4", fps=30, width=1080, height=1920, frames=12),
        skeleton=skeleton,
        quality=postprocess.quality,
        angles=angles,
        pose_frames=postprocess.frames,
        config=ExportConfig(granularity=granularity, **kwargs),
    )


def test_full_granularity_has_angles_and_optional_joints():
    document = _document("full")
    assert len(document.frames) == 12
    assert len(document.jointNames) == 22
    assert any(frame.angles for frame in document.frames)
    assert document.frames[0].joints["left_hand_joint"].visibility == pytest.approx(0.95, abs=0.01)


def test_keypoints_granularity_drops_angles():
    document = _document("keypoints")
    assert len(document.jointNames) == 22
    assert all(frame.angles == {} for frame in document.frames)


def test_compact_granularity_keeps_only_18_arkit_joints():
    document = _document("compact")
    assert len(document.jointNames) == 18
    assert all(set(frame.joints) <= set(document.jointNames) for frame in document.frames)


def test_raw_landmarks_are_opt_in():
    with_raw = _document("full", include_raw_landmarks=True)
    without_raw = _document("full")
    assert with_raw.frames[0].rawLandmarks is not None
    assert len(with_raw.frames[0].rawLandmarks or {}) == 33
    assert without_raw.frames[0].rawLandmarks is None


def test_values_rounded_to_four_decimals():
    document = _document("full")
    for frame in document.frames[:5]:
        for joint in frame.joints.values():
            for value in (joint.x, joint.y, joint.z, joint.visibility):
                assert round(value, 4) == value


def test_filename_rule():
    stamp = datetime(2026, 1, 2, 3, 4, 5)
    name = export_filename("C:/videos/我的 舞蹈.MOV", "full", timestamp=stamp)
    assert name == "我的_舞蹈_20260102-030405_posefull.json"
    assert export_filename("dance.mp4", "compact", timestamp=stamp).endswith("_posecompact.json")


def test_streaming_write_and_self_check(tmp_path):
    document = _document("full", include_raw_landmarks=True)
    target = tmp_path / "out.json"
    artifact = write_document(document, target, mode="full")

    assert artifact.validated is True
    assert artifact.frames == 12
    assert artifact.bytes == target.stat().st_size
    reloaded = read_document(target)
    assert reloaded.version == SCHEMA_VERSION
    assert len(reloaded.frames) == len(document.frames)
    assert reloaded.frames[-1].index == document.frames[-1].index


def test_export_document_naming(tmp_path):
    document = _document("compact")
    artifact = export_document(document, tmp_path, mode="compact")
    assert artifact.filename.endswith("_posecompact.json")
    assert (tmp_path / artifact.filename).is_file()


def test_invalid_file_raises_validation_error(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"version": "1.0.0", "frames": []}', encoding="utf-8")
    with pytest.raises(ExportValidationError):
        read_document(broken)


def test_schema_rejects_non_monotonic_frames():
    payload = build_sample_document().model_dump(mode="json")
    payload["frames"][1]["index"] = payload["frames"][0]["index"]
    with pytest.raises(Exception):
        ExportDocument.model_validate(payload)


def test_schema_file_is_in_sync():
    stored = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert stored == schema_payload()
    assert stored["x-schema-version"] == SCHEMA_VERSION


def test_expected_pose_sample_matches_current_pipeline():
    stored = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    regenerated = build_sample_document().model_dump(mode="json")
    assert stored == regenerated


# --------------------------------------------------------------------------- #
# 深度可信度字段（任务 5.1 / 5.4 / 5.5）
# --------------------------------------------------------------------------- #
def test_schema_version_is_minor_bump():
    """新增可选字段只升 minor：1.0.0 → 1.1.0，同 major 视为兼容。"""
    from app.backend.export.schema import is_compatible, major_version

    assert SCHEMA_VERSION == "1.1.0"
    assert major_version(SCHEMA_VERSION) == 1
    assert is_compatible("1.0.0") is True
    assert is_compatible("2.0.0") is False


def test_document_exposes_depth_fields():
    document = _document("full")
    assert document.depthIsEstimated is True
    assert 0.0 <= document.depthConfidence <= 1.0
    assert isinstance(document.warnings, list)
    quality = document.quality
    assert quality.depthConfidence == pytest.approx(document.depthConfidence, abs=1e-4)
    assert quality.zJitterRms >= 0
    assert 0.0 <= quality.boneLengthCv <= 1.0


def test_old_document_without_depth_fields_still_loads():
    """1.0.0 时代的文件必须还能读入（新字段有默认值）。"""
    payload = build_sample_document().model_dump(mode="json")
    for key in ("depthIsEstimated", "depthConfidence", "warnings"):
        payload.pop(key, None)
    for key in ("zJitterRms", "boneLengthCv", "meanReprojError", "depthConfidence"):
        payload["quality"].pop(key, None)

    restored = ExportDocument.model_validate(payload)
    assert restored.depthIsEstimated is True
    assert restored.depthConfidence == 0.0
    assert restored.warnings == []
    assert restored.quality.depthConfidence == 0.0


def test_low_depth_confidence_writes_warnings():
    """深度不可信时必须在 JSON 的 warnings 里给出可读原因（任务 5.4）。"""
    from app.backend.export.builder import depth_warnings
    from app.backend.postprocess.quality import QualityReport

    report = QualityReport(depth_confidence=0.2, z_jitter_rms=0.04, bone_length_cv=0.22)
    warnings = depth_warnings(report)
    assert warnings
    assert any("depthConfidence" in item for item in warnings)
    assert any("侧面" in item for item in warnings)

    good = QualityReport(depth_confidence=0.9)
    assert depth_warnings(good) == []
