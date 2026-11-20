"""分析与导出接口（任务 8 后端支撑）：起任务 → 轮询 → 帧数据 → 导出 → 下载。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_pipeline import FakePoseEngine

from app.backend.core.settings import Settings
from app.backend.jobs import AnalysisJobManager
from app.backend.main import create_app


@pytest.fixture()
def client(video_settings: Settings) -> Iterator[TestClient]:
    app = create_app(video_settings)
    app.state.analysis_jobs = AnalysisJobManager(
        video_settings, app.state.video_store, engine_factory=FakePoseEngine
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, sample_video: Path) -> str:
    response = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", sample_video.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200, response.text
    return response.json()["video"]["id"]


def _wait_done(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        status = client.get(f"/api/analysis/{job_id}").json()
        if status["phase"] in {"done", "error", "cancelled"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 超时未结束")


def test_analysis_then_frames_and_export(client: TestClient, sample_video: Path, tmp_path: Path) -> None:
    video_id = _upload(client, sample_video)
    started = client.post(
        "/api/analysis",
        json={
            "video_id": video_id,
            "options": {
                "engine": "auto",
                "skeleton": {"include_optional": True},
                "perf": {"batch_size": 2, "queue_size": 8},
            },
        },
    )
    assert started.status_code == 201, started.text
    job = started.json()
    assert job["phase"] in {"queued", "running"}
    assert job["canCancel"] is True

    status = _wait_done(client, job["id"])
    assert status["phase"] == "done", status
    assert status["summary"]["frames"] == 40
    assert status["progress"]["frames_done"] == 40
    assert status["canExport"] is True
    assert status["selection"]["engine"] == "fake"

    frames = client.get(f"/api/analysis/{job['id']}/frames").json()
    assert frames["totalFrames"] == 40
    assert len(frames["frames"]) == 40
    first = frames["frames"][0]
    assert len(first["landmarks"]) == 33
    assert first["joints"]["left_hand_joint"]["x"] is not None
    assert first["angles"]

    sliced = client.get(f"/api/analysis/{job['id']}/frames?offset=10&limit=5").json()
    assert [item["index"] for item in sliced["frames"]] == list(range(10, 15))
    assert sliced["hasMore"] is True

    exported = client.post(
        f"/api/analysis/{job['id']}/export",
        json={"granularity": "full", "output_dir": str(tmp_path), "include_raw_landmarks": True},
    )
    assert exported.status_code == 200, exported.text
    payload = exported.json()
    artifact = payload["artifact"]
    assert payload["downloadUrl"] is None  # 自定义目录不出对外下载链接
    assert artifact["frames"] == 40
    assert artifact["mode"] == "full"

    document = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert document["version"] == "1.1.0"
    assert document["coordinateSystem"] == "arkit-right-handed-y-up"
    assert document["depthIsEstimated"] is True
    assert 0.0 <= document["depthConfidence"] <= 1.0
    assert isinstance(document["warnings"], list)
    assert payload["depthConfidence"] == document["depthConfidence"]
    assert len(document["frames"]) == 40
    assert document["frames"][0]["rawLandmarks"]


def test_export_motion_json(
    client: TestClient, sample_video: Path, tmp_path: Path
) -> None:
    video_id = _upload(client, sample_video)
    job_id = client.post("/api/analysis", json={"video_id": video_id}).json()["id"]
    _wait_done(client, job_id)

    exported = client.post(
        f"/api/analysis/{job_id}/export",
        json={
            "format": "motion",
            "output_dir": str(tmp_path),
            "motion": {"sample_interval_ms": 100, "name": "Air Squat", "name_zh": "深蹲"},
        },
    )
    assert exported.status_code == 200, exported.text
    payload = exported.json()
    assert payload["format"] == "motion"
    assert payload["schemaVersion"] == "1"
    artifact = payload["artifact"]
    assert artifact["mode"] == "motion"
    assert artifact["filename"].endswith("_motion.json")

    document = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert document["schemaVersion"] == 1
    assert document["coordinateSpace"] == "hip-local-meters-yup"
    assert document["exerciseID"] == "air-squat"
    assert document["nameZh"] == "深蹲"
    assert document["sampleIntervalMs"] == 100
    assert document["frames"]
    first = document["frames"][0]
    assert first["pelvisOffsetMeters"] == [0.0, 0.0, 0.0]
    assert set(first["joints"]) == set(document["skeleton"])
    assert set(document["skeleton"]) <= {
        "root",
        "pelvis",
        "spine",
        "neck",
        "centerShoulder",
        "leftShoulder",
        "leftElbow",
        "leftWrist",
        "rightShoulder",
        "rightElbow",
        "rightWrist",
        "leftHip",
        "leftKnee",
        "leftAnkle",
        "rightHip",
        "rightKnee",
        "rightAnkle",
    }


def test_export_to_default_dir_has_download_url(
    client: TestClient, sample_video: Path, video_settings: Settings
) -> None:
    video_id = _upload(client, sample_video)
    job_id = client.post("/api/analysis", json={"video_id": video_id}).json()["id"]
    _wait_done(client, job_id)

    exported = client.post(
        f"/api/analysis/{job_id}/export", json={"granularity": "compact"}
    ).json()
    url = exported["downloadUrl"]
    assert url and url.startswith("/api/exports/")

    download = client.get(url)
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/json")

    listing = client.get("/api/exports").json()
    assert any(item["name"] == exported["artifact"]["filename"] for item in listing)
    assert video_settings.exports_dir.is_dir()


def test_clear_cache_removes_checkpoint_files(
    client: TestClient, video_settings: Settings
) -> None:
    """清缓存接口只负责删断点文件，让下一次识别从第 1 帧开始。"""
    cache = video_settings.checkpoints_dir
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "deadbeef.pose.jsonl"
    target.write_text('{"kind": "header"}\n', encoding="utf-8")

    response = client.delete("/api/analysis/cache")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["removed"] == 1
    assert payload["directory"] == str(cache)
    assert not target.exists()


def test_schema_endpoint(client: TestClient) -> None:
    response = client.get("/api/exports/schema")
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert payload["x-schema-version"] == "1.1.0"


def test_frames_before_done_is_rejected(client: TestClient, sample_video: Path) -> None:
    video_id = _upload(client, sample_video)
    job_id = client.post("/api/analysis", json={"video_id": video_id}).json()["id"]
    response = client.get(f"/api/analysis/{job_id}/frames")
    assert response.status_code in {422, 200}
    if response.status_code == 422:
        assert response.json()["error"]["code"] == "validation_failed"
    _wait_done(client, job_id)


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/analysis/nope").status_code == 404


def test_preview_frames_available_while_postprocessing(
    client: TestClient, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后处理卡住时也能看到推理原始骨架；完成后自动切成最终结果。"""
    import app.backend.pipeline as pipeline_module

    original = pipeline_module.run_postprocess
    entered = threading.Event()
    release = threading.Event()

    def slow_postprocess(frames, config, **kwargs):
        entered.set()
        release.wait(10)
        return original(frames, config, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_postprocess", slow_postprocess)
    video_id = _upload(client, sample_video)
    job_id = client.post("/api/analysis", json={"video_id": video_id}).json()["id"]
    assert entered.wait(10)

    deadline = time.perf_counter() + 5
    status: dict = {}
    while time.perf_counter() < deadline:
        status = client.get(f"/api/analysis/{job_id}").json()
        if status.get("canPreview"):
            break
        time.sleep(0.05)
    assert status["phase"] == "running"
    assert status["canPreview"] is True
    assert status["progress"]["stage"] not in {"inference", "done"}

    payload = client.get(f"/api/analysis/{job_id}/frames?offset=0&limit=5").json()
    assert payload["totalFrames"] == 40
    assert len(payload["frames"]) == 5
    assert len(payload["frames"][0]["landmarks"]) == 33
    assert payload["frames"][0]["joints"]

    release.set()
    done = _wait_done(client, job_id)
    assert done["canPreview"] is False
    assert done["progress"]["stage"] == "done"
    final = client.get(f"/api/analysis/{job_id}/frames").json()
    assert final["totalFrames"] == 40


def test_frames_available_while_inferring(
    client: TestClient, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """推理还没跑完也要能读到"已经识别出来的帧"，这样生成过程中就能边播边看骨架。"""
    original = FakePoseEngine.infer_batch
    release = threading.Event()
    calls = {"count": 0}

    def slow_infer_batch(self, items):
        calls["count"] += 1
        if calls["count"] == 2:
            release.wait(10)
        return original(self, items)

    monkeypatch.setattr(FakePoseEngine, "infer_batch", slow_infer_batch)
    video_id = _upload(client, sample_video)
    job_id = client.post("/api/analysis", json={"video_id": video_id}).json()["id"]

    deadline = time.perf_counter() + 10
    status: dict = {}
    while time.perf_counter() < deadline:
        status = client.get(f"/api/analysis/{job_id}").json()
        if status.get("canPreview"):
            break
        time.sleep(0.05)
    assert status["phase"] == "running"
    assert status["canPreview"] is True

    payload = client.get(f"/api/analysis/{job_id}/frames?offset=0&limit=50").json()
    assert 0 < payload["totalFrames"] < 40
    assert len(payload["frames"]) == payload["totalFrames"]
    assert len(payload["frames"][0]["landmarks"]) == 33
    # 边算边看：推理中的部分帧也要带米制关节与角度，否则 3D / 白膜卡片一直空着
    assert payload["jointNames"]
    assert payload["frames"][0]["joints"]
    assert payload["frames"][0]["angles"]
    assert payload["hasMore"] is False

    release.set()
    done = _wait_done(client, job_id)
    assert done["phase"] == "done"
    final = client.get(f"/api/analysis/{job_id}/frames").json()
    assert final["totalFrames"] == 40
    assert final["frames"][0]["joints"]
