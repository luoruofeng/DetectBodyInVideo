"""首启模型检测与下载接口（任务 10.4）。"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.backend.core.settings import Settings
from app.backend.engine import model_downloads
from app.backend.engine.model_downloads import ModelDownloadManager, model_inventory
from app.backend.main import create_app


def test_inventory_marks_present_and_mirror(video_settings: Settings) -> None:
    video_settings.models_dir.mkdir(parents=True, exist_ok=True)
    (video_settings.models_dir / "pose_landmarker_lite.task").write_bytes(b"fake")

    inventory = model_inventory(video_settings)
    lite = next(item for item in inventory if item.tier == "lite")
    full = next(item for item in inventory if item.tier == "full")
    assert lite.present is True
    assert lite.size_bytes == 4
    assert full.mirrorUrl == full.url

    mirrored = video_settings.model_copy(
        update={"model_mirror_url": "https://mirror.example.com/mediapipe-models"}
    )
    full_mirrored = next(item for item in model_inventory(mirrored) if item.tier == "full")
    assert full_mirrored.mirrorUrl is not None
    assert full_mirrored.mirrorUrl.startswith("https://mirror.example.com/mediapipe-models/pose_landmarker/")


def test_download_task_reports_progress(video_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url, destination, *, progress=None, settings=None, use_mirror=None, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 100
        for chunk in (40, 40, 20):
            if progress is not None:
                progress(min(total, chunk), total)
        destination.write_bytes(b"x" * 100)
        if progress is not None:
            progress(total, total)
        return destination

    monkeypatch.setattr(model_downloads, "download_model", fake_download)

    manager = ModelDownloadManager(video_settings)
    task = manager.start(kind="pose", tier="full")
    deadline = time.perf_counter() + 3
    status = None
    while time.perf_counter() < deadline:
        status = manager.status()
        current = next(item for item in status.downloads if item.id == task.id)
        if current.state in {"done", "error"}:
            break
        time.sleep(0.02)

    assert current.state == "done", current
    assert current.percent == 100.0
    assert (video_settings.models_dir / "pose_landmarker_full.task").is_file()
    assert manager.status().poseReady is True


def test_duplicate_download_returns_same_task(
    video_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow_download(url, destination, *, progress=None, settings=None, use_mirror=None, **kwargs):
        release.wait(timeout=3)
        destination.write_bytes(b"x")
        return destination

    monkeypatch.setattr(model_downloads, "download_model", slow_download)
    manager = ModelDownloadManager(video_settings)
    first = manager.start(kind="pose", tier="heavy")
    second = manager.start(kind="pose", tier="heavy")
    assert first.id == second.id
    release.set()
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        if manager.status().downloads[0].state == "done":
            break
        time.sleep(0.02)


def test_models_api(video_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(video_settings)
    monkeypatch.setattr(
        model_downloads,
        "download_model",
        lambda url, destination, **kwargs: destination.write_bytes(b"x") or destination,
    )
    # 只认本测试的数据目录，避免仓库 models/ 里已有的模型让 poseReady 预置为 True
    monkeypatch.setattr(
        model_downloads,
        "find_model",
        lambda filename, settings=None: (
            candidate
            if (candidate := (settings or video_settings).models_dir / filename).is_file()
            else None
        ),
    )
    with TestClient(app) as client:
        listed = client.get("/api/models")
        assert listed.status_code == 200
        payload = listed.json()
        assert len(payload["models"]) == 5
        assert payload["poseReady"] is False
        assert payload["mirrorConfigured"] is False

        invalid = client.post("/api/models/download", json={"kind": "pose", "tier": "unknown"})
        assert invalid.status_code == 422

        started = client.post(
            "/api/models/download", json={"kind": "pose", "tier": "lite", "use_mirror": True}
        )
        assert started.status_code == 200
        assert started.json()["state"] in {"queued", "running", "done"}

        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline:
            current = client.get("/api/models").json()
            task = next(item for item in current["downloads"] if item["id"] == started.json()["id"])
            if task["state"] in {"done", "error"}:
                break
            time.sleep(0.03)
        assert task["state"] == "done"
        assert client.get("/api/models").json()["poseReady"] is True
