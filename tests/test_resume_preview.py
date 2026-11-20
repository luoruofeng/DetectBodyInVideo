"""取消 → 断点续算：已算出的关键点必须能继续预览（不能只剩新帧）。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_pipeline import FakePoseEngine

from app.backend.core.settings import Settings
from app.backend.jobs import AnalysisJobManager
from app.backend.main import create_app


class SlowPoseEngine(FakePoseEngine):
    def infer(self, image, *, index: int = 0, timestamp_ms: float = 0.0):
        time.sleep(0.01)
        return super().infer(image, index=index, timestamp_ms=timestamp_ms)


@pytest.fixture()
def client(video_settings: Settings) -> Iterator[TestClient]:
    app = create_app(video_settings)
    app.state.analysis_jobs = AnalysisJobManager(
        video_settings, app.state.video_store, engine_factory=SlowPoseEngine
    )
    with TestClient(app) as test_client:
        yield test_client


def _status(client: TestClient, job_id: str) -> dict:
    return client.get(f"/api/analysis/{job_id}").json()


def _wait(client: TestClient, job_id: str, predicate, timeout: float = 20.0) -> dict:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        status = _status(client, job_id)
        if predicate(status):
            return status
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 等待超时：{status}")


def test_resume_preview_includes_previously_done_frames(
    client: TestClient, sample_video: Path
) -> None:
    video_id = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", sample_video.read_bytes(), "video/mp4")},
    ).json()["video"]["id"]
    options = {"perf": {"checkpoint": True, "resume": True, "batch_size": 1}}

    first = client.post(
        "/api/analysis", json={"video_id": video_id, "options": options}
    ).json()["id"]
    _wait(client, first, lambda item: item["progress"]["frames_done"] >= 12)
    client.post(f"/api/analysis/{first}/cancel")
    cancelled = _wait(client, first, lambda item: item["phase"] == "cancelled")

    # 取消后已算出的帧仍可回放，而不是 422 把之前的关键点全丢掉
    assert cancelled["canPreview"] is True
    partial = client.get(f"/api/analysis/{first}/frames")
    assert partial.status_code == 200
    assert partial.json()["frames"][0]["index"] == 0

    second = client.post(
        "/api/analysis", json={"video_id": video_id, "options": options}
    ).json()["id"]
    running = _wait(client, second, lambda item: item["canPreview"])
    assert running["phase"] == "running"

    # 续算一开始就把恢复的帧交给预览，前端不必等新帧算出来才画得出旧骨架
    frames = client.get(f"/api/analysis/{second}/frames").json()
    indices = [item["index"] for item in frames["frames"]]
    assert indices[0] == 0
    assert len(indices) >= 12
    assert indices == sorted(indices)

    done = _wait(client, second, lambda item: item["phase"] == "done")
    assert done["summary"]["frames"] == 40
    final = client.get(f"/api/analysis/{second}/frames").json()
    assert [item["index"] for item in final["frames"]] == list(range(40))
