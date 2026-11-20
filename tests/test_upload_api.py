"""上传 / 分块 / 去重 / 元信息接口的集成测试（任务 3.1 - 3.4）。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backend.core.settings import Settings
from app.backend.main import create_app

CHUNK = 256 * 1024


@pytest.fixture()
def client(video_settings: Settings) -> Iterator[TestClient]:
    app = create_app(video_settings)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str, payload: bytes, content_type: str = "video/mp4"):
    return client.post("/api/videos/upload", files={"file": (name, payload, content_type)})


def test_upload_then_dedupe(client: TestClient, sample_video: Path) -> None:
    payload = sample_video.read_bytes()

    first = _upload(client, "clip.mp4", payload)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["deduped"] is False
    video = data["video"]
    assert video["filename"] == "clip.mp4"
    assert video["frame_count"] >= 30
    assert video["duration_s"] > 0
    assert video["thumbnail_url"]

    again = _upload(client, "clip-copy.mp4", payload)
    assert again.status_code == 200
    assert again.json()["deduped"] is True
    assert again.json()["video"]["id"] == video["id"]

    listing = client.get("/api/videos").json()
    assert any(item["id"] == video["id"] for item in listing)


def test_uploaded_video_assets(client: TestClient, sample_video: Path) -> None:
    video = _upload(client, "assets.mp4", sample_video.read_bytes()).json()["video"]
    video_id = video["id"]

    meta = client.get(f"/api/videos/{video_id}")
    assert meta.status_code == 200 and meta.json()["id"] == video_id

    thumbnail = client.get(f"/api/videos/{video_id}/thumbnail")
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/jpeg"
    assert len(thumbnail.content) > 0

    source = client.get(f"/api/videos/{video_id}/file")
    assert source.status_code == 200
    assert source.content[:4] != b""


def test_reject_unsupported_container(client: TestClient) -> None:
    response = _upload(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 415
    body = response.json()["error"]
    assert body["code"] == "unsupported_media"
    assert "容器" in body["message"]
    assert "MP4" in body["details"]["supported"]


def test_reject_empty_file(client: TestClient) -> None:
    response = _upload(client, "empty.mp4", b"")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"


def test_reject_corrupt_video(client: TestClient) -> None:
    response = _upload(client, "broken.mp4", b"not a real video at all")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"


def test_plan_endpoint(client: TestClient, sample_video: Path) -> None:
    video = _upload(client, "plan.mp4", sample_video.read_bytes()).json()["video"]
    response = client.post(
        f"/api/videos/{video['id']}/plan",
        json={"every_n_frames": 2, "trim_empty": False},
    )
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["stride"] == 2
    assert plan["selected_frames"] > 0
    assert plan["effective_fps"] > 0


def test_chunked_upload_flow(client: TestClient, sample_video: Path) -> None:
    payload = sample_video.read_bytes()
    init = client.post(
        "/api/videos/uploads",
        json={"filename": "chunked.mp4", "size_bytes": len(payload), "chunk_size": CHUNK},
    )
    assert init.status_code == 200, init.text
    session = init.json()
    assert session["total_chunks"] == 1
    upload_id = session["upload_id"]

    put = client.put(
        f"/api/videos/uploads/{upload_id}/chunks/0",
        content=payload,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert put.status_code == 200, put.text
    assert put.json()["received_chunks"] == [0]
    assert put.json()["progress"] == 1.0

    status = client.get(f"/api/videos/uploads/{upload_id}")
    assert status.status_code == 200

    done = client.post(f"/api/videos/uploads/{upload_id}/complete")
    assert done.status_code == 200, done.text
    assert done.json()["video"]["id"]


def test_chunked_upload_missing_chunks(client: TestClient) -> None:
    init = client.post(
        "/api/videos/uploads",
        json={"filename": "big.mp4", "size_bytes": 1024 * 1024, "chunk_size": CHUNK},
    )
    session = init.json()
    assert session["total_chunks"] == 4

    response = client.post(f"/api/videos/uploads/{session['upload_id']}/complete")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"


def test_chunk_size_mismatch_is_rejected(client: TestClient) -> None:
    init = client.post(
        "/api/videos/uploads",
        json={"filename": "wrong.mp4", "size_bytes": 1024 * 1024, "chunk_size": CHUNK},
    )
    upload_id = init.json()["upload_id"]
    response = client.put(
        f"/api/videos/uploads/{upload_id}/chunks/0",
        content=b"too short",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
    assert "分块大小不符" in response.json()["error"]["message"]


def test_unknown_upload_and_video(client: TestClient) -> None:
    assert client.get("/api/videos/uploads/not-a-session").status_code == 404
    assert client.get("/api/videos/deadbeefdeadbeef").status_code == 404
