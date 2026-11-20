"""测试夹具：合成短视频（不依赖外部素材），并隔离数据目录。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.backend.core.settings import Settings

TEST_FPS = 10.0
TEST_FRAMES = 40
TEST_SIZE = (160, 120)


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成 4 秒 / 10fps 的人形色块视频，用于探测、读取、上传测试。"""
    directory = tmp_path_factory.mktemp("videos")
    path = directory / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), TEST_FPS, TEST_SIZE
    )
    if not writer.isOpened():
        pytest.skip("当前 OpenCV 缺少 mp4v 编码器，无法生成测试视频")

    width, height = TEST_SIZE
    for index in range(TEST_FRAMES):
        frame = np.full((height, width, 3), 28, dtype=np.uint8)
        x = 8 + index * 3
        cv2.rectangle(frame, (x, 25), (x + 35, height - 5), (205, 205, 205), -1)
        writer.write(frame)
    writer.release()

    assert path.stat().st_size > 0
    return path


@pytest.fixture()
def video_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        static_dir=tmp_path / "static",
        log_level="ERROR",
        max_upload_bytes=64 * 1024 * 1024,
    )
