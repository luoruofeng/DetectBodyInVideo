"""断点续算（任务 9.3）。

格式为 JSONL：一行一帧，``{"index": 12, "frames": [...]}``。
- 追加写 + 每批 flush，进程被杀最多丢最后一批；
- 读取时容错：半截行、坏行直接跳过，不让断点文件本身成为故障源；
- 断点 key 由视频元信息 + 采样计划派生，换视频/换采样策略不会误用旧断点。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from app.backend.core.logging import get_logger
from app.backend.engine.base import PoseFrame
from app.backend.media.probe import VideoProbe
from app.backend.media.sampling import AnalysisPlan

logger = get_logger("perf.checkpoint")

CHECKPOINT_VERSION = 1


def checkpoint_key(video_path: Path | str, probe: VideoProbe, plan: AnalysisPlan) -> str:
    """视频 + 采样计划的稳定指纹（同名不同内容也不会串）。"""
    path = Path(video_path)
    try:
        stat = path.stat()
        stat_key = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stat_key = path.name
    payload = {
        "v": CHECKPOINT_VERSION,
        "file": stat_key,
        "fps": round(float(probe.fps or 0), 4),
        "frames": int(probe.frame_count or 0),
        "plan": plan.model_dump(mode="json"),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest[:16]


class FrameCheckpoint:
    """一帧一行的结果缓存；``append`` 后必须 ``flush`` 才保证落盘。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._sink = None
        self._pending = 0
        self.written = 0

    # ---- 生命周期 ----
    def _open(self):
        if self._sink is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sink = self.path.open("a", encoding="utf-8", newline="\n")
        return self._sink

    def close(self) -> None:
        if self._sink is not None:
            self._sink.flush()
            self._sink.close()
            self._sink = None

    def remove(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)

    # ---- 写 ----
    def append(self, index: int, frames: list[PoseFrame]) -> None:
        sink = self._open()
        record = {
            "index": int(index),
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }
        sink.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.written += 1
        self._pending += 1

    def flush(self) -> None:
        if self._sink is not None and self._pending:
            self._sink.flush()
            self._pending = 0

    # ---- 读 ----
    def load(self) -> dict[int, list[PoseFrame]]:
        """返回 ``index -> frames``；坏行跳过并计数。"""
        result: dict[int, list[PoseFrame]] = {}
        if not self.path.is_file():
            return result
        skipped = 0
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if record.get("kind") == "header" or "index" not in record:
                    continue
                try:
                    frames = [
                        PoseFrame.model_validate(item) for item in record.get("frames", [])
                    ]
                except Exception:  # noqa: BLE001 - 单行损坏不影响其余帧
                    skipped += 1
                    continue
                result[int(record["index"])] = frames
        if skipped:
            logger.warning("断点文件有 %s 行损坏，已跳过：%s", skipped, self.path.name)
        return result

    @classmethod
    def open(cls, directory: Path | str, key: str) -> FrameCheckpoint:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = cls(folder / f"{key}.pose.jsonl")
        if not checkpoint.path.is_file():
            header = {"kind": "header", "version": CHECKPOINT_VERSION, "key": key, "created": time.time()}
            with checkpoint.path.open("w", encoding="utf-8", newline="\n") as sink:
                sink.write(json.dumps(header, ensure_ascii=False) + "\n")
        return checkpoint


def clear_checkpoints(directory: Path | str) -> int:
    """删除目录下全部断点文件，返回删除数量（目录不存在视为 0）。

    单个文件删不掉只记日志，不让整个清理失败：调用方据此决定是否提示。
    """
    folder = Path(directory)
    if not folder.is_dir():
        return 0
    removed = 0
    for path in folder.glob("*.pose.jsonl"):
        try:
            path.unlink()
            removed += 1
        except OSError as exc:  # noqa: PERF203 - 单个文件失败不影响其余文件
            logger.warning("断点文件删除失败（%s）：%s", path.name, exc)
    if removed:
        logger.info("已清理 %s 个断点文件：%s", removed, folder)
    return removed


__all__ = ["CHECKPOINT_VERSION", "FrameCheckpoint", "checkpoint_key", "clear_checkpoints"]
