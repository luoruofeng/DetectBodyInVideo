"""生产者 / 消费者推理循环（任务 9.1 / 9.3）。

```
[解码线程] --有界队列(背压)--> [推理线程: 攒批 → engine.infer_batch] --> 断点 JSONL
```

- 解码与推理彻底分离：GPU 推理时 CPU 继续解下一帧，避免"读一帧等一帧"；
- 队列有界：推理慢时解码自动阻塞，内存不会无限增长；
- 取消：``should_cancel`` 返回 True 立刻停线程、清队列并抛 ``AnalysisCancelledError``；
- 断点：按帧追加，取消/崩溃后再次运行可跳过已完成帧（并对全量帧重新分配 track id，
  避免断点前后 track 编号冲突）。
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.errors import AnalysisCancelledError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import PoseEngine, PoseFrame
from app.backend.engine.tracking import IouTracker
from app.backend.media.probe import VideoProbe
from app.backend.media.reader import FramePacket, VideoFrameReader
from app.backend.media.sampling import AnalysisPlan
from app.backend.perf.checkpoint import FrameCheckpoint, checkpoint_key
from app.backend.perf.config import RuntimeConfig

logger = get_logger("perf.runner")

_SENTINEL = object()


class RuntimeProgress(BaseModel):
    frames_done: int = 0
    frames_total: int = 0
    processed_fps: float = 0.0
    eta_seconds: float | None = None
    message: str = ""


class RuntimeResult(BaseModel):
    frames: list[PoseFrame] = Field(default_factory=list)
    processed: int = 0
    resumed_frames: int = 0
    elapsed_seconds: float = 0.0
    processed_fps: float = 0.0
    batch_size: int = 1
    checkpoint: str | None = None
    notes: list[str] = Field(default_factory=list)


ProgressCallback = Callable[[RuntimeProgress], None]
CancelCallback = Callable[[], bool]
# 推理进行中把"到目前为止的帧"快照交出去，用于边算边看骨架。
FramesCallback = Callable[[list[PoseFrame]], None]
# 每帧推理后做手/脸增强：入参为原始 BGR 帧与该帧的姿态结果。
RefineCallback = Callable[[np.ndarray, list[PoseFrame]], None]

# 快照节流：太频繁会白白复制列表 + 序列化，太慢又会让预览明显落后于视频。
_PREVIEW_INTERVAL_S = 0.4


def _iter_ranges(plan: AnalysisPlan) -> list[tuple[float, float]]:
    if plan.active_ranges:
        return [(item.start_s, item.end_s) for item in plan.active_ranges]
    return [(plan.start_s, plan.end_s)]


def _packets(
    path: Path,
    probe: VideoProbe,
    plan: AnalysisPlan,
    *,
    skip: set[int],
) -> Iterator[FramePacket]:
    reader = VideoFrameReader(path, probe)
    try:
        for start_s, end_s in _iter_ranges(plan):
            for packet in reader.frames(
                start_s=start_s, end_s=end_s, stride=max(1, plan.stride)
            ):
                if packet.index in skip:
                    continue
                yield packet
    finally:
        reader.close()


def _assign_tracks(
    detected: list[PoseFrame], tracker: IouTracker, engine: PoseEngine, index: int
) -> None:
    if not detected:
        return
    if any(item.track_id is None for item in detected):
        if len(detected) > 1 or engine.capabilities.multi_person:
            tracker.update(detected, index)
        else:
            detected[0].track_id = 1


def _retrack(frames: list[PoseFrame], settings: Settings | None) -> None:
    """断点续算后统一重排 track id，保证断点前后不串人。"""
    tracker = IouTracker(
        iou_threshold=settings.track_iou_threshold if settings else 0.3,
        max_age=settings.track_max_age if settings else 15,
    )
    grouped: dict[int, list[PoseFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.index, []).append(frame)
    for index in sorted(grouped):
        detected = grouped[index]
        for frame in detected:
            frame.track_id = None
        tracker.update(detected, index)


def run_inference(
    engine: PoseEngine,
    path: Path | str,
    probe: VideoProbe,
    plan: AnalysisPlan,
    *,
    config: RuntimeConfig | None = None,
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    max_frames: int | None = None,
    checkpoint_dir: Path | str | None = None,
    on_frames: FramesCallback | None = None,
    refine_frame: RefineCallback | None = None,
) -> RuntimeResult:
    """跑完采样计划里的所有帧；取消会抛 ``AnalysisCancelledError``。"""
    config = config or RuntimeConfig()
    video_path = Path(path)
    total = max(0, plan.selected_frames)
    notes: list[str] = []

    checkpoint: FrameCheckpoint | None = None
    done: dict[int, list[PoseFrame]] = {}
    if config.checkpoint and checkpoint_dir is not None:
        key = checkpoint_key(video_path, probe, plan)
        checkpoint = FrameCheckpoint.open(checkpoint_dir, key)
        if config.resume:
            done = checkpoint.load()
        else:
            # 新任务不复用旧断点，避免上一次的半成品污染本次结果
            checkpoint.remove()
            checkpoint = FrameCheckpoint.open(checkpoint_dir, key)

    resumed = sum(len(value) for value in done.values())
    if done:
        notes.append(f"断点续算：跳过已完成 {len(done)} 帧")

    remaining_total = max(0, total - len(done))
    if max_frames is not None:
        remaining_total = min(remaining_total, max(0, max_frames - len(done)))

    batch_size = (
        max(1, config.effective_batch_size)
        if getattr(engine.capabilities, "supports_batching", False)
        else 1
    )
    if batch_size > 1:
        notes.append(f"批处理 batch={batch_size}（引擎支持批量推理）")
    if config.segment_frames:
        segments = max(1, math.ceil(max(1, total) / config.segment_frames))
        notes.append(f"分段处理：{segments} 段，每段 ≤{config.segment_frames} 帧")

    started = time.perf_counter()
    state = {
        "new": 0,
        "frames": [frame for items in done.values() for frame in items],
        "error": None,
    }
    stop = threading.Event()
    producer_finished = threading.Event()

    def cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    def emit(message: str = "") -> None:
        if progress is None:
            return
        done_count = len(done) + state["new"]
        elapsed = max(1e-6, time.perf_counter() - started)
        fps = state["new"] / elapsed
        remaining = max(0, min(total, len(done) + remaining_total) - done_count)
        progress(
            RuntimeProgress(
                frames_done=done_count,
                frames_total=total,
                processed_fps=round(fps, 2),
                eta_seconds=round(remaining / fps, 1) if fps > 0 else None,
                message=message,
            )
        )

    def produce(sink: queue.Queue[object]) -> None:
        try:
            for packet in _packets(video_path, probe, plan, skip=set(done)):
                while not stop.is_set():
                    try:
                        sink.put(packet, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - 交给消费侧统一上报
            state["error"] = exc
            logger.warning("解码线程异常", exc_info=True)
        finally:
            producer_finished.set()
            while True:
                try:
                    sink.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    if stop.is_set():
                        break

    tracker = IouTracker(
        iou_threshold=settings.track_iou_threshold if settings else 0.3,
        max_age=settings.track_max_age if settings else 15,
    )
    work: queue.Queue[object] = queue.Queue(maxsize=config.queue_size)
    preview_at = 0.0
    producer = threading.Thread(target=produce, args=(work,), name="frame-producer", daemon=True)

    def publish_preview() -> None:
        """把已有帧的浅拷贝交给上层（边推理边预览），失败不影响分析。"""
        if on_frames is None:
            return
        try:
            on_frames(list(state["frames"]))
        except Exception:  # noqa: BLE001 - 预览是附带能力
            logger.debug("推理预览回调失败", exc_info=True)

    def process(batch: list[FramePacket]) -> None:
        if cancelled():
            raise AnalysisCancelledError("分析已被取消")
        room = remaining_total - state["new"]
        if room <= 0:
            return
        batch = batch[:room]  # max_frames 截断：不额外多算帧
        outputs = engine.infer_batch(
            [(packet.index, packet.timestamp_ms, packet.image) for packet in batch]
        )
        for packet, detected in zip(batch, outputs, strict=False):
            _assign_tracks(detected, tracker, engine, packet.index)
            if refine_frame is not None and detected:
                try:  # 增强失败不应打断主链路推理
                    refine_frame(packet.image, detected)
                except Exception:  # noqa: BLE001
                    logger.debug("逐帧增强失败（index=%s）", packet.index, exc_info=True)
            state["frames"].extend(detected)
            state["new"] += 1
            if checkpoint is not None:
                checkpoint.append(packet.index, detected)
        if checkpoint is not None:
            checkpoint.flush()

        nonlocal preview_at
        if on_frames is not None:
            now = time.perf_counter()
            if now - preview_at >= _PREVIEW_INTERVAL_S:
                preview_at = now
                publish_preview()

        processed = state["new"]
        if config.segment_frames:
            index = min(
                total,
                len(done) + processed,
            )
            segment = index // config.segment_frames + 1
            message = f"第 {segment} 段"
        else:
            message = "识别中"
        if processed % max(1, batch_size) == 0:
            emit(message)

    emit("开始解码")
    producer.start()
    # 断点续算：先把手上已有的帧交给预览，旧关键点不用等到新帧算出来才可见。
    publish_preview()
    try:
        batch: list[FramePacket] = []
        finished = False
        while not finished:
            if cancelled():
                raise AnalysisCancelledError("分析已被取消")
            if state["new"] >= remaining_total:
                stop.set()
                break
            try:
                item = work.get(timeout=0.05)
            except queue.Empty:
                if producer_finished.is_set() and work.empty():
                    if batch:
                        process(batch)
                        batch = []
                    break
                continue

            if item is _SENTINEL:
                finished = True
                if batch:
                    process(batch)
                    batch = []
                break

            batch.append(item)  # type: ignore[arg-type]
            if len(batch) >= batch_size:
                process(batch)
                batch = []
                continue

            # 攒到 batch_size：只等极短时间（解码一帧通常远小于 10ms）
            while len(batch) < batch_size:
                try:
                    extra = work.get(timeout=0.01)
                except queue.Empty:
                    break
                if extra is _SENTINEL:
                    finished = True
                    break
                batch.append(extra)  # type: ignore[arg-type]
            if batch:
                process(batch)
                batch = []
    finally:
        stop.set()
        if producer.is_alive():
            producer.join(timeout=5.0)
        # 清空队列，让可能阻塞在 put 的生产者尽快退出
        while True:
            try:
                work.get_nowait()
            except queue.Empty:
                break
        if checkpoint is not None:
            checkpoint.flush()
        if cancelled():
            # 取消：把最后一批帧也交给预览，再保留断点文件方便下次续算；引擎由上层 close()
            publish_preview()
            if checkpoint is not None:
                checkpoint.close()
            raise AnalysisCancelledError("分析已被取消")

    if state["error"] is not None:
        if checkpoint is not None:
            checkpoint.close()
        raise state["error"]  # type: ignore[misc]

    if done:
        _retrack(state["frames"], settings)
        notes.append("断点续算完成，已对全量帧重新分配 track id")

    if checkpoint is not None:
        checkpoint.remove()  # 成功完成，断点无意义

    elapsed = time.perf_counter() - started
    outcome = RuntimeResult(
        frames=state["frames"],
        processed=state["new"],
        resumed_frames=resumed,
        elapsed_seconds=round(elapsed, 3),
        processed_fps=round(state["new"] / elapsed, 2) if elapsed > 0 else 0.0,
        batch_size=batch_size,
        checkpoint=None,
        notes=notes,
    )
    logger.info(
        "推理完成：新处理 %s 帧（续算 %s 帧），%.1f fps，batch=%s",
        outcome.processed,
        outcome.resumed_frames,
        outcome.processed_fps,
        outcome.batch_size,
    )
    emit("推理完成")
    publish_preview()  # 补一次完整快照，避免最后不足一个节流周期的帧漏掉
    return outcome


__all__ = [
    "FramesCallback",
    "RefineCallback",
    "RuntimeProgress",
    "RuntimeResult",
    "run_inference",
]
