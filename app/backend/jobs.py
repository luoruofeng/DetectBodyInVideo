"""分析任务管理（任务 8 后端支撑）。

一个任务 = 一个后台线程跑 ``analyze_video``，前端通过轮询 ``status()`` 拿真实帧数、
fps 与 ETA；取消用 ``threading.Event``，取消后线程退出、引擎 ``close()`` 释放显存。

任务结果只放内存（进程内本地服务），重启后需要重新分析；长视频续算靠
``perf.checkpoint`` 的断点文件，与任务生命周期解耦。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.errors import AnalysisCancelledError, AppError, NotFoundError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import EngineSelection, PoseEngine, PoseFrame
from app.backend.media.probe import VideoProbe
from app.backend.media.store import VideoMetadata, VideoStore
from app.backend.pipeline import (
    AnalysisOptions,
    AnalysisProgress,
    AnalysisResult,
    analyze_video,
    summarize,
)
from app.backend.skeleton.bundle import SkeletonBundle
from app.backend.skeleton.live import LiveFrameJoints, LiveSkeletonMapper

logger = get_logger("jobs")

JobPhase = Literal["queued", "running", "done", "cancelled", "error"]


class JobStatus(BaseModel):
    """给前端的任务快照（不含逐帧数据）。"""

    id: str
    video_id: str
    filename: str
    phase: JobPhase = "queued"
    progress: AnalysisProgress = Field(default_factory=AnalysisProgress)
    selection: EngineSelection | None = None
    summary: dict[str, object] | None = None
    error: str | None = None
    error_code: str | None = None
    error_details: dict[str, object] | None = None
    cancel_requested: bool = False
    elapsedSeconds: float = 0.0
    canCancel: bool = False
    canExport: bool = False
    canPreview: bool = False


@dataclass
class PreviewData:
    """推理完成、后处理尚未结束时的只读预览数据。"""

    frames: list[PoseFrame]
    skeleton: SkeletonBundle


class AnalysisJob:
    def __init__(
        self,
        job_id: str,
        metadata: VideoMetadata,
        path,
        options: AnalysisOptions,
        settings: Settings,
        engine_factory: Callable[[], PoseEngine] | None = None,
    ) -> None:
        self.id = job_id
        self.metadata = metadata
        self.path = path
        self.options = options
        self.settings = settings
        self.engine_factory = engine_factory
        self.probe = VideoProbe.model_validate(metadata.model_dump())
        self._lock = threading.RLock()
        self._phase: JobPhase = "queued"
        self._progress = AnalysisProgress()
        self._selection: EngineSelection | None = None
        self._result: AnalysisResult | None = None
        self._preview: PreviewData | None = None
        # 推理进行中的部分帧（边算边看）：只保留一份快照，随进度反复替换。
        self._live_frames: list[PoseFrame] = []
        # 边算边看的米制关节 / 角度：3D 预览与白膜 Mesh 卡片靠它同步播放头。
        self._live_mapper = LiveSkeletonMapper(options.skeleton, options.postprocess.angles)
        self._live_preview: tuple[LiveFrameJoints, ...] = ()
        self._error: str | None = None
        self._error_code: str | None = None
        self._error_details: dict[str, object] | None = None
        self._cancel = threading.Event()
        self._started = time.perf_counter()
        self._finished_at: float | None = None
        self._thread = threading.Thread(target=self._run, name=f"job-{job_id}", daemon=True)

    # ---- 生命周期 ----
    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> bool:
        if self._phase in {"done", "cancelled", "error"}:
            return False
        self._cancel.set()
        return True

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def finished(self) -> bool:
        return self._phase in {"done", "cancelled", "error"}

    @property
    def result(self) -> AnalysisResult | None:
        with self._lock:
            return self._result

    @property
    def preview(self) -> PreviewData | None:
        with self._lock:
            return self._preview

    @property
    def live_frames(self) -> list[PoseFrame]:
        with self._lock:
            return self._live_frames

    @property
    def live_preview(self) -> tuple[LiveFrameJoints, ...]:
        """推理中已映射好的米制关节快照（按追加顺序，与 live_frames 前缀一一对应）。"""
        with self._lock:
            return self._live_preview

    @property
    def live_joint_names(self) -> list[str]:
        return list(self._live_mapper.joint_names)

    def _set_live_frames(self, frames: Sequence[PoseFrame]) -> None:
        """推理线程回传的部分帧快照；只替换引用，不原地修改，读侧无需深拷贝。"""
        snapshot = list(frames)
        preview: tuple[LiveFrameJoints, ...] | None = None
        if self._live_mapper is not None:
            try:
                # 增量映射：只算新追加的帧，开销摊在推理线程里（每批几帧，忽略不计）。
                self._live_mapper.extend(snapshot)
                preview = self._live_mapper.snapshot()
            except Exception as exc:  # noqa: BLE001 - 预览映射失败不能影响分析
                logger.warning("推理中的骨骼映射失败：%s", exc)
        with self._lock:
            self._live_frames = snapshot
            if preview is not None:
                self._live_preview = preview

    def _set_preview(self, frames: Sequence[PoseFrame], skeleton: SkeletonBundle) -> None:
        with self._lock:
            self._preview = PreviewData(frames=list(frames), skeleton=skeleton)

    def _on_progress(self, progress: AnalysisProgress) -> None:
        with self._lock:
            self._progress = progress

    def _run(self) -> None:
        with self._lock:
            self._phase = "running"
        engine = self.engine_factory() if self.engine_factory is not None else None
        try:
            result = analyze_video(
                self.path,
                probe=self.probe,
                options=self.options,
                settings=self.settings,
                engine=engine,
                progress=self._on_progress,
                should_cancel=self._cancel.is_set,
                on_inference_done=self._set_preview,
                on_partial_frames=self._set_live_frames,
            )
            with self._lock:
                self._result = result
                self._selection = result.selection
                self._preview = None  # 最终结果已就绪，释放预览副本
                self._live_frames = []
                self._live_preview = ()
                self._phase = "done"
                self._progress = AnalysisProgress(
                    frames_done=result.postprocess.quality.frames_total,
                    frames_total=result.postprocess.quality.frames_total,
                    processed_fps=result.processed_fps,
                    eta_seconds=0.0,
                    message="完成",
                    stage="done",
                    stage_done=result.postprocess.quality.frames_total,
                    stage_total=result.postprocess.quality.frames_total,
                )
            logger.info("任务 %s 完成（%s 帧）", self.id, len(result.frames))
        except AnalysisCancelledError as exc:
            with self._lock:
                self._phase = "cancelled"
                self._error = exc.message
                self._error_code = exc.code
                self._error_details = exc.details or None
            logger.info("任务 %s 已取消", self.id)
        except AppError as exc:
            with self._lock:
                self._phase = "error"
                self._error = exc.message
                self._error_code = exc.code
                self._error_details = exc.details or None
            logger.warning("任务 %s 失败：%s", self.id, exc.message)
        except Exception as exc:  # noqa: BLE001 - 后台线程必须兜住
            with self._lock:
                self._phase = "error"
                self._error = str(exc)
                self._error_code = "internal_error"
            logger.exception("任务 %s 异常", self.id)
        finally:
            if engine is not None:
                engine.close()
            with self._lock:
                if self._result is not None:
                    # 成功任务已经有最终结果，边算边看的快照可以释放
                    self._live_frames = []
                    self._live_preview = ()
                # 取消/失败时保留已算出的帧，用户仍能回放检查之前的关键点
                self._finished_at = time.perf_counter()

    # ---- 查询 ----
    def status(self) -> JobStatus:
        with self._lock:
            elapsed = (self._finished_at or time.perf_counter()) - self._started
            summary = summarize(self._result) if self._result is not None else None
            return JobStatus(
                id=self.id,
                video_id=self.metadata.id,
                filename=self.metadata.filename,
                phase=self._phase,
                progress=self._progress.model_copy(),
                selection=self._selection,
                summary=summary,
                error=self._error,
                error_code=self._error_code,
                error_details=self._error_details,
                cancel_requested=self._cancel.is_set(),
                elapsedSeconds=round(elapsed, 3),
                canCancel=not self.finished,
                canExport=self._result is not None,
                canPreview=self._preview is not None or bool(self._live_frames),
            )


class AnalysisJobManager:
    """进程内任务表；最多保留 ``max_jobs`` 个，超出的先清理最老的已结束任务。"""

    def __init__(
        self,
        settings: Settings,
        store: VideoStore,
        *,
        max_jobs: int = 8,
        engine_factory: Callable[[], PoseEngine] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.max_jobs = max(1, max_jobs)
        self.engine_factory = engine_factory
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()

    def start(self, video_id: str, options: AnalysisOptions | None = None) -> AnalysisJob:
        metadata = self.store.get(video_id)
        path = self.store.path_for(video_id)
        options = options or AnalysisOptions()
        # UI 默认导出完整骨骼（含 forearm / toes），避免"仅关键点"和"精简"粒度无法区分
        options = options.model_copy(
            update={"skeleton": options.skeleton.model_copy(update={"include_optional": True})}
        )
        job = AnalysisJob(
            uuid.uuid4().hex[:12],
            metadata,
            path,
            options,
            self.settings,
            engine_factory=self.engine_factory,
        )
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        job.start()
        return job

    def get(self, job_id: str) -> AnalysisJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("分析任务不存在或已过期", details={"jobId": job_id})
        return job

    def cancel(self, job_id: str) -> AnalysisJob:
        job = self.get(job_id)
        job.cancel()
        return job

    def list(self) -> list[JobStatus]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.status() for job in sorted(jobs, key=lambda item: item._started, reverse=True)]

    @property
    def active(self) -> int:
        """仍在排队 / 运行的任务数；清理全局缓存前用它做保护。"""
        with self._lock:
            return sum(1 for job in self._jobs.values() if not job.finished)

    def cancel_all(self) -> int:
        """退出前取消所有任务，尽快释放线程与显存（任务 10.1）。"""
        with self._lock:
            jobs = [job for job in self._jobs.values() if not job.finished]
        return sum(1 for job in jobs if job.cancel())

    def _prune(self) -> None:
        if len(self._jobs) < self.max_jobs:
            return
        finished = [job for job in self._jobs.values() if job.finished]
        finished.sort(key=lambda job: job._finished_at or 0.0)
        for job in finished[: max(1, len(self._jobs) - self.max_jobs + 1)]:
            self._jobs.pop(job.id, None)


__all__ = ["AnalysisJob", "AnalysisJobManager", "JobPhase", "JobStatus"]
