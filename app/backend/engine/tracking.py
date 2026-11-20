"""多人 track 管理（任务 4.3 / 4.4）。

- ``IouTracker``：用 bbox IoU 做贪心匹配，把跨帧检测串成 track。
- track 丢失后保留 ``max_age`` 帧，期间重现且 IoU 仍可匹配则沿用原 id；
  超过 ``max_age`` 直接销毁，避免把两个人串成一条轨迹。
- ``summarize_tracks`` / ``primary_track_id``：给 UI "跟随目标人物" 提供候选，
  默认选出现时间最长（帧数最多）的 track。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.backend.core.logging import get_logger
from app.backend.engine.base import BoundingBox, PoseFrame

logger = get_logger("engine.tracking")


class Track(BaseModel):
    """内部状态机：只存最近一次观测，不缓存历史帧。"""

    id: int
    bbox: BoundingBox
    first_index: int
    last_index: int
    hits: int = 1
    misses: int = 0
    score_sum: float = 0.0

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.hits if self.hits else 0.0


class TrackSummary(BaseModel):
    id: int
    frames: int
    first_index: int
    last_index: int
    span_frames: int
    coverage: float
    mean_score: float
    mean_bbox_area: float = 0.0


class IouTracker:
    """单路视频的多人跟踪器（有状态，请按帧顺序调用 ``update``）。"""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = max(0, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._areas: dict[int, list[float]] = {}

    @property
    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._areas.clear()
        self._next_id = 1

    # ---- 匹配 ----
    def _match(self, bbox: BoundingBox) -> Track | None:
        best: Track | None = None
        best_iou = self.iou_threshold
        for track in self._tracks.values():
            score = bbox.iou(track.bbox)
            if score > best_iou:
                best, best_iou = track, score
        return best

    def update(self, frames: list[PoseFrame], index: int | None = None) -> list[PoseFrame]:
        """为一帧内的若干个人分配 ``track_id``，返回同一批 frame（已打标）。"""
        if not frames:
            self._age_tracks(index)
            return frames

        frame_index = index if index is not None else frames[0].index
        claimed: set[int] = set()

        # 按面积从大到小匹配：先安排主体，避免小目标抢走 track
        ordered = sorted(
            frames,
            key=lambda item: (item.bbox or item.bbox_from_landmarks() or BoundingBox()).area,
            reverse=True,
        )
        for frame in ordered:
            bbox = frame.bbox or frame.bbox_from_landmarks()
            frame.bbox = bbox
            if bbox is None:
                frame.track_id = None
                continue
            track = self._match(bbox)
            if track is not None and track.id not in claimed:
                claimed.add(track.id)
                track.bbox = bbox
                track.last_index = frame_index
                track.hits += 1
                track.misses = 0
                track.score_sum += frame.score
                frame.track_id = track.id
                self._areas.setdefault(track.id, []).append(bbox.area)
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    id=track_id,
                    bbox=bbox,
                    first_index=frame_index,
                    last_index=frame_index,
                    hits=1,
                    score_sum=frame.score,
                )
                self._areas[track_id] = [bbox.area]
                frame.track_id = track_id

        self._age_tracks(frame_index, keep=claimed)
        return frames

    def _age_tracks(self, index: int | None, keep: set[int] | None = None) -> None:
        keep = keep or set()
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            if index is not None and track_id not in keep and track.last_index < index:
                track.misses += 1
            if track.misses > self.max_age:
                del self._tracks[track_id]
                self._areas.pop(track_id, None)

    # ---- 汇总 ----
    def summaries(self) -> list[TrackSummary]:
        return _summaries(self._tracks, self._areas)


def summarize_tracks(frames: list[PoseFrame]) -> list[TrackSummary]:
    """从已有 ``track_id`` 的帧序列反推 track 概要（无需 tracker 实例）。"""
    grouped: dict[int, list[PoseFrame]] = {}
    for frame in frames:
        if frame.track_id is None:
            continue
        grouped.setdefault(frame.track_id, []).append(frame)

    trackers: dict[int, Track] = {}
    areas: dict[int, list[float]] = {}
    for track_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: item.index)
        bbox = ordered[-1].bbox or ordered[-1].bbox_from_landmarks() or BoundingBox()
        trackers[track_id] = Track(
            id=track_id,
            bbox=bbox,
            first_index=ordered[0].index,
            last_index=ordered[-1].index,
            hits=len(ordered),
            score_sum=sum(item.score for item in ordered),
        )
        areas[track_id] = [
            (item.bbox.area if item.bbox is not None else 0.0) for item in ordered
        ]
    return _summaries(trackers, areas)


def _summaries(trackers: dict[int, Track], areas: dict[int, list[float]]) -> list[TrackSummary]:
    if not trackers:
        return []
    span = max(track.last_index for track in trackers.values()) - min(
        track.first_index for track in trackers.values()
    ) + 1
    span = max(1, span)
    result: list[TrackSummary] = []
    for track in trackers.values():
        own_span = max(1, track.last_index - track.first_index + 1)
        area_list = areas.get(track.id) or [0.0]
        result.append(
            TrackSummary(
                id=track.id,
                frames=track.hits,
                first_index=track.first_index,
                last_index=track.last_index,
                span_frames=own_span,
                coverage=round(min(1.0, min(track.hits, own_span) / own_span), 4),
                mean_score=round(track.mean_score, 4),
                mean_bbox_area=round(sum(area_list) / len(area_list), 5),
            )
        )
    result.sort(key=lambda item: (-item.frames, item.first_index))
    return result


def primary_track_id(frames: list[PoseFrame]) -> int | None:
    """默认目标人物：出现帧数最多；并列时取面积更大（更靠近镜头）的。"""
    summaries = summarize_tracks(frames)
    if not summaries:
        return None
    best = summaries[0]
    ties = [item for item in summaries if item.frames == best.frames]
    if len(ties) > 1:
        ties.sort(key=lambda item: item.mean_bbox_area, reverse=True)
        return ties[0].id
    return best.id


def filter_track(frames: list[PoseFrame], track_id: int) -> list[PoseFrame]:
    return [frame for frame in frames if frame.track_id == track_id]


def drop_short_tracks(frames: list[PoseFrame], min_frames: int = 3) -> list[PoseFrame]:
    """丢弃只闪 1~2 帧的误检，减少抖动源。"""
    keeps = {item.id for item in summarize_tracks(frames) if item.frames >= min_frames}
    return [frame for frame in frames if frame.track_id in keeps]
