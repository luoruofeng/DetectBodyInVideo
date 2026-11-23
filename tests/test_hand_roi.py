"""手腕 ROI 手部检测的几何与流程测试（不需要 mediapipe 权重）。

为什么要专门测这块：远景全身视频里一只手只有 40~60 px，整帧检测的召回实测只有
三成左右，白膜手指就会整段不动。ROI 裁剪把召回拉到九成以上，但它的正确性完全
依赖"框算得对、坐标能换回去、左右手别认错"这三件事，所以逐项钉住：
ROI 几何、越界平移、坐标回映射、误检过滤、ROI → 整帧兜底的降级顺序。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    Landmark,
)
from app.backend.engine.enhancements import (
    HAND_LANDMARK_COUNT,
    MediaPipeRefiner,
    RefinementConfig,
    _closest_hand,
    _hand_roi_box,
    _roi_to_landmark,
)
from tests.synth_pose import make_frames


class _FakePoint:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeCategory:
    def __init__(self, name: str = "Right") -> None:
        self.category_name = name


class _FakeResult:
    def __init__(self, hit: bool, label: str = "Right") -> None:
        if not hit:
            self.hand_landmarks = []
            self.handedness = []
            return
        # 掌指关节（index 9）与腕点（index 0）相距 0.09（ROI 归一化），落在尺寸容差内。
        self.hand_landmarks = [
            [
                _FakePoint(0.5, 0.45 + 0.01 * index, 0.0)
                for index in range(HAND_LANDMARK_COUNT)
            ]
        ]
        for index in (9,):
            self.hand_landmarks[0][index] = _FakePoint(0.5, 0.54, 0.0)
        self.handedness = [[_FakeCategory(label)]]


class _FakeImage:
    def __init__(self, **kwargs: object) -> None:
        self.data = kwargs.get("data")


class _FakeMp:
    class ImageFormat:  # noqa: D106 - 模拟 mediapipe 的枚举
        SRGB = 1

    Image = _FakeImage


class _FakeHand:
    """按调用顺序决定命中与否的假 HandLandmarker。"""

    def __init__(self, hits: list[bool], label: str = "Right") -> None:
        self.hits = hits
        self.label = label
        self.calls = 0

    def detect(self, image: object) -> _FakeResult:
        position = self.calls
        self.calls += 1
        hit = self.hits[position] if position < len(self.hits) else self.hits[-1]
        return _FakeResult(hit, self.label)


def _wrist_landmark(frame, index: int) -> Landmark:
    return frame.landmarks[index]


def _image(width: int = 640, height: int = 360) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# ROI 几何
# --------------------------------------------------------------------------- #
def test_roi_box_is_ahead_of_wrist_along_forearm() -> None:
    frame = make_frames(1)[0]
    box = _hand_roi_box(
        _wrist_landmark(frame, LEFT_WRIST),
        _wrist_landmark(frame, LEFT_ELBOW),
        640,
        360,
        scale=2.6,
        min_box_px=64,
        side="Left",
    )
    assert box is not None
    assert box.side == "Left"
    assert box.width == box.height or abs(box.width - box.height) <= 1
    # 手在腕点下方（合成姿态里手臂自然下垂），框中心也必须比腕点低。
    wrist_y = _wrist_landmark(frame, LEFT_WRIST).y * 360
    assert box.y0 + box.height / 2 > wrist_y
    # 框必须完整落在画面内
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x0 + box.width <= 640 and box.y0 + box.height <= 360


def test_roi_box_shifts_inside_image_instead_of_shrinking() -> None:
    """手贴画面右边缘时整块框平移进来，尺寸不变——裁小会把手指切掉。"""
    wrist = Landmark(x=0.995, y=0.5, z=0.0, visibility=0.99)
    elbow = Landmark(x=0.995, y=0.35, z=0.0, visibility=0.99)
    box = _hand_roi_box(
        wrist, elbow, 640, 360, scale=2.6, min_box_px=64, side="Left"
    )
    assert box is not None
    assert box.x0 + box.width <= 640
    assert box.width >= 64
    inner = _hand_roi_box(
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99),
        Landmark(x=0.5, y=0.35, z=0.0, visibility=0.99),
        640,
        360,
        scale=2.6,
        min_box_px=64,
        side="Left",
    )
    assert inner is not None
    assert abs(box.width - inner.width) <= 1


def test_roi_box_respects_min_size_for_tiny_hand() -> None:
    wrist = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99)
    elbow = Landmark(x=0.5, y=0.495, z=0.0, visibility=0.99)
    box = _hand_roi_box(wrist, elbow, 640, 360, scale=2.6, min_box_px=64, side="Left")
    assert box is not None
    assert min(box.width, box.height) >= 64


def test_roi_coordinates_map_back_to_full_frame() -> None:
    frame = make_frames(1)[0]
    box = _hand_roi_box(
        _wrist_landmark(frame, RIGHT_WRIST),
        _wrist_landmark(frame, LEFT_ELBOW),
        640,
        360,
        scale=2.6,
        min_box_px=64,
        side="Right",
    )
    assert box is not None
    point = _FakePoint(0.25, 0.75, 0.1)
    mapped = _roi_to_landmark(point, box, 640, 360)
    assert mapped.x == pytest.approx((box.x0 + 0.25 * box.width) / 640)
    assert mapped.y == pytest.approx((box.y0 + 0.75 * box.height) / 360)
    # z 按 ROI 宽度比例换算，不能直接沿用裁剪内的归一化深度
    assert mapped.z == pytest.approx(0.1 * box.width / 640)
    # 手部增强没有 visibility 字段，检测到的点按可见处理
    assert mapped.visibility == 1.0


# --------------------------------------------------------------------------- #
# 误检过滤
# --------------------------------------------------------------------------- #
def _box():
    return _hand_roi_box(
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99),
        Landmark(x=0.5, y=0.35, z=0.0, visibility=0.99),
        640,
        360,
        scale=2.6,
        min_box_px=64,
        side="Left",
    )


def _hand(wrist_uv: tuple[float, float], span: float = 0.09) -> list[_FakePoint]:
    points = [_FakePoint(wrist_uv[0], wrist_uv[1]) for _ in range(HAND_LANDMARK_COUNT)]
    points[9] = _FakePoint(wrist_uv[0], wrist_uv[1] + span)
    return points


def test_closest_hand_rejects_far_wrist() -> None:
    box = _box()
    assert box is not None
    near = _hand(box.wrist_uv)
    far = _hand((box.wrist_uv[0] + 0.6, box.wrist_uv[1]))
    assert _closest_hand([far, near], box) is near


def test_closest_hand_rejects_implausible_size() -> None:
    box = _box()
    assert box is not None
    tiny = _hand(box.wrist_uv, span=0.001)
    assert _closest_hand([tiny], box) is None


# --------------------------------------------------------------------------- #
# 流程：ROI 优先、整帧兜底
# --------------------------------------------------------------------------- #
def _refiner(hits: list[bool]) -> MediaPipeRefiner:
    refiner = MediaPipeRefiner(RefinementConfig(hands=True, face=False))
    refiner._prepared = True
    refiner._mp = _FakeMp
    refiner._hand = _FakeHand(hits)
    return refiner


def test_infer_uses_wrist_roi_and_keeps_pose_side_labels() -> None:
    frame = make_frames(1)[0]
    # 只让第一只（左侧）命中，右手全部漏检
    refiner = _refiner([True, False])
    result = refiner.infer(_image(), [frame])

    # 假检测器的 handedness 一律报 "Right"；ROI 路径必须用姿态给的左右侧覆盖它。
    assert result.hand_sides == ["Left"]
    assert len(result.hands) == 1
    assert len(result.hands[0]) == HAND_LANDMARK_COUNT
    assert refiner.stats["roiHits"] == 1
    assert refiner.stats["roiHands"] == 2
    assert refiner.stats["fullFallbacks"] == 0


def test_infer_walks_scale_ladder_until_hit() -> None:
    frame = make_frames(1)[0]
    # 左手：第一个尺度漏检、第二个命中；右手所有尺度都漏。
    # 640×360 下最大那片 ROI 已超过画面短边的 60%，会被跳过，所以每只手只有 2 次裁剪。
    refiner = _refiner([False, True, False, False])
    result = refiner.infer(_image(), [frame])
    assert len(result.hands) == 1
    assert refiner.stats["roiCrops"] == 4
    assert refiner.stats["fullFallbacks"] == 0


def test_oversized_roi_is_capped_by_frame_share() -> None:
    """画面越小，可用尺度越少——大框等于整帧检测，没有意义还多花算力。"""
    frame = make_frames(1)[0]
    big = _refiner([False, False, False, False, False, False])
    big.infer(np.zeros((720, 1280, 3), dtype=np.uint8), [frame])
    small = _refiner([False, False])
    small.infer(np.zeros((240, 240, 3), dtype=np.uint8), [frame])
    assert big.stats["roiCrops"] >= 2
    assert small.stats["roiCrops"] >= 1
    assert big.stats["roiCrops"] >= small.stats["roiCrops"]


def test_infer_falls_back_to_full_frame_when_roi_misses() -> None:
    frame = make_frames(1)[0]
    config = RefinementConfig(hands=True, face=False, hand_roi_scales=(2.6,))
    refiner = MediaPipeRefiner(config)
    refiner._prepared = True
    refiner._mp = _FakeMp
    # 左右腕两次 ROI 都漏，随后整帧兜底命中（第三次调用）
    refiner._hand = _FakeHand([False, False, True])
    result = refiner.infer(_image(), [frame])
    assert refiner.stats["fullFallbacks"] == 1
    assert len(result.hands) == 1
    assert refiner.stats["roiHits"] == 0


def test_infer_without_pose_frames_still_works() -> None:
    """不传姿态时退回整帧检测（旧行为），保证外部调用方不被破坏。"""
    refiner = _refiner([True])
    result = refiner.infer(_image())
    assert len(result.hands) == 1
    assert refiner.stats["roiHands"] == 0
    assert refiner.stats["fullHits"] == 1


def test_skips_sides_with_invisible_wrist() -> None:
    frame = make_frames(1)[0]
    for index in (LEFT_WRIST, RIGHT_WRIST):
        frame.landmarks[index].visibility = 0.0
    refiner = _refiner([True])
    refiner.infer(_image(), [frame])
    assert refiner.stats["wristSkipped"] == 2
    assert refiner.stats["roiHands"] == 0
    # 腕点看不见时 ROI 一次都不裁，直接走整帧兜底
    assert refiner.stats["fullFallbacks"] == 1


def test_stats_note_reports_hit_rate() -> None:
    frame = make_frames(1)[0]
    refiner = _refiner([True, False])
    refiner.infer(_image(), [frame])
    note = refiner.stats_note()
    assert len(note) == 1
    assert "1/2" in note[0]


def test_landmark_count_matches_pose_frame_size() -> None:
    frame = make_frames(1)[0]
    assert len(frame.landmarks) == LANDMARK_COUNT
