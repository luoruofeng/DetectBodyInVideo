"""平滑算法集合与调度（任务 5.1）。

- ``one-euro``：默认，自适应、延迟低。
- ``moving-average``：最直观，作为对照组（会让快速动作变钝）。
- ``savitzky-golay``：保峰值（跳跃、投掷类动作更友好），纯 numpy 实现。

导出时把所用方法与参数原样记录到 JSON 的 ``quality.smoothing``，保证可复现。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame
from app.backend.postprocess.one_euro import (
    OneEuroParams,
    default_one_euro_params,
    default_world_one_euro_params,
)
from app.backend.postprocess.one_euro import smooth_array as _one_euro_array

logger = get_logger("postprocess.smoothing")

SmoothingMethod = Literal["none", "one-euro", "moving-average", "savitzky-golay"]


class SmoothingConfig(BaseModel):
    method: SmoothingMethod = "one-euro"
    one_euro: OneEuroParams = Field(
        default_factory=default_one_euro_params,
        description="归一化坐标参数（x/y 与 z 分开配置，z 更弱）",
    )
    window: int = Field(5, ge=3, le=31, description="移动平均 / SG 的窗口帧数（奇数）")
    polyorder: int = Field(2, ge=1, le=5, description="SG 多项式阶数")
    use_world_landmarks: bool = Field(
        False, description="只平滑 world_landmarks（归一化点不动）；一般用下面的 extra_world_pass"
    )
    extra_world_pass: bool = Field(
        True,
        description="在归一化点之外，再对 world_landmarks 单独平滑一次（任务 4.3）",
    )
    world_one_euro: OneEuroParams = Field(
        default_factory=default_world_one_euro_params,
        description="world_landmarks 的米制参数，不能与归一化坐标共用同一组数值",
    )


class SmoothingInfo(BaseModel):
    applied: bool = False
    method: SmoothingMethod = "none"
    params: dict[str, object] = Field(default_factory=dict)
    frames: int = 0
    notes: list[str] = Field(default_factory=list)


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """边界收缩的移动平均：不补零、不镜像，避免首尾被拉向 0。"""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
        squeeze = True
    else:
        squeeze = False
    length = data.shape[0]
    window = max(1, min(int(window), length))
    if length == 0 or window <= 1:
        result = data.astype(np.float32)
        return result[:, 0] if squeeze else result

    cumsum = np.cumsum(np.insert(data, 0, 0.0, axis=0), axis=0)
    result = np.empty_like(data)
    half = window // 2
    for i in range(length):
        low = max(0, i - half)
        high = min(length, i + half + 1)
        result[i] = (cumsum[high] - cumsum[low]) / (high - low)
    result = result.astype(np.float32)
    return result[:, 0] if squeeze else result


def savitzky_golay(values: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """一维/二维（T, D）SG 滤波；窗口不足时原样返回。"""
    data = np.asarray(values, dtype=np.float64)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[:, None]
    length = data.shape[0]
    if length < 3:
        return data.astype(np.float32)[:, 0] if squeeze else data.astype(np.float32)

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, length if length % 2 == 1 else length - 1)
    if window < 3 or polyorder >= window:
        logger.debug("SG 窗口/阶数不合法（%s/%s），跳过平滑", window, polyorder)
        return data.astype(np.float32)[:, 0] if squeeze else data.astype(np.float32)

    half = window // 2
    padded = np.pad(data, ((half, half), (0, 0)), mode="edge")
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(design)

    result = np.empty_like(data)
    for i in range(length):
        window_values = padded[i : i + window]
        coefficients = pinv @ window_values  # (polyorder+1, D)
        result[i] = coefficients[0]  # x=0 处的拟合值
    result = result.astype(np.float32)
    return result[:, 0] if squeeze else result


def _filter_array(
    array: np.ndarray,
    timestamps_ms: Sequence[float],
    config: SmoothingConfig,
    *,
    params: OneEuroParams | None = None,
) -> np.ndarray:
    if config.method == "one-euro":
        return _one_euro_array(array, timestamps_ms, params or config.one_euro)

    result = array.copy()
    for landmark_index in range(array.shape[1]):
        for axis in range(3):
            column = array[:, landmark_index, axis]
            smoothed = (
                moving_average(column, config.window)
                if config.method == "moving-average"
                else savitzky_golay(column, config.window, config.polyorder)
            )
            result[:, landmark_index, axis] = smoothed
    return result


def smooth_frames(
    frames: Sequence[PoseFrame],
    config: SmoothingConfig | None = None,
    *,
    use_world: bool | None = None,
    params: OneEuroParams | None = None,
) -> tuple[list[PoseFrame], SmoothingInfo]:
    """平滑位置，visibility 保持不变。

    ``use_world`` / ``params`` 用于"第二趟"：在归一化点平滑完之后，
    再用米制参数对 ``world_landmarks`` 单独平滑一次（任务 4.3）。
    没有世界坐标的帧原样返回，不会被写坏。
    """
    config = config or SmoothingConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    if config.method == "none" or len(ordered) < 3:
        return [frame.model_copy(deep=True) for frame in ordered], SmoothingInfo(
            applied=False,
            method=config.method,
            frames=len(ordered),
            notes=["帧数不足 3，跳过平滑"] if config.method != "none" else [],
        )

    use_world = config.use_world_landmarks if use_world is None else use_world
    usable = [frame for frame in ordered if not use_world or frame.world_landmarks]
    result = [frame.model_copy(deep=True) for frame in ordered]
    if len(usable) < 3:
        return result, SmoothingInfo(
            applied=False,
            method=config.method,
            frames=len(ordered),
            notes=["没有可用的 world_landmarks，跳过世界点平滑"]
            if use_world
            else ["没有有效点"],
        )

    array = np.stack([frame.as_array(use_world=use_world) for frame in usable], axis=0)
    if not np.any(array[:, :, 3] > 0):
        return result, SmoothingInfo(
            applied=False, method=config.method, frames=len(ordered), notes=["没有有效点"]
        )

    smoothed = _filter_array(
        array, [frame.timestamp_ms for frame in usable], config, params=params
    )

    position_of = {frame.index: position for position, frame in enumerate(ordered)}
    for row, frame in enumerate(usable):
        copy = result[position_of[frame.index]]
        source = copy.world_landmarks if use_world else copy.landmarks
        for index, landmark in enumerate(source):
            if index >= LANDMARK_COUNT:
                break
            landmark.x = float(smoothed[row, index, 0])
            landmark.y = float(smoothed[row, index, 1])
            landmark.z = float(smoothed[row, index, 2])

    active = params or config.one_euro
    params_out: dict[str, object] = {
        "window": config.window,
        "polyorder": config.polyorder,
    }
    if config.method == "one-euro":
        params_out = dict(active.to_dict())

    return result, SmoothingInfo(
        applied=True,
        method=config.method,
        params=params_out,
        frames=len(usable),
        notes=[
            f"仅平滑位置，visibility 未改动；坐标系={'world' if use_world else 'normalized'}"
        ],
    )


def smooth_world_pass(
    frames: Sequence[PoseFrame], config: SmoothingConfig | None = None
) -> tuple[list[PoseFrame], SmoothingInfo]:
    """任务 4.3：对 ``world_landmarks`` 的独立平滑（米制参数）。

    归一化点是无量纲、世界点是米制，两者的噪声量级与速度量级都不同，
    所以必须用 ``config.world_one_euro`` 而不是 ``config.one_euro``。
    """
    config = config or SmoothingConfig()
    return smooth_frames(frames, config, use_world=True, params=config.world_one_euro)


def estimate_jitter(frames: Sequence[PoseFrame], *, landmark_index: int = 23) -> float:
    """用二阶差分均值估计抖动强度（越小越稳），供 UI 对比滤波前后。"""
    values = [frame.landmark(landmark_index) for frame in frames]
    points = np.asarray(
        [(item.x, item.y) for item in values if item is not None], dtype=np.float64
    )
    if points.shape[0] < 3:
        return 0.0
    second = np.diff(points, n=2, axis=0)
    return float(np.mean(np.linalg.norm(second, axis=1)))


def auto_window(fps: float, *, seconds: float = 0.15) -> int:
    """按帧率把"半秒级"窗口换算成奇数帧数，避免不同 fps 参数不可比。"""
    window = max(3, int(round(fps * seconds)))
    return window if window % 2 == 1 else window + 1


def smoothness_gain(before: Sequence[PoseFrame], after: Sequence[PoseFrame]) -> float:
    """抖动下降比例（0~1），用于质量报告展示。"""
    old = estimate_jitter(before)
    new = estimate_jitter(after)
    if old <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - new / old)))


__all__ = [
    "SmoothingConfig",
    "SmoothingInfo",
    "SmoothingMethod",
    "auto_window",
    "estimate_jitter",
    "moving_average",
    "savitzky_golay",
    "smooth_frames",
    "smooth_world_pass",
    "smoothness_gain",
]
