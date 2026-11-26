"""One-Euro 平滑（任务 5.1，默认滤波器）。

参考 Casiez et al. 2012：``cutoff = min_cutoff + beta * |dx_hat|``，
低速时强平滑压抖动，高速时自动放开延迟。
只平滑位置（x/y/z），**不平滑 visibility** —— 置信度是观测事实，不能被滤波。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame

logger = get_logger("postprocess.one_euro")

DEFAULT_FPS = 30.0

# 逐轴参数（任务 4.1 / 4.2）
# 为什么 z 要单独一套：单目视频的 z 是模型猜的，噪声量级比 x/y **大一个数量级**
# （合成数据实测 z 的相邻帧 RMS ≈ 0.03 m，x/y ≈ 0.006 m）。用同一组 cutoff 时，
# x/y 需要的"跟手"力度会把 z 的抖动原样放过去，所以：
# - z 的 min_cutoff 取 x/y 的 30%（1.0 → 0.3），低速时把抖动压得更狠；
# - z 的 beta 同比降到 30%（0.5 → 0.15），避免快速动作时 z 又被放开成抖动；
# - d_cutoff 保持与 x/y 相同（它是"速度估计"的低通，与位置噪声量级无关）。
# 三个系数都做成常量，方便基准脚本（`scripts/bench_depth.py`）按需改。
Z_MIN_CUTOFF_SCALE = 0.3
Z_BETA_SCALE = 0.3


class OneEuroParams(BaseModel):
    """One-Euro 参数。

    ``min_cutoff`` / ``beta`` / ``d_cutoff`` 作用于 x/y；``*_z`` 作用于 z，
    留 ``None`` 表示**沿用 x/y 的值** —— 这样老代码
    （``OneEuroParams(min_cutoff=1.0, beta=0.5)``）的输出逐位不变（任务 4.5 的向后兼容）。
    生产默认请用 :func:`default_one_euro_params` / :func:`default_world_one_euro_params`。
    """

    min_cutoff: float = Field(1.0, gt=0, le=20)
    beta: float = Field(0.5, ge=0, le=10)
    d_cutoff: float = Field(1.0, gt=0, le=20)
    min_cutoff_z: float | None = Field(None, gt=0, le=20, description="z 轴 min_cutoff；None 沿用 min_cutoff")
    beta_z: float | None = Field(None, ge=0, le=10, description="z 轴 beta；None 沿用 beta")
    d_cutoff_z: float | None = Field(None, gt=0, le=20, description="z 轴 d_cutoff；None 沿用 d_cutoff")

    def z_params(self) -> OneEuroParams:
        """z 轴实际使用的参数（把 None 解析成 x/y 的值）。"""
        return OneEuroParams(
            min_cutoff=self.min_cutoff if self.min_cutoff_z is None else self.min_cutoff_z,
            beta=self.beta if self.beta_z is None else self.beta_z,
            d_cutoff=self.d_cutoff if self.d_cutoff_z is None else self.d_cutoff_z,
        )

    def axis_params(self, axis: int) -> OneEuroParams:
        """按轴取参数：``axis == 2`` 用 z 的一套，其余用 x/y 的一套。"""
        return self.z_params() if axis == 2 else self

    def to_dict(self) -> dict[str, float]:
        data = {
            "minCutoff": self.min_cutoff,
            "beta": self.beta,
            "dCutoff": self.d_cutoff,
        }
        if self.min_cutoff_z is not None:
            data["minCutoffZ"] = self.min_cutoff_z
        if self.beta_z is not None:
            data["betaZ"] = self.beta_z
        if self.d_cutoff_z is not None:
            data["dCutoffZ"] = self.d_cutoff_z
        return data


def default_one_euro_params() -> OneEuroParams:
    """**归一化坐标**下的生产默认值：x/y = 1.0/0.5，z = 0.3/0.15。"""
    base = OneEuroParams()
    return base.model_copy(
        update={
            "min_cutoff_z": round(base.min_cutoff * Z_MIN_CUTOFF_SCALE, 4),
            "beta_z": round(base.beta * Z_BETA_SCALE, 4),
            "d_cutoff_z": base.d_cutoff,
        }
    )


def default_world_one_euro_params() -> OneEuroParams:
    """**米制世界坐标**下的生产默认值（任务 4.3）。

    世界点的单位是米，相邻帧位移比归一化坐标大两个数量级（0.03 m vs 0.0003），
    所以 beta 要按"米/秒"重新标：``beta=1.0`` 对应 1 m/s 时 cutoff 抬到 2 Hz，
    刚好跟得上挥手 / 跨步，又不会把静止时的抖动放过去。
    """
    base = OneEuroParams(min_cutoff=1.0, beta=1.0, d_cutoff=1.0)
    return base.model_copy(
        update={
            "min_cutoff_z": round(base.min_cutoff * Z_MIN_CUTOFF_SCALE, 4),
            "beta_z": round(base.beta * Z_BETA_SCALE, 4),
            "d_cutoff_z": base.d_cutoff,
        }
    )


def default_hand_one_euro_params() -> OneEuroParams:
    """**手部归一化点**的生产默认值（``postprocess/hands.py``）。

    手只有 21 个点、尺寸小、关节多，握拳/张开的指尖速度比躯干大一个量级；
    用身体的 ``min_cutoff=1.0 / beta=0.5`` 会把手指动作抹成"半握"。
    这里把基准截止频率抬到 1.5 Hz、beta 抬到 1.5：静止时仍能压抖，快速屈伸跟得住。
    z 的深度估计更不可靠，继续按 30% 缩放（与身体一致）。
    """
    base = OneEuroParams(min_cutoff=1.5, beta=1.5, d_cutoff=1.0)
    return base.model_copy(
        update={
            "min_cutoff_z": round(base.min_cutoff * Z_MIN_CUTOFF_SCALE, 4),
            "beta_z": round(base.beta * Z_BETA_SCALE, 4),
            "d_cutoff_z": base.d_cutoff,
        }
    )


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter1D:
    """标量滤波器（每个关键点的每个轴一个实例）。"""

    def __init__(self, params: OneEuroParams | None = None) -> None:
        self.params = params or OneEuroParams()
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0

    def filter(self, value: float, dt: float) -> float:
        if not math.isfinite(value):
            return self._x if self._x is not None else 0.0
        if dt <= 0:
            dt = 1.0 / DEFAULT_FPS
        if self._x is None:
            self._x = float(value)
            self._dx = 0.0
            return self._x

        raw_dx = (value - self._x) / dt
        a_d = _alpha(self.params.d_cutoff, dt)
        self._dx += a_d * (raw_dx - self._dx)

        cutoff = self.params.min_cutoff + self.params.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x += a * (value - self._x)
        return self._x


class OneEuroVectorFilter:
    """多维滤波器（一次处理一个关键点的 x/y/z）。

    逐轴参数：``axis == 2``（z）用 ``params.z_params()`` 单独的一套实例，
    x/y 共用原参数。``filter(values)`` 的签名与行为保持不变。
    """

    def __init__(self, dims: int = 3, params: OneEuroParams | None = None) -> None:
        self.params = params or OneEuroParams()
        self._filters = [OneEuroFilter1D(self.params.axis_params(axis)) for axis in range(dims)]

    def reset(self) -> None:
        for item in self._filters:
            item.reset()

    def filter(self, values: Sequence[float], dt: float) -> np.ndarray:
        return np.asarray(
            [item.filter(float(v), dt) for item, v in zip(self._filters, values, strict=False)],
            dtype=np.float32,
        )


def timestamps_to_dt(
    timestamps_ms: Sequence[float], fps: float = DEFAULT_FPS
) -> list[float]:
    """时间戳 → 每帧 dt（秒），异常值回退 ``1/fps``。"""
    default = 1.0 / fps if fps > 0 else 1.0 / DEFAULT_FPS
    deltas: list[float] = []
    previous: float | None = None
    for stamp in timestamps_ms:
        if previous is None:
            deltas.append(default)
        else:
            dt = (float(stamp) - previous) / 1000.0
            deltas.append(dt if 1e-4 < dt < 1.0 else default)
        previous = float(stamp)
    return deltas


def smooth_array(
    array: np.ndarray, timestamps_ms: Sequence[float], params: OneEuroParams | None = None
) -> np.ndarray:
    """``(T, 33, 4)`` → 平滑后的同形状数组（visibility 列原样保留）。

    ``params`` 留空时用生产默认值（x/y 与 z 分开配置，见 :func:`default_one_euro_params`）。
    """
    params = params or default_one_euro_params()
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 3 or data.shape[1] != LANDMARK_COUNT:
        raise ValueError(f"期望 (T, {LANDMARK_COUNT}, 4) 的数组，收到 {data.shape}")

    deltas = timestamps_to_dt(timestamps_ms)
    result = data.copy()
    filters = [OneEuroVectorFilter(3, params) for _ in range(LANDMARK_COUNT)]
    for t in range(data.shape[0]):
        dt = deltas[t]
        for index, flt in enumerate(filters):
            result[t, index, :3] = flt.filter(data[t, index, :3], dt)
    return result


def smooth_array_zero_phase(
    array: np.ndarray, timestamps_ms: Sequence[float], params: OneEuroParams | None = None
) -> np.ndarray:
    """前向 + 后向 One-Euro（零相位近似）。

    单向 One-Euro 是**因果**滤波：输出必然滞后输入，且"压抖"与"跟手"只能二选一。
    把序列倒过来再滤一遍，两次的相位延迟互相抵消，同一组参数能再压低一截抖动，
    而快速动作的峰值位置不会被推移（``angle_drift`` / 峰值时刻不变）。

    代价是**非因果**：必须拿到整段序列才能算，不适合逐帧实时预览；
    本项目是"整段视频离线分析"，正好不冲突。
    """
    params = params or default_one_euro_params()
    data = np.asarray(array, dtype=np.float32)
    if data.shape[0] < 3:
        return data.copy()
    forward = smooth_array(data, timestamps_ms, params)
    stamps = [float(stamp) for stamp in timestamps_ms]
    backward = smooth_array(forward[::-1].copy(), list(reversed(stamps)), params)
    return np.ascontiguousarray(backward[::-1])


def smooth_frames(
    frames: Sequence[PoseFrame],
    params: OneEuroParams | None = None,
    *,
    use_world: bool = False,
) -> list[PoseFrame]:
    """对连续帧做 One-Euro 平滑，返回新的 ``PoseFrame`` 列表。

    ``use_world=True`` 时只动 ``world_landmarks``；没有世界坐标的帧原样返回
    （**不能**回退去写归一化点 —— 米制数值写进无量纲字段会把坐标彻底搞坏）。
    """
    params = params or default_one_euro_params()
    if not frames:
        return []
    if len(frames) == 1:
        return [frame.model_copy(deep=True) for frame in frames]

    ordered = sorted(frames, key=lambda item: item.index)
    usable = [frame for frame in ordered if not use_world or frame.world_landmarks]
    result = [frame.model_copy(deep=True) for frame in ordered]
    if len(usable) < 2:
        return result

    array = np.stack([frame.as_array(use_world=use_world) for frame in usable], axis=0)
    smoothed = smooth_array(array, [frame.timestamp_ms for frame in usable], params)

    position_of = {frame.index: position for position, frame in enumerate(ordered)}
    for row, frame in enumerate(usable):
        copy = result[position_of[frame.index]]
        source = copy.world_landmarks if use_world else copy.landmarks
        for index, landmark in enumerate(source):
            landmark.x = float(smoothed[row, index, 0])
            landmark.y = float(smoothed[row, index, 1])
            landmark.z = float(smoothed[row, index, 2])
    return result
