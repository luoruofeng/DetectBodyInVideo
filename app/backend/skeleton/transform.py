"""坐标转换模块（任务 6.2）。

唯一路径：MediaPipe / ONNX 引擎给的 **world landmarks**（已是米制、原点≈两髋中点），
精确减去髋中心，再做轴向翻转 ``x'=x, y'=-y, z'=-z``。人物整体位移不可恢复。

历史上的 ``calibrated`` 路径（"人物身高 ÷ 归一化头脚像素跨度"反解米制尺度）已删除：
该尺度的误差随姿态（下蹲/抬手）、机位和人物远近放大，实测效果远不如直接用 world 点。
没有 world landmarks 的引擎因此不再产出米制关节（帧记 ``valid=False``）。

输出坐标系严格对齐 RealityKit：**米制 / 右手系 / y 向上 / -z 向前**。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.skeleton.mapping import SkeletonFrame

logger = get_logger("skeleton.transform")

COORDINATE_SYSTEM = "arkit-right-handed-y-up"
UNITS = "meters"


class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def scaled(self, factor: float) -> Vector3:
        return Vector3(x=self.x * factor, y=self.y * factor, z=self.z * factor)

    @property
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class JointPosition(Vector3):
    visibility: float = Field(0.0, ge=0, le=1)
    source: Literal["direct", "derived", "estimated"] = "direct"

    def to_vector(self) -> Vector3:
        return Vector3(x=self.x, y=self.y, z=self.z)


class ConvertedFrame(BaseModel):
    """joints 为髋中心相对位置，不是逐骨段的父关节局部坐标；root 单独存平移。"""

    index: int = 0
    t: float = 0.0
    valid: bool = True
    interpolated: bool = False
    track_id: int | None = None
    joints: dict[str, JointPosition] = Field(default_factory=dict)
    root: Vector3 = Field(default_factory=Vector3)


def to_realitykit_vector(point: Vector3) -> Vector3:
    """``x'=x, y'=-y, z'=-z``；该操作自逆，可用于往返校验。"""
    return Vector3(x=point.x, y=-point.y, z=-point.z)


def from_realitykit_vector(point: Vector3) -> Vector3:
    return to_realitykit_vector(point)


class CoordinateTransform:
    """把世界坐标 ``SkeletonFrame`` 转成 RealityKit 可用的米制关节。"""

    def convert(self, world: SkeletonFrame) -> ConvertedFrame:
        """只接受 ``coordinate_space="world"`` 的帧；其它情况返回 ``valid=False``。"""
        if world.coordinate_space != "world" or not world.valid or not world.joints:
            return _unavailable(world)
        root_point = world.joint("hips_joint") or world.joint("root")
        if root_point is None:
            return _unavailable(world)

        joints: dict[str, JointPosition] = {}
        for name, point in world.joints.items():
            flipped = to_realitykit_vector(Vector3(
                x=point.x - root_point.x,
                y=point.y - root_point.y,
                z=point.z - root_point.z,
            ))
            joints[name] = JointPosition(
                x=flipped.x,
                y=flipped.y,
                z=flipped.z,
                visibility=point.visibility,
                source=point.source,
            )
        return ConvertedFrame(
            index=world.index,
            t=world.timestamp_ms / 1000.0,
            valid=world.valid,
            interpolated=world.interpolated,
            track_id=world.track_id,
            joints=joints,
            root=Vector3(),
        )


def _unavailable(world: SkeletonFrame) -> ConvertedFrame:
    """缺少 world landmarks 的帧：保留帧号/时间轴，但没有任何米制关节。"""
    return ConvertedFrame(
        index=world.index,
        t=world.timestamp_ms / 1000.0,
        valid=False,
        interpolated=world.interpolated,
        track_id=world.track_id,
    )


def convert_frames(world: list[SkeletonFrame]) -> list[ConvertedFrame]:
    """整段转换；入参必须是 ``map_frames(..., use_world=True)`` 的结果。"""
    transform = CoordinateTransform()
    return [transform.convert(frame) for frame in sorted(world, key=lambda item: item.index)]


def roundtrip_error(point: Vector3) -> float:
    """往返一致性误差（任务 11.1 的断言基准）。"""
    back = from_realitykit_vector(to_realitykit_vector(point))
    return (back - point).length


__all__ = [
    "COORDINATE_SYSTEM",
    "UNITS",
    "ConvertedFrame",
    "CoordinateTransform",
    "JointPosition",
    "Vector3",
    "convert_frames",
    "from_realitykit_vector",
    "roundtrip_error",
    "to_realitykit_vector",
]
