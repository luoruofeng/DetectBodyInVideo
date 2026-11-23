"""层级与根运动（任务 6.3）。

- ``translation``：髋中心相对首帧的位移（米），可直接挂到 ``ModelEntity`` 的 root 上。
- ``rotation``：绕 Y 轴的四元数（yaw），由左右髋（或左右肩）连线求得；
  ``yaw = 0`` 表示人物正面朝相机（左右髋连线与 +X 重合）。
- 导出模式：
  - ``world``：带位移播放（默认）；
  - ``in_place``：固定 root（位移归零，只保留旋转），适合原地动画/卡点视频。

注意：MediaPipe world landmarks 以髋为原点，绝对位移会被抹掉；
竖直方向由脚部着地锚定补回下蹲下沉，水平方向由**脚底水平锚定**反解出
"脚踩住地面时髋相对脚怎么动"（脚不滑步，髋绕脚晃）；
两者都不是绝对走位：人物真的整体平移时，水平分量会被吃掉一部分；
``total_translation`` 是首尾净位移，不是累计路程。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.backend.core.logging import get_logger
from app.backend.skeleton.transform import ConvertedFrame, Vector3

logger = get_logger("skeleton.root_motion")

YAW_CONVENTION = "yaw=0 时左右髋连线指向 +X（人物正面朝 -Z / 相机）"


class Quaternion(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def normalized(self) -> Quaternion:
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if length < 1e-9:
            return Quaternion()
        return Quaternion(
            x=self.x / length, y=self.y / length, z=self.z / length, w=self.w / length
        )


class RootMotionFrame(BaseModel):
    index: int = 0
    t: float = 0.0
    translation: Vector3 = Field(default_factory=Vector3)
    rotation: Quaternion = Field(default_factory=Quaternion)


class RootMotionConfig(BaseModel):
    mode: Literal["world", "in_place"] = "world"
    yaw_source: Literal["hips", "shoulders"] = "hips"
    normalize_start: bool = Field(True, description="以首帧为原点（推荐，避免绝对坐标系漂移）")
    round_digits: int = Field(4, ge=0, le=8)


class RootMotionTrack(BaseModel):
    mode: Literal["world", "in_place"] = "world"
    frames: list[RootMotionFrame] = Field(default_factory=list)
    total_translation: float = 0.0
    convention: str = YAW_CONVENTION
    notes: list[str] = Field(default_factory=list)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    half = yaw / 2.0
    return Quaternion(x=0.0, y=math.sin(half), z=0.0, w=math.cos(half))


def yaw_between(left: Vector3, right: Vector3) -> float | None:
    """左右点连线 → 绕 Y 轴 yaw（弧度）。两点重合时返回 ``None``。"""
    dx = right.x - left.x
    dz = right.z - left.z
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return None
    return math.atan2(-dz, dx)


def unwrap_angles(angles: Sequence[float | None]) -> list[float]:
    """把 yaw 展开到连续区间，避免 ±180° 处四元数跳变（缺失帧沿用上一角度）。"""
    result: list[float] = []
    previous: float | None = None
    for value in angles:
        if value is None:
            result.append(previous if previous is not None else 0.0)
            continue
        current = value
        if previous is not None:
            while current - previous > math.pi:
                current -= 2 * math.pi
            while current - previous < -math.pi:
                current += 2 * math.pi
        result.append(current)
        previous = current
    return result


def _yaw_of(frame: ConvertedFrame, yaw_source: str) -> float | None:
    if yaw_source == "shoulders":
        left = frame.joints.get("left_shoulder_1_joint")
        right = frame.joints.get("right_shoulder_1_joint")
    else:
        left = frame.joints.get("left_upLeg_joint")
        right = frame.joints.get("right_upLeg_joint")
    if left is None or right is None:
        return None
    return yaw_between(left.to_vector(), right.to_vector())


def build_root_motion(
    frames: Sequence[ConvertedFrame], config: RootMotionConfig | None = None
) -> RootMotionTrack:
    config = config or RootMotionConfig()
    ordered = sorted((frame for frame in frames if frame.valid), key=lambda item: item.index)
    track = RootMotionTrack(mode=config.mode, convention=YAW_CONVENTION)
    if not ordered:
        track.notes.append("没有有效帧，rootMotion 为空")
        return track

    origin = ordered[0].root
    yaws = unwrap_angles([_yaw_of(frame, config.yaw_source) for frame in ordered])

    for position, frame in enumerate(ordered):
        translation = frame.root - origin if config.normalize_start else frame.root
        if config.mode == "in_place":
            translation = Vector3()

        rotation = quaternion_from_yaw(yaws[position]).normalized()

        track.frames.append(
            RootMotionFrame(
                index=frame.index,
                t=round(frame.t, 4),
                translation=Vector3(
                    x=round(translation.x, config.round_digits),
                    y=round(translation.y, config.round_digits),
                    z=round(translation.z, config.round_digits),
                ),
                rotation=Quaternion(
                    x=round(rotation.x, config.round_digits),
                    y=round(rotation.y, config.round_digits),
                    z=round(rotation.z, config.round_digits),
                    w=round(rotation.w, config.round_digits),
                ),
            )
        )

    last = ordered[-1].root - origin
    track.total_translation = round(math.sqrt(last.x**2 + last.y**2 + last.z**2), config.round_digits)
    if config.mode == "in_place":
        track.notes.append("原地模式：translation 全为 0，仅保留旋转")
    else:
        track.notes.append(
            "world landmarks 以髋为原点：y 位移来自脚部着地锚定的髋部下沉，"
            "x/z 来自脚底水平锚定（脚踩住地面时髋相对脚的运动），不是绝对走位"
        )

    return track


def split_translation_rotation(
    track: RootMotionTrack,
) -> tuple[list[Vector3], list[Quaternion]]:
    """导出端通常分两条通道写：位移给 root 节点，旋转给髋节点。"""
    return [frame.translation for frame in track.frames], [frame.rotation for frame in track.frames]


__all__ = [
    "YAW_CONVENTION",
    "Quaternion",
    "RootMotionConfig",
    "RootMotionFrame",
    "RootMotionTrack",
    "build_root_motion",
    "quaternion_from_yaw",
    "split_translation_rotation",
    "unwrap_angles",
    "yaw_between",
]
