"""推理进行中的增量骨骼映射（边算边看的 3D 骨架 / 白膜 Mesh 预览）。

平滑、补帧、IK 拟合这些后处理必须等整段推理结束才开始，长视频要干等十几分钟。
在那之前帧数据只有 2D 关键点，导致 3D 预览卡片与白膜 Mesh 卡片一直是空的。
这里对"已经识别出来的帧"做与最终结果同一套 33 点 → ARKit 关节转换（不做平滑/拟合），
让这两张卡片在推理阶段就能跟着播放头同步。

内存上刻意不保存 ``ConvertedFrame``（每帧 22 个 Pydantic 对象，两万帧就到百兆级），
只按固定顺序存扁平坐标元组，取帧时再按请求窗口还原成接口模型。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.backend.core.logging import get_logger
from app.backend.engine.base import LANDMARK_COUNT, PoseFrame
from app.backend.postprocess.angles import AngleConfig, compute_angles
from app.backend.skeleton.bundle import SkeletonConfig
from app.backend.skeleton.mapping import joint_names, map_frame
from app.backend.skeleton.transform import ConvertedFrame, CoordinateTransform

logger = get_logger("skeleton.live")

# 每帧每个关节存 (x, y, z, visibility) 四个浮点
_STRIDE = 4


@dataclass(slots=True)
class LiveFrameJoints:
    """单帧的轻量关节快照：``coords`` 按 ``joint_names`` 顺序展开为 x,y,z,visibility。"""

    index: int
    coords: tuple[float, ...]
    angles: dict[str, float] = field(default_factory=dict)

    def as_joints(self, names: Sequence[str]) -> list[tuple[str, float, float, float, float]]:
        values = self.coords
        result: list[tuple[str, float, float, float, float]] = []
        for position, name in enumerate(names):
            offset = position * _STRIDE
            if offset + _STRIDE > len(values):
                break
            result.append(
                (name, values[offset], values[offset + 1], values[offset + 2], values[offset + 3])
            )
        return result


class LiveSkeletonMapper:
    """按帧追加的增量映射器：只处理新帧，避免每次轮询重算整段。"""

    def __init__(
        self,
        config: SkeletonConfig | None = None,
        angle_config: AngleConfig | None = None,
    ) -> None:
        config = config or SkeletonConfig()
        self._config = config
        self._angle_config = angle_config or AngleConfig()
        self._transform = CoordinateTransform()
        self.joint_names: list[str] = joint_names(
            config.include_optional, config.include_all_joints
        )
        self._entries: list[LiveFrameJoints] = []
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """已映射的帧数（与传入的帧序列前缀一一对应）。"""
        return self._consumed

    def extend(self, frames: Sequence[PoseFrame]) -> None:
        """追加 ``frames[consumed:]`` 的映射结果；已处理过的帧不重算。

        单帧映射失败只丢该帧的关节（3D 卡片少画一帧），不能让它和帧序列错位。
        """
        pending = list(frames[self._consumed :])
        if not pending:
            return
        for frame in pending:
            try:
                self._entries.append(self._map_one(frame))
            except Exception as exc:  # noqa: BLE001 - 预览映射失败不能影响分析
                logger.warning("帧 %s 的骨骼映射失败：%s", frame.index, exc)
                self._entries.append(LiveFrameJoints(index=frame.index, coords=()))
            self._consumed += 1

    def snapshot(self) -> tuple[LiveFrameJoints, ...]:
        """当前全部映射结果（追加不修改旧元素，返回副本避免读侧看到半截状态）。"""
        return tuple(self._entries)

    # ---- 内部 ----
    def _map_one(self, frame: PoseFrame) -> LiveFrameJoints:
        include = {
            "include_optional": self._config.include_optional,
            "include_all": self._config.include_all_joints,
        }
        angles = compute_angles(frame, self._angle_config) if frame.valid else {}
        if len(frame.world_landmarks) < LANDMARK_COUNT:
            # 没有米制 world 点就没有 3D 关节；只保留角度，不伪造尺度。
            return LiveFrameJoints(index=frame.index, coords=(), angles=angles)
        converted = self._transform.convert(map_frame(frame, use_world=True, **include))
        return LiveFrameJoints(
            index=frame.index,
            coords=self._flatten(converted),
            angles=angles,
        )

    def _flatten(self, converted: ConvertedFrame) -> tuple[float, ...]:
        coords: list[float] = []
        for name in self.joint_names:
            point = converted.joints.get(name)
            if point is None:
                coords.extend((0.0, 0.0, 0.0, 0.0))
                continue
            coords.extend((point.x, point.y, point.z, float(point.visibility)))
        return tuple(coords)


__all__ = ["LiveFrameJoints", "LiveSkeletonMapper"]
