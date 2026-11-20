"""骨骼映射与 RealityKit 契约（任务 6）。

- ``map_frame``：33 点 → 18 个 ARKit 关节（缺 spine/neck/head 用插值推导）。
- ``CoordinateTransform``：世界坐标（米制）→ 米制 / y-up / 右手 / -z 向前。
- ``build_root_motion``：髋中心位移 + 绕 Y 旋转，支持原地播放。
- ``build_skeleton_bundle``：一步完成 6.1~6.3，交给任务 7 的导出层。
"""

from __future__ import annotations

from app.backend.skeleton.bundle import (
    SkeletonBundle,
    SkeletonConfig,
    build_skeleton_bundle,
    measure_person_height,
)
from app.backend.skeleton.live import LiveFrameJoints, LiveSkeletonMapper
from app.backend.skeleton.mapping import (
    ARKIT_JOINTS,
    DERIVED_JOINTS,
    DIRECT_MAP,
    LEFT_JOINTS,
    OPTIONAL_JOINTS,
    OPTIONAL_MAP,
    RIGHT_JOINTS,
    JointPoint,
    SkeletonFrame,
    joint_names,
    map_frame,
    map_frames,
    validate_mapping,
)
from app.backend.skeleton.root_motion import (
    YAW_CONVENTION,
    Quaternion,
    RootMotionConfig,
    RootMotionFrame,
    RootMotionTrack,
    build_root_motion,
    quaternion_from_yaw,
    split_translation_rotation,
    unwrap_angles,
    yaw_between,
)
from app.backend.skeleton.transform import (
    COORDINATE_SYSTEM,
    UNITS,
    ConvertedFrame,
    CoordinateTransform,
    JointPosition,
    Vector3,
    convert_frames,
    from_realitykit_vector,
    roundtrip_error,
    to_realitykit_vector,
)

__all__ = [
    "ARKIT_JOINTS",
    "COORDINATE_SYSTEM",
    "DERIVED_JOINTS",
    "DIRECT_MAP",
    "LEFT_JOINTS",
    "OPTIONAL_JOINTS",
    "OPTIONAL_MAP",
    "RIGHT_JOINTS",
    "UNITS",
    "YAW_CONVENTION",
    "ConvertedFrame",
    "CoordinateTransform",
    "JointPoint",
    "JointPosition",
    "LiveFrameJoints",
    "LiveSkeletonMapper",
    "Quaternion",
    "RootMotionConfig",
    "RootMotionFrame",
    "RootMotionTrack",
    "SkeletonBundle",
    "SkeletonConfig",
    "SkeletonFrame",
    "Vector3",
    "build_root_motion",
    "build_skeleton_bundle",
    "convert_frames",
    "from_realitykit_vector",
    "joint_names",
    "map_frame",
    "map_frames",
    "measure_person_height",
    "quaternion_from_yaw",
    "roundtrip_error",
    "split_translation_rotation",
    "to_realitykit_vector",
    "unwrap_angles",
    "validate_mapping",
    "yaw_between",
]
