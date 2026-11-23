"""骨架拓扑（骨长标定、yaw 估计、IK 拟合、基准脚本共用的唯一真源）。

为什么单独一层：
- ``skeleton/mapping.py`` 决定"33 点 → ARKit 18 关节"的**命名**，但它不含"谁是谁的父节点"；
- 逆运动学需要父子关系才能做 FK，基准脚本需要骨段列表才能算骨长变异系数；
- 如果各自维护一份，导出侧与拟合侧迟早会不一致 —— 所以拓扑只在这里定义一次。

节点分两类：
- **实节点**：直接对应 BlazePose 33 点之一（``index`` 非空），拟合结果可写回 ``world_landmarks``；
- **虚节点**：由两个点求中点得到（``pelvis`` = 两髋中点、``chest`` = 两肩中点），
  只用于连接父子链，不写回。

区间（``low``/``high``）是人体测量学常识范围（米），用于把误检/遮挡产生的离谱骨长钳住。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.backend.engine.base import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

PELVIS = "pelvis"
CHEST = "chest"
HEAD = "head"

# 参与拟合/标定的实节点（写回 world_landmarks 的索引）
CORE_INDICES: tuple[int, ...] = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

# 不需要依赖世界坐标就能测宽度的点对（用于 yaw 的宽度比）
SHOULDER_PAIR: tuple[int, int] = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIP_PAIR: tuple[int, int] = (LEFT_HIP, RIGHT_HIP)

# 鼻尖相对「颅骨中心（两耳中点）/肩中点铅垂线」的解剖前伸量（米）。
# 解剖站立姿势的铅垂线穿过耳垂、肩峰与股骨大转子：颅骨中心几乎正落在肩中点上方，
# 鼻尖在它前方 9~10 cm。这个常数有三处用途，必须是同一个值，否则三处会互相打架：
# - ``ik_fit`` 头骨段（肩中点→鼻尖）的深度目标与骨长上限；
# - ``mapping`` 里 head_joint 相对鼻尖的后移量（``NOSE_PROTRUSION_METERS``）；
# - 输出侧「颅骨中心前探上限」的标尺。
HEAD_FORWARD_METERS = 0.09


@dataclass(frozen=True)
class NodeSpec:
    """一个骨段：``parent`` 指向 ``child``，长度落在 ``[low, high]`` 米。"""

    key: str
    parent: str | None
    index: int | None
    joint: str | None
    low: float
    high: float

    @property
    def mid(self) -> float:
        return round((self.low + self.high) / 2.0, 4)


# 顺序即 FK 顺序：父节点必须排在子节点之前
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(PELVIS, None, None, None, 0.0, 0.0),
    # ---- 躯干 ----
    NodeSpec("left_hip", PELVIS, LEFT_HIP, "left_upLeg_joint", 0.08, 0.18),
    NodeSpec("right_hip", PELVIS, RIGHT_HIP, "right_upLeg_joint", 0.08, 0.18),
    NodeSpec(CHEST, PELVIS, None, "spine_7_joint", 0.35, 0.60),
    # 肩中点 → 鼻尖。解剖上 = hypot(竖直 0.13~0.18, 前伸 ``HEAD_FORWARD_METERS``)
    # ≈ 0.17~0.21 m。旧区间 (0.24, 0.34) 是按 MediaPipe 的 world z 量出来的，而那一截
    # 深度本身就是"脸朝镜头"的先验量（实测 0.20~0.27）：当成骨长固定下来之后，
    # "固定骨长 + 2D 重投影" 会把多出来的 5~8 cm 全部塞进 Δz，于是直立站立时
    # 头部稳定前探 15~20 cm —— 这就是"正对镜头站立却明显头前倾"的直接来源。
    # 区间收到解剖范围，越界时回落到 0.21（``ik_fit`` 另有更严的解剖上限）。
    NodeSpec(HEAD, CHEST, NOSE, "head_joint", 0.14, 0.28),
    # ---- 左臂 ----
    NodeSpec("left_shoulder", CHEST, LEFT_SHOULDER, "left_shoulder_1_joint", 0.10, 0.28),
    NodeSpec("left_elbow", "left_shoulder", LEFT_ELBOW, "left_arm_joint", 0.24, 0.36),
    NodeSpec("left_wrist", "left_elbow", LEFT_WRIST, "left_hand_joint", 0.22, 0.33),
    # ---- 右臂 ----
    NodeSpec("right_shoulder", CHEST, RIGHT_SHOULDER, "right_shoulder_1_joint", 0.10, 0.28),
    NodeSpec("right_elbow", "right_shoulder", RIGHT_ELBOW, "right_arm_joint", 0.24, 0.36),
    NodeSpec("right_wrist", "right_elbow", RIGHT_WRIST, "right_hand_joint", 0.22, 0.33),
    # ---- 左腿 ----
    NodeSpec("left_knee", "left_hip", LEFT_KNEE, "left_leg_joint", 0.35, 0.50),
    NodeSpec("left_ankle", "left_knee", LEFT_ANKLE, "left_foot_joint", 0.34, 0.46),
    NodeSpec("left_toes", "left_ankle", LEFT_FOOT_INDEX, "left_toes_joint", 0.12, 0.24),
    # ---- 右腿 ----
    NodeSpec("right_knee", "right_hip", RIGHT_KNEE, "right_leg_joint", 0.35, 0.50),
    NodeSpec("right_ankle", "right_knee", RIGHT_ANKLE, "right_foot_joint", 0.34, 0.46),
    NodeSpec("right_toes", "right_ankle", RIGHT_FOOT_INDEX, "right_toes_joint", 0.12, 0.24),
)

BONE_SEGMENTS: tuple[NodeSpec, ...] = tuple(node for node in NODES if node.parent is not None)

# 导出侧关节名 → 子节点 key（外部 USDZ 骨长按关节名覆盖）
JOINT_TO_NODE: dict[str, str] = {
    node.joint: node.key for node in BONE_SEGMENTS if node.joint is not None
}

# 关节分组（基准指标的统计口径）
JITTER_GROUPS: dict[str, tuple[int, ...]] = {
    "torso": (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),
    "limb": (LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE),
    "extremity": (LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
}


def clamp_to_range(value: float, node: NodeSpec) -> float:
    """把骨长钳到人体测量学区间；越界时回落到区间中值（噪声大时中值更安全）。"""
    if not (value > 0.0) or value < node.low or value > node.high:
        return node.mid
    return float(value)


def node_map() -> dict[str, NodeSpec]:
    return {node.key: node for node in NODES}


__all__ = [
    "BONE_SEGMENTS",
    "CHEST",
    "CORE_INDICES",
    "HEAD",
    "HEAD_FORWARD_METERS",
    "HIP_PAIR",
    "JITTER_GROUPS",
    "JOINT_TO_NODE",
    "NODES",
    "PELVIS",
    "SHOULDER_PAIR",
    "NodeSpec",
    "clamp_to_range",
    "node_map",
]
