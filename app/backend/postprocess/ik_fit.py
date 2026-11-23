"""固定骨长 + 2D 重投影拟合（任务 2）。

思路（InstantHMR 的 kinematic fit 轻量版，纯 CPU）：

1. **骨长标定**：逐帧量每条骨段长度，只保留"骨段正对镜头"的帧取时间中位数
   （骨段越接近沿视线方向，深度噪声越会把它量长），钳进人体测量学区间，
   并强制左右同名骨段对称；也可以直接用 USDZ 模型实测骨长覆盖。
2. **参数化**：每个骨段用"相对初值方向的二维角度偏移"``(a, b)`` 表示，
   根节点另有 3 维平移。骨长是**硬约束**（FK 时只乘固定骨长，不参与优化），
   因此输出的骨长严格恒定，参数维度也更低（每骨段 2 维，无冗余自由度）。
3. **残差**：
   - 主残差：FK 关节按弱透视投影到归一化图像坐标，与 2D ``landmarks`` 求差，按 visibility 加权；
   - 遮挡保持：可见度不足 / 单帧方向突变的骨段，保持上一次可信的**相对骨段角度**；
     目标方向随父关节旋转，因此被挡住的手臂/腿会跟着身体走，不会被低置信误检拉错位；
   - 次残差：相对上一帧的关节位置变化（抑制抖动）；
   - 根平移先验：压住深度方向（z 没有 2D 观测，只能靠先验固定）。
4. **求解**：``scipy.optimize.least_squares(method="trf", loss=..., max_nfev=...)``。
   不收敛 / 无 world landmarks / 数据退化 → 原样返回输入帧并记 ``converged=False``，
   **绝不抛异常中断整段分析**。

为什么"硬约束"比"软惩罚"好：软惩罚要在 3N 个自由坐标里搜索，容易收敛到
"骨长差一点、重投影好一点"的折中解；硬约束把搜索空间直接限制在"所有骨长都对的
姿态流形"上，维度更低、结果可解释（骨长误差恒为 0）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.errors import AnalysisCancelledError
from app.backend.core.logging import get_logger
from app.backend.engine.base import (
    LANDMARK_COUNT,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Landmark,
    PoseFrame,
)
from app.backend.skeleton.topology import (
    BONE_SEGMENTS,
    CHEST,
    CORE_INDICES,
    HEAD,
    HEAD_FORWARD_METERS,
    NODES,
    PELVIS,
    NodeSpec,
    clamp_to_range,
    node_map,
)

logger = get_logger("postprocess.ik_fit")

try:  # scipy 是唯一的重型依赖；缺失时自动降级为"只做骨长钳制 + 原样透传"
    from scipy.optimize import least_squares as _least_squares
except Exception:  # pragma: no cover - 只在依赖缺失的环境触发
    _least_squares = None  # type: ignore[assignment]
    logger.warning("未安装 scipy，骨长约束拟合自动关闭（pip install scipy）")

DEFAULT_ASPECT = 9 / 16.0

# 根平移只放开"图像平面内"的两个方向：单目视频里人物离镜头多远是**不可观测**的，
# 让优化器去调它只会得到一个每帧乱飘的深度（而且导出用的是相对 root 的关节坐标，
# 调它没有任何收益）。深度沿用引擎给的值，这也让"再拟合一次"成为不动点。
ROOT_PARAMS = 2

# 骨长合理性区间（米）与钳制规则见 skeleton/topology.py；这里只放"宽度"的兜底区间。
# 髋宽是"两髋关节中心"的距离（不是骨盆外形宽度），成年人约 0.16~0.34 m。
SHOULDER_WIDTH_RANGE = (0.30, 0.50)
HIP_WIDTH_RANGE = (0.16, 0.34)

# 参与"正对帧筛选"的最短骨段（米）：短于此的骨段与深度噪声同量级，筛了反而样本太少。
MIN_GATED_BONE_LENGTH = 0.20

# 左右同名骨段（人体左右长度差异通常 < 2%，观测到的偏差主要是单侧深度噪声）。
SYMMETRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_toes", "right_toes"),
)


# --------------------------------------------------------------------------- #
# 配置与报告
# --------------------------------------------------------------------------- #
class HeadDepthConfig(BaseModel):
    """``head``（胸中点 → 鼻尖）骨段的深度专项先验。

    为什么单独给头一组参数：正面机位下头的前后位置**完全没有观测**，``|Δz|`` 被骨长
    与 2D 投影长度定死、只剩符号。默认的 ``depth_prior_weight`` 会把方向拉回 MediaPipe
    的输入方向（而输入方向本身就带"脸朝镜头"的先验），``depth_flat`` 又被 ``ratio⁴``
    门控削弱到基准的五六成 —— 两者平衡的结果就是"直立站着头稳定前探几厘米"。

    这里做三件事，缺一不可：

    1. **压平加强、不再朝输入方向回归**（``flat_boost`` / ``prior_scale``）；
    2. **压平目标不是 0**（``target_forward_meters``）：解剖上鼻尖本来就在肩中点前方
       约 9 cm，"把 Δz 压到 0"等于让鼻尖退到肩中点正上方，头中心（两耳中点）反而
       落到胸后方 9 cm —— 前探和仰头都是这么来的；
    3. **骨长按解剖上限封顶**（见 ``calibrate_bone_lengths``）：头骨段的 3D 距离里
       有一大截是 MediaPipe 的深度先验，当成骨长固定下来就会把"多出来的长度"全部
       变成 Δz。

    代价（有意为之）：真正把脖子往前探的姿势也会被压平一部分。可接受，因为那种幅度
    主要靠"鼻尖相对胸中点往下掉"的 2D 位移表达，仍然保留。
    """

    enabled: bool = True
    prior_scale: float = Field(
        0.0,
        ge=0,
        le=1,
        description="头部方向朝初始方向（d0）回归的系数；0 = 不让 d0 把头部拽回前探位",
    )
    flat_boost: float = Field(
        2.0, ge=0, le=10, description="头部深度压平权重的倍数（1 = 与其它骨段一致）"
    )
    flat_gate_power: float = Field(
        0.0,
        ge=0,
        le=8,
        description="头部压平门控指数；0 = 不随投影比衰减（头部骨段在正面机位下恒被压平）",
    )
    flat_gate_floor: float = Field(
        0.6,
        ge=0,
        le=1,
        description="投影比下限：即使骨段几乎躺在画面平面内，也至少按该比例给压平权重",
    )
    target_forward_meters: float = Field(
        HEAD_FORWARD_METERS,
        ge=0,
        le=0.3,
        description="头部深度目标：鼻尖应位于胸中点（两肩中点）**前方**该距离（米）。"
        "解剖上颅骨中心几乎正落在肩中点铅垂线上，鼻尖在它前方 9~10 cm，"
        "所以 Δz(鼻尖−胸中点) 的目标是 −9 cm 而不是 0。它同时管住三件事："
        "方向（不许后仰/缩脖子）、幅度（骨长被封顶时不许再往前顶）、"
        "低头时不许把'鼻尖往下掉'翻译成'头往前探'；填 0 回到"
        "「把鼻尖压到肩中点正上方」的旧行为",
    )


class FitConfig(BaseModel):
    """骨长约束拟合的参数。默认开启；``enabled=False`` 时输出与改动前逐位一致。"""

    enabled: bool = Field(True, description="总开关，关闭后完全跳过拟合（A/B 对照用）")
    max_nfev: int = Field(60, ge=1, le=500, description="least_squares 的函数求值上限")
    reproj_weight: float = Field(1.0, gt=0, description="2D 重投影残差权重")
    bone_weight: float = Field(1.0, gt=0, description="骨长软残差权重（仅 hard_bones=False 时生效）")
    temporal_weight: float = Field(0.25, ge=0, le=5, description="相对上一帧位置的抑制权重")
    temporal_axes: Literal["z", "xyz"] = Field(
        "z",
        description="时域抑制作用在哪些轴上。默认只作用在 z：x/y 有 2D 观测（可信），"
        "对它们做时域平滑只会把结果往上一帧拽、牺牲 2D 一致性（实测 0 → 2 cm 漂移）；"
        "深度没有观测，才需要时域先验。改回 'xyz' 可复现改动前的行为",
    )
    root_prior_weight: float = Field(0.2, ge=0, le=5, description="根平移先验权重（压住深度漂移）")
    depth_prior_weight: float = Field(
        0.5,
        ge=0,
        le=5,
        description="骨段方向先验权重：2D 只约束两个方向，深度必须靠先验选分支（同时把欠定方程变成超定）。"
        "取 0.5 是为了让「顺着上一轮的平滑结果」成为默认解：0.05 时 z 会逐帧自由漂移，"
        "把前一步 z 强平滑的收益又浪费掉（实测 z 抖动 0.0042 → 0.0009，重投影不变）",
    )
    depth_flat_weight: float = Field(
        1.5,
        ge=0,
        le=20,
        description="深度压平先验权重（正面/侧面通用的默认解）。单目 2D 只能约束骨段在画面平面内的"
        "两个方向，沿视线方向的分量没有观测；该项按「骨段投影长度 / 真实骨长」的比例加权，"
        "把不可观测的深度分量压向 0。0 = 关闭（等价于旧行为）。"
        "权重单位是「1 米深度误差 ↔ 多少归一化图像高度」，1.0 表示把深度误差当作等量的图像误差来罚",
    )
    depth_flat_gate_power: float = Field(
        4.0,
        ge=0,
        le=8,
        description="压平先验的门控指数：投影比例≈1（骨段几乎躺在画面平面内，深度完全不可观测）"
        "时先验最强；比例变小时（骨段真的朝向/远离相机，2D 已把深度大小定死）先验迅速失效，"
        "4 次方让「正面站直的小腿」被压平、而「正面伸向镜头的胳膊」不受影响",
    )
    depth_sign_weight: float = Field(
        1.0,
        ge=0,
        le=5,
        description="深度符号先验权重（``postprocess.depth_sign`` 提供）。单目下每根骨段只剩"
        "「Δz 取正还是取负」一个比特：骨长与 2D 投影长度已经把 |Δz| 定死，符号才是"
        "「前踢腿被解成后踢腿」的根因。该项只在符号反了时施加惩罚（hinge），符号正确时"
        "残差与梯度都为 0，因此不会抢 2D 重投影或骨长的任何自由度。0 = 不用符号先验",
    )
    depth_sign_flat_suppression: float = Field(
        1.0,
        ge=0,
        le=1,
        description="有符号先验的骨段按该比例衰减「深度压平」权重：压平会把 Δz 往 0 拽，"
        "与符号先验直接冲突。1.0 = 完全交给符号先验；0 = 保留压平权重",
    )
    depth_direction_weight: float = Field(
        0.0,
        ge=0,
        le=5,
        description="§4 方向向量先验权重。与符号先验（hinge）不同：它要求整条骨段的**方向**"
        "对齐「方向线段」，因此能约束幅度；默认 0（关闭），打开前必须用基准扫权重",
    )
    depth_direction_min_confidence: float = Field(
        0.5,
        ge=0,
        le=1,
        description="方向先验只采信置信度不低于该值的骨段；摆动腿通常 0.8+，支撑腿被折扣后低于它",
    )
    head_depth: HeadDepthConfig = Field(
        default_factory=HeadDepthConfig,
        description="头部骨段（胸中点→鼻尖）的深度专项：单目正面机位下它只剩一个先验，"
        "必须压得比其它骨段更狠、且不朝输入方向回归，否则「直立站着头前探」会稳定残留",
    )
    depth_sparsity_weight: float = Field(
        0.0,
        ge=0,
        le=2,
        description="深度稀疏项：把 Δz 往 0 拉（倾向「贴着正面平面」的动作）。默认 0 = 关闭；"
        "它只作用于**没有方向先验**的骨段，避免与方向项互相拉扯",
    )
    yaw_weight: float = Field(0.6, ge=0, le=5, description="yaw 先验权重（0 = 不用 yaw）")
    max_yaw_deviation: float = Field(25.0, gt=0, le=90, description="允许偏离 2D yaw 的角度（度）")
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    min_visibility: float = Field(0.5, ge=0, lt=1, description="低于该可见度的点不参与残差")
    hold_occluded_limbs: bool = Field(
        True,
        description="肢体被遮挡时保持上一帧的相对骨段角度：低可见度或单帧方向突变的关节"
        "不再被噪声观测牵着走，而是随父关节的旋转继续运动，避免错位/扭曲",
    )
    occlusion_weight: float = Field(
        1.5,
        ge=0,
        le=10,
        description="遮挡保持残差权重（相对 2D 重投影）；0 等价于只按可见点拟合",
    )
    occlusion_max_frames: int = Field(
        30,
        ge=1,
        le=300,
        description="连续遮挡保持的最大帧数；超过后停止保持，不长期凭空维持旧姿态",
    )
    occlusion_jump_degrees: float = Field(
        75.0,
        gt=0,
        le=180,
        description="单帧骨段方向突变阈值（度）；超过视为误检/瞬时遮挡，按上一姿态保持",
    )
    max_reproj_error: float = Field(
        0.05,
        gt=0,
        le=0.3,
        description="拟合后 2D 一致性上限（归一化坐标，0.05 ≈ 画面高度 5%）；"
        "超过说明 3D 与 2D 本身不自洽，该帧原样透传",
    )
    hard_bones: bool = Field(True, description="骨长作为硬约束；关闭后改用 bone_weight 软惩罚")
    rigidify_failed_frames: bool = Field(
        True,
        description="拟合被跳过的**有效帧**也按「观测方向 + 标定骨长」重建，"
        "保证整段视频骨长恒定；关闭后这些帧原样透传（旧行为，相邻帧骨长会跳变）",
    )
    rigidify_max_joint_shift: float = Field(
        0.12,
        gt=0,
        le=1,
        description="重建前允许的最大单骨段位移（米）：输入骨长与标定值差得太多时说明这帧的"
        "世界坐标本身已崩，强行归一反而会把关节推得更远，此时仍然原样透传",
    )
    max_reproj_increase: float = Field(
        0.01,
        ge=0,
        le=0.1,
        description="允许拟合结果比输入差多少重投影误差（归一化高度）。"
        "深度压平先验是**故意**用一点平面内误差换掉不可观测的深度，"
        "所以兜底门限不能沿用「不许比输入差」这种零容差判据",
    )
    refine_passes: int = Field(
        2, ge=1, le=5, description="迭代细化趟数：每趟以上一趟输出为中心，直到变化小于阈值"
    )
    refine_tolerance: float = Field(
        0.01, gt=0, le=0.2, description="迭代细化的收敛阈值（米），默认 1 cm"
    )
    aspect_ratio: float | None = Field(
        None, description="画面宽高比；留空则按 9/16（竖屏手机视频）"
    )
    bone_lengths_override: dict[str, float] | None = Field(
        None, description="外部骨长覆盖（键为 ARKit 关节名或节点 key），优先于标定结果"
    )
    bone_sample_min_projected_ratio: float = Field(
        0.85,
        ge=0,
        le=1,
        description="骨长采样门控：只有「2D 投影长度 ÷ 理论投影长度」≥ 该值的帧参与中位数。"
        "骨段越接近沿视线方向，深度噪声越会把它量长（σ 叠加成 sqrt(L²+σ²)，永远偏大不会偏小），"
        "只取正对镜头的帧估计骨长偏差最小。0 = 关闭（所有帧等权，等价于旧行为）",
    )
    symmetrize_bones: bool = Field(
        True,
        description="左右同名骨段取同一长度：人体左右差异通常 <2%，观测偏差主要是单侧深度噪声，"
        "平均后能把噪声摊掉一半，也避免播放时出现「一条腿明显比另一条长」",
    )
    max_left_right_asymmetry: float = Field(
        0.08,
        gt=0,
        le=0.5,
        description="左右实测骨长相对差超过该比例时**不**强制对称（说明是系统性误检/遮挡，"
        "直接平均会把好的一侧也带偏），只在 notes 里报告",
    )


class FrameFitReport(BaseModel):
    index: int = 0
    fitted: bool = False
    iterations: int = 0
    cost: float = 0.0
    reproj_error_norm: float = 0.0
    bone_length_error_pct: float = 0.0
    converged: bool = True
    rigidified: bool = Field(
        False, description="未做 2D 拟合，但已按标定骨长重建（骨长仍恒定）"
    )
    held_joints: list[str] = Field(
        default_factory=list, description="本帧因遮挡/突变而沿用上一姿态的骨段节点名"
    )
    note: str = ""


class FitReport(BaseModel):
    """拟合汇总：``per_frame`` 供 UI / 质检定位到具体帧。"""

    applied: bool = False
    frames: int = 0
    frames_fitted: int = 0
    frames_skipped: int = 0
    frames_rigidified: int = Field(
        0, description="未参与 2D 拟合、但已按标定骨长重建的帧数（骨长仍恒定）"
    )
    frames_occlusion_held: int = Field(
        0, description="至少有一个肢体关节被遮挡保持的帧数"
    )
    iterations: int = 0
    cost: float = 0.0
    reproj_error_norm: float = 0.0
    bone_length_error_pct: float = 0.0
    converged: bool = True
    milliseconds_per_frame: float = 0.0
    bone_lengths: dict[str, float] = Field(default_factory=dict)
    per_frame: list[FrameFitReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BoneLengths(BaseModel):
    """标定后的参考骨长（米）。``lengths`` 按节点 key，``joint_lengths`` 按 ARKit 关节名。"""

    lengths: dict[str, float] = Field(default_factory=dict)
    shoulder_width: float = 0.40
    hip_width: float = 0.28
    torso_length: float = 0.45
    source: Literal["calibrated", "override", "mixed", "default"] = "default"
    notes: list[str] = Field(default_factory=list)

    def get(self, key: str, fallback: float | None = None) -> float:
        value = self.lengths.get(key)
        if value is not None and value > 0:
            return value
        return float(fallback) if fallback is not None else node_map()[key].mid

    def joint_lengths(self) -> dict[str, float]:
        """``dict[joint, meters]``，键与导出侧 ``jointNames`` 一致。"""
        names = {node.key: node.joint for node in BONE_SEGMENTS}
        return {
            names[key]: round(value, 5)
            for key, value in self.lengths.items()
            if key in names and names[key] is not None
        }


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def estimate_image_scale(frame: PoseFrame, aspect: float) -> tuple[float, float, float]:
    """估计弱透视投影参数 ``(k, u0, v0)``：``u = k*X + u0``、``v = k*Y + v0``。

    ``u`` 用画面高度归一化（``x_norm * aspect``），``v`` 即 ``y_norm``。
    用全部核心点做加权最小二乘，避免单个坏点决定尺度；
    MediaPipe 的 world landmarks 与归一化坐标本来就是弱透视关系，所以这个 ``k``
    等价于"相机焦距 / 人物距离"，不需要猜 FOV。
    """
    if len(frame.landmarks) < LANDMARK_COUNT or len(frame.world_landmarks) < LANDMARK_COUNT:
        return 0.0, 0.0, 0.0
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in CORE_INDICES:
        world = frame.world_landmarks[index]
        norm = frame.landmarks[index]
        weight = max(0.0, min(1.0, float(norm.visibility)))
        if weight <= 0.0:
            continue
        rows.append([world.x * weight, weight, 0.0])
        rows.append([world.y * weight, 0.0, weight])
        targets.append(norm.x * aspect * weight)
        targets.append(norm.y * weight)
    if len(rows) < 8:
        return 0.0, 0.0, 0.0
    design = np.asarray(rows, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)
    return float(solution[0]), float(solution[1]), float(solution[2])


def _node_positions(points: Sequence[Landmark]) -> dict[str, np.ndarray]:
    """按拓扑取出所有节点位置（虚节点用中点补齐）。"""
    if len(points) < LANDMARK_COUNT:
        return {}
    positions: dict[str, np.ndarray] = {}
    for node in NODES:
        if node.index is None:
            continue
        point = points[node.index]
        positions[node.key] = np.array([point.x, point.y, point.z], dtype=np.float64)
    if "left_hip" in positions and "right_hip" in positions:
        positions[PELVIS] = (positions["left_hip"] + positions["right_hip"]) / 2.0
    if "left_shoulder" in positions and "right_shoulder" in positions:
        positions[CHEST] = (positions["left_shoulder"] + positions["right_shoulder"]) / 2.0
    return positions


def node_positions(points: Sequence[Landmark]) -> dict[str, np.ndarray]:
    """公开别名：``depth_sign`` / 质检 / 基准脚本复用同一套拓扑解析，避免多份实现漂移。"""
    return _node_positions(points)


def _positions_array(positions: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([positions[node.key] for node in NODES], axis=0)


def _segment_lengths(positions: Mapping[str, np.ndarray]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for node in BONE_SEGMENTS:
        parent = positions.get(node.parent or "")
        child = positions.get(node.key)
        if parent is None or child is None:
            continue
        lengths[node.key] = float(np.linalg.norm(child - parent))
    return lengths


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _pair_visible(frame: PoseFrame, first: int, second: int, threshold: float) -> bool:
    if len(frame.landmarks) <= max(first, second):
        return False
    return min(frame.landmarks[first].visibility, frame.landmarks[second].visibility) >= threshold


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """归一化方向；零向量/非有限值回落到 ``fallback``（默认竖直方向）。"""
    default = np.array([0.0, 1.0, 0.0], dtype=np.float64) if fallback is None else fallback
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9 or not math.isfinite(norm):
        return np.asarray(default, dtype=np.float64).copy()
    return np.asarray(vector, dtype=np.float64) / norm


def _rotate_vector(source: np.ndarray, target: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """把 ``vector`` 按 source→target 的最小旋转变换（罗德里格斯公式，不构造矩阵）。"""
    a, b = _unit(source), _unit(target)
    axis = np.cross(a, b)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine <= 1e-6:
        return np.asarray(vector, dtype=np.float64).copy()
    axis /= sine
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
    )


def _occlusion_weight(age: int, config: FitConfig) -> float:
    """保持强度随连续遮挡帧数线性衰减，避免长时间遮挡把旧姿态永久钉死。"""
    limit = config.occlusion_max_frames
    if age >= limit:
        return 0.0
    fade = max(1, limit // 3)
    if age <= limit - fade:
        return 1.0
    return float(limit - age) / fade


def _occlusion_targets(
    frame: PoseFrame,
    current: np.ndarray,
    previous: np.ndarray | None,
    parent: np.ndarray,
    last_dirs: np.ndarray | None,
    last_parent_dirs: np.ndarray | None,
    hold_age: Sequence[int],
    config: FitConfig,
) -> tuple[dict[int, tuple[np.ndarray, float]], list[bool]]:
    """找出本帧需要保持的骨段，并算出它们在当前父关节坐标下的目标方向。

    保持的是**相对角度**而不是绝对坐标：被挡住的关节跟着父骨段一起旋转，
    因此躯干转身/抬手时不会凭空断裂；骨段长度仍由 FK 的硬约束保证不变。
    """
    count = len(NODES)
    directions: list[np.ndarray | None] = [None] * count
    current_dirs: list[np.ndarray | None] = [None] * count
    occluded = [False] * count
    cosine_limit = math.cos(math.radians(config.occlusion_jump_degrees))

    for i in range(1, count):
        node = NODES[i]
        p = int(parent[i])
        # 虚节点（chest/pelvis）没有可见度，但它们的方向是子关节的旋转参考，必须算
        current_dirs[i] = _unit(
            current[i] - current[p],
            directions[p] if directions[p] is not None else None,
        )
        if node.index is None or node.index >= len(frame.landmarks):
            continue
        low_visibility = frame.landmarks[node.index].visibility < config.min_visibility
        jump = False
        if not low_visibility and previous is not None and last_dirs is not None and p != 0:
            # 先用"相对上一帧的原始方向变化"做廉价门控：绝大多数帧方向变化很小，
            # 不必再算父关节旋转；只有变化确实大时才检查是不是随父关节一起转的。
            raw_change = float(np.dot(_unit(previous[i] - previous[p]), current_dirs[i]))
            if raw_change < cosine_limit:
                grandparent = int(parent[p])
                prev_dir = _unit(previous[p] - previous[grandparent])
                if float(np.linalg.norm(prev_dir)) <= 1e-9 and last_parent_dirs is not None:
                    prev_dir = last_parent_dirs[i]
                if float(np.linalg.norm(prev_dir)) > 1e-9:
                    curr_dir = _unit(current[p] - current[grandparent])
                    expected = _rotate_vector(prev_dir, curr_dir, last_dirs[i])
                    jump = float(np.dot(_unit(expected), current_dirs[i])) < cosine_limit
        occluded[i] = low_visibility or jump

    held: dict[int, tuple[np.ndarray, float]] = {}
    for i in range(1, count):
        p = int(parent[i])
        if current_dirs[i] is None:
            continue
        if not occluded[i] or previous is None or last_dirs is None or hold_age[i] >= config.occlusion_max_frames:
            directions[i] = current_dirs[i]
            continue

        reference = last_parent_dirs[i] if last_parent_dirs is not None else None
        if p == 0 or directions[p] is None or reference is None or float(np.linalg.norm(reference)) <= 1e-9:
            target = last_dirs[i]
        else:
            target = _rotate_vector(reference, directions[p], last_dirs[i])
        directions[i] = _unit(target, current_dirs[i])
        weight = _occlusion_weight(hold_age[i], config) * config.occlusion_weight
        if weight > 0:
            held[i] = (directions[i], math.sqrt(weight))
    return held, occluded


def _normalized_positions(frame: PoseFrame, aspect: float) -> dict[str, np.ndarray]:
    """按拓扑取出**归一化图像坐标**（``u = x*aspect``、``v = y``），虚节点用中点补齐。

    与 ``_node_positions`` 一一对应，只差"用的是 2D 观测还是米制世界点"。
    """
    if len(frame.landmarks) < LANDMARK_COUNT:
        return {}
    positions: dict[str, np.ndarray] = {}
    for node in NODES:
        if node.index is None:
            continue
        point = frame.landmarks[node.index]
        positions[node.key] = np.array([point.x * aspect, point.y], dtype=np.float64)
    if "left_hip" in positions and "right_hip" in positions:
        positions[PELVIS] = (positions["left_hip"] + positions["right_hip"]) / 2.0
    if "left_shoulder" in positions and "right_shoulder" in positions:
        positions[CHEST] = (positions["left_shoulder"] + positions["right_shoulder"]) / 2.0
    return positions


def _gate_eligible(node: NodeSpec) -> bool:
    """该骨段是否参与"只取正对帧"的筛选。

    短骨段（锁骨/骨盆半宽 0.19/0.13 m、脚掌 0.18 m）本身与深度噪声同量级：
    硬筛正对帧会把样本量砍到个位数，而这两段还是 yaw 的**参考宽度**来源，
    参考宽度在帧间保持一致比绝对值精确更重要 —— 所以对它们保留全样本中位数。
    """
    return node.mid >= MIN_GATED_BONE_LENGTH


def _projected_ratio(
    node: NodeSpec,
    lengths_2d: Mapping[str, np.ndarray],
    length_3d: float,
    scale: float,
) -> float | None:
    """骨段的「2D 投影长度 / 理论投影长度」。

    ``scale`` 是弱透视尺度 ``k``（归一化高度/米），所以 ``k*L`` 就是"骨段正对镜头时
    应有的投影长度"。比值≈1 → 骨段躺在画面平面内，深度分量几乎为 0，此时的长度测量
    最干净；比值≪1 → 骨段指向/远离镜头，深度误差会把长度量长。
    """
    parent = lengths_2d.get(node.parent or "")
    child = lengths_2d.get(node.key)
    expected = scale * length_3d
    if parent is None or child is None or expected <= 1e-9:
        return None
    projected = float(math.hypot(float(child[0] - parent[0]), float(child[1] - parent[1])))
    return min(1.0, max(0.0, projected / expected))


def _symmetrize(
    lengths: dict[str, float], max_asymmetry: float
) -> tuple[dict[str, float], list[str], list[str]]:
    """把左右同名骨段拉成同一长度。

    ``max_asymmetry`` 是守卫：真实人体左右差异通常 <2%，观测到 8% 以上的差异基本是
    单侧遮挡/误检，此时直接平均会把好的一侧也带偏，所以只报告不对称。
    """
    updated = dict(lengths)
    symmetrized: list[str] = []
    skipped: list[str] = []
    for left, right in SYMMETRIC_PAIRS:
        a = updated.get(left)
        b = updated.get(right)
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        mean = (a + b) / 2.0
        if abs(a - b) / mean > max_asymmetry:
            skipped.append(f"{left}/{right}（{abs(a - b) / mean:.0%}）")
            continue
        updated[left] = round(mean, 5)
        updated[right] = round(mean, 5)
        symmetrized.append(left)
    return updated, symmetrized, skipped


# --------------------------------------------------------------------------- #
# 骨长标定（任务 2.2）
# --------------------------------------------------------------------------- #
def calibrate_bone_lengths(
    frames: Sequence[PoseFrame],
    config: FitConfig | None = None,
    *,
    overrides: Mapping[str, float] | None = None,
) -> BoneLengths:
    """逐帧量骨段长度 → 抗噪中位数 → 钳进人体区间 → 左右对称 → 应用外部覆盖。

    中位数不是简单地对所有帧取：只有「骨段正对镜头」的帧才进样本
    （见 ``bone_sample_min_projected_ratio``），因为沿视线方向的深度噪声只会把骨段量长。
    某条骨段的正对帧太少时退回全样本，避免极端机位下无样本可用。

    例外是 ``head``（肩中点→鼻尖）：它 3D 距离里那一大截深度本身就是 MediaPipe 的
    "脸朝镜头"先验，不是可观测的骨长。这里用「2D 可见的竖直跨度 + 解剖前伸量
    ``HEAD_FORWARD_METERS``」给它封顶，只往下压、不往上抬，见下方 ``head_vertical``。
    """
    config = config or FitConfig()
    aspect = config.aspect_ratio or DEFAULT_ASPECT
    series: dict[str, list[float]] = {node.key: [] for node in BONE_SEGMENTS}
    frontal: dict[str, list[float]] = {node.key: [] for node in BONE_SEGMENTS}
    head_vertical: list[float] = []
    shoulders: list[float] = []
    hips: list[float] = []

    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        positions = _node_positions(frame.world_landmarks)
        if not positions:
            continue
        scale, _, _ = estimate_image_scale(frame, aspect)
        flat = _normalized_positions(frame, aspect) if scale > 1e-9 else {}
        for node in BONE_SEGMENTS:
            parent = positions.get(node.parent or "")
            child = positions.get(node.key)
            if parent is None or child is None:
                continue
            value = float(np.linalg.norm(child - parent))
            if value <= 1e-6:
                continue
            series[node.key].append(value)
            if node.key == HEAD:
                # 2D 能看见的那一部分（竖直跨度，米）：头前倾/低头都不会改变它太多，
                # 但深度先验会；配合解剖前伸量就能把"被吹大的骨长"顶掉。
                head_vertical.append(abs(float(child[1] - parent[1])))
            if _gate_eligible(node):
                ratio = _projected_ratio(node, flat, value, scale) if flat else None
                if ratio is None or ratio >= config.bone_sample_min_projected_ratio:
                    frontal[node.key].append(value)
        if "left_shoulder" in positions and "right_shoulder" in positions:
            shoulders.append(float(np.linalg.norm(positions["left_shoulder"] - positions["right_shoulder"])))
        if "left_hip" in positions and "right_hip" in positions:
            hips.append(float(np.linalg.norm(positions["left_hip"] - positions["right_hip"])))

    lengths: dict[str, float] = {}
    clamped: list[str] = []
    gated: list[str] = []
    head_capped = 0.0
    for node in BONE_SEGMENTS:
        values = series.get(node.key) or []
        chosen = frontal.get(node.key) or []
        # 至少 3 帧、且占全样本 1/4 以上才相信"正对帧"子集，否则退回全样本
        if len(chosen) >= max(3, len(values) // 4) and len(chosen) < len(values):
            values = chosen
            gated.append(node.key)
        median = float(np.median(values)) if values else node.mid
        length = clamp_to_range(median, node)
        if abs(length - median) > 1e-6:
            clamped.append(node.key)
        if node.key == HEAD and head_vertical:
            # 肩中点→鼻尖的**解剖**长度：竖直跨度（2D 可见）+ 前伸 ``HEAD_FORWARD_METERS``。
            # 实测 3D 距离 0.23~0.29 m，比解剖值长 5~10 cm —— 那一段全是"脸朝镜头"的
            # 深度先验；当成骨长固定下来后，|Δz| = sqrt(L² − p²) 只能把多出来的长度
            # 变成头前探。取 min 只在"被吹大"时生效，侧视（深度真的可观测）原样保留。
            # 放在 clamp 之后：这是本视频的实际几何，不该再被静态区间抬回去。
            anatomical = math.hypot(float(np.median(head_vertical)), HEAD_FORWARD_METERS)
            if anatomical < length:
                head_capped = length - anatomical
                length = anatomical
        lengths[node.key] = round(length, 5)

    notes: list[str] = []
    if head_capped > 1e-6:
        notes.append(
            f"头骨段（肩中点→鼻尖）实测长度比解剖上限长 {head_capped * 100:.1f} cm，"
            f"已按「竖直跨度 + 前伸 {HEAD_FORWARD_METERS:.2f} m」封顶："
            "多出来的长度只可能来自深度先验，固定成骨长就会变成头前探"
        )
    if gated and config.bone_sample_min_projected_ratio > 0:
        notes.append(
            f"{len(gated)} 条骨段只用了正对镜头的帧估长度"
            f"（投影比 ≥ {config.bone_sample_min_projected_ratio:.2f}），以避开深度噪声把骨段量长"
        )

    symmetrized: list[str] = []
    skipped: list[str] = []
    if config.symmetrize_bones:
        lengths, symmetrized, skipped = _symmetrize(lengths, config.max_left_right_asymmetry)
        if symmetrized:
            notes.append(f"左右同名骨段已强制对称：{len(symmetrized)} 对")
        if skipped:
            notes.append(
                "左右骨长差异超过 "
                f"{config.max_left_right_asymmetry:.0%}，未强制对称（疑似单侧遮挡/误检）："
                + "、".join(skipped[:4])
            )

    # 宽度由"锁骨/骨盆半宽"两段骨长相加得到 —— 这两段是刚体骨长，
    # 绕竖直轴转身时不变；直接用观测到的肩宽会随 yaw 一起缩小（yaw 估计就会失效）。
    shoulder_width = lengths.get("left_shoulder", 0.0) + lengths.get("right_shoulder", 0.0)
    if not (SHOULDER_WIDTH_RANGE[0] <= shoulder_width <= SHOULDER_WIDTH_RANGE[1]):
        shoulder_width = _percentile(shoulders, 90) or (
            (SHOULDER_WIDTH_RANGE[0] + SHOULDER_WIDTH_RANGE[1]) / 2.0
        )
        if not (SHOULDER_WIDTH_RANGE[0] <= shoulder_width <= SHOULDER_WIDTH_RANGE[1]):
            shoulder_width = (SHOULDER_WIDTH_RANGE[0] + SHOULDER_WIDTH_RANGE[1]) / 2.0
    hip_width = lengths.get("left_hip", 0.0) + lengths.get("right_hip", 0.0)
    if not (HIP_WIDTH_RANGE[0] <= hip_width <= HIP_WIDTH_RANGE[1]):
        hip_width = _percentile(hips, 90) or ((HIP_WIDTH_RANGE[0] + HIP_WIDTH_RANGE[1]) / 2.0)
        if not (HIP_WIDTH_RANGE[0] <= hip_width <= HIP_WIDTH_RANGE[1]):
            hip_width = (HIP_WIDTH_RANGE[0] + HIP_WIDTH_RANGE[1]) / 2.0

    if clamped:
        notes.insert(
            0,
            f"{len(clamped)} 条骨段实测长度越界，已回落到人体测量学区间中值："
            + "、".join(sorted(clamped)[:6]),
        )
    if not shoulders:
        notes.append("没有可用的肩宽观测，肩宽按 0.40 m 兜底（yaw 估计会降权）")

    result = BoneLengths(
        lengths=lengths,
        shoulder_width=round(shoulder_width, 5),
        hip_width=round(hip_width, 5),
        torso_length=lengths.get(CHEST, 0.45),
        source="calibrated",
        notes=notes,
    )

    applied = overrides if overrides is not None else config.bone_lengths_override
    if applied:
        result = _apply_override(result, applied)
    return result


def _apply_override(lengths: BoneLengths, overrides: Mapping[str, float]) -> BoneLengths:
    """外部骨长（USDZ 实测）优先级最高：不做区间钳制，只做合法性检查。"""
    joints = {node.key: node.joint for node in BONE_SEGMENTS}
    updated = dict(lengths.lengths)
    used: list[str] = []
    for key, value in overrides.items():
        node_key = key if key in updated else next(
            (nk for nk, joint in joints.items() if joint == key), None
        )
        if node_key is None or node_key not in updated:
            continue
        try:
            length = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(length) or length <= 0:
            continue
        updated[node_key] = round(length, 5)
        used.append(node_key)

    if not used:
        return lengths.model_copy(
            update={"notes": [*lengths.notes, "外部骨长覆盖未命中任何已知骨段，已忽略"]}
        )
    shoulder_width = lengths.shoulder_width
    if "left_shoulder" in updated and "right_shoulder" in updated:
        # 肩宽 = 两侧锁骨的水平跨度近似；这样 yaw 的参考宽度与骨长保持一致
        shoulder_width = round(updated["left_shoulder"] + updated["right_shoulder"], 5)
    hip_width = lengths.hip_width
    if "left_hip" in updated and "right_hip" in updated:
        hip_width = round(updated["left_hip"] + updated["right_hip"], 5)
    return lengths.model_copy(
        update={
            "lengths": updated,
            "source": "override" if lengths.source == "override" else "mixed",
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "torso_length": updated.get(CHEST, lengths.torso_length),
            "notes": [*lengths.notes, f"已应用 {len(used)} 条外部骨长（优先级最高）"],
        }
    )


def load_bone_lengths_override(payload: Mapping[str, object]) -> dict[str, float]:
    """解析外部骨长 JSON（``{"left_arm_joint": 0.28, ...}``，兼容外层包一层 ``boneLengths``）。"""
    source: Mapping[str, object] = payload
    nested = payload.get("boneLengths")
    if isinstance(nested, Mapping):
        source = nested
    result: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        result[str(key)] = float(value)
    return result


# --------------------------------------------------------------------------- #
# yaw 先验（任务 3.2）
# --------------------------------------------------------------------------- #
def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_positions(positions: np.ndarray, left: int, right: int) -> float:
    """从（右→左）横向量算 yaw：``θ = atan2(-dz, dx)``。

    约定与 ``skeleton/orientation.py`` 一致：``yaw = 0`` 正对镜头，
    正角度表示左肩转向镜头。
    """
    vector = positions[left] - positions[right]
    return math.atan2(-float(vector[2]), float(vector[0]))


# --------------------------------------------------------------------------- #
# 求解器
# --------------------------------------------------------------------------- #
class _FrameContext:
    """一帧的常量（不随参数变化的部分），避免残差函数里重复计算。"""

    __slots__ = (
        "aspect",
        "base",
        "d0",
        "flat_scale",
        "flat_target",
        "fk_terms",
        "held_dir",
        "held_idx",
        "held_weight",
        "initial",
        "prior_gain",
        "prior_scale",
        "e1",
        "e2",
        "length",
        "obs_u",
        "obs_v",
        "parent",
        "prev",
        "reproj_idx",
        "root_scale",
        "scale",
        "sign_idx",
        "sign_value",
        "sign_weight",
        "temporal_rows",
        "temporal_scale",
        "u0",
        "v0",
        "weights",
        "yaw",
        "yaw_confidence",
        "yaw_index",
        "yaw_limit",
        "yaw_scale",
        "sparsity",
        "dir_idx",
        "dir_weight",
        "dir_e1",
        "dir_e2",
        "sparse_scale",
    )


def _flat_scale(
    nodes: Sequence[NodeSpec],
    positions: Mapping[str, np.ndarray],
    length: np.ndarray,
    scale: float,
    config: FitConfig,
) -> np.ndarray:
    """「深度压平先验」的逐骨段权重（单位：归一化图像高度 / 米）。

    为什么需要它：弱透视下 2D 只观测到骨段在画面平面内的两个方向，沿视线的分量
    完全不可观测。给定骨长 ``L`` 与 2D 投影长度 ``p``，深度分量的大小其实被
    ``|dz| = sqrt(L² - p²)`` 定死，**只有符号是自由的**；而根号在 ``p ≈ L`` 附近
    极其敏感：骨长只差 5%，就会凭空长出 ``0.3 L``（小腿 0.42 m → 13 cm）的深度，
    侧视看就是"站直的腿被解成屈膝"。

    所以这里按投影比例 ``r = p / (k·L)`` 给权重：``r ≈ 1`` 表示骨段几乎躺在画面
    平面内、深度纯属编造 → 权重最大；``r ≪ 1`` 表示骨段真的朝向/远离相机、
    2D 已经把深度大小定死 → 权重迅速衰减（``r^4``），交给 2D 决定。

    权重乘上 ``scale``（1/米）之后，残差 ``dz · w`` 与重投影残差同量纲，
    于是"1 米深度误差"可以按"等价多少图像误差"来权衡，不需要凭空调参。
    """
    flat = np.zeros(len(nodes), dtype=np.float64)
    if config.depth_flat_weight <= 0 or scale <= 1e-9:
        return flat
    baseline_power = max(0.0, float(config.depth_flat_gate_power))
    head = config.head_depth
    base = math.sqrt(float(config.depth_flat_weight)) * scale
    for i, node in enumerate(nodes):
        if node.parent is None:
            continue
        child = positions.get(node.key)
        parent = positions.get(node.parent)
        if child is None or parent is None:
            continue
        expected = scale * float(length[i])
        if expected <= 1e-9:
            continue
        projected = float(math.hypot(child[0] - parent[0], child[1] - parent[1]))
        ratio = min(1.0, max(0.0, projected / expected))
        power = baseline_power
        boost = 1.0
        if head.enabled and node.key == HEAD:
            # 头部：不衰减（power=0）+ 整体加强，让"压平"赢过朝 d0 的回归项
            power = float(head.flat_gate_power)
            boost = float(head.flat_boost)
            ratio = max(float(head.flat_gate_floor), ratio)
        flat[i] = base * boost * (ratio**power)
    return flat


def _entry_parts(entry: object) -> tuple[int, float, np.ndarray | None]:
    """统一解包符号条目：``(符号, 置信度)`` 或 ``(符号, 置信度, 方向向量)``。

    两元组是改造前的格式（也是单测里手写的格式），三元组是本轮新增的方向线段。
    两种都接受，这样"只用符号"的调用方不需要跟着改。
    """
    values = list(entry)  # type: ignore[arg-type]
    sign = int(values[0]) if values else 0
    confidence = float(values[1]) if len(values) > 1 else 0.0
    direction: np.ndarray | None = None
    if len(values) > 2 and values[2] is not None:
        vector = np.asarray(values[2], dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            direction = vector / norm
    return sign, confidence, direction


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给单位方向找一组正交基（用于把"方向不对齐"拆成两个残差分量）。"""
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(direction, helper)
    norm = float(np.linalg.norm(first))
    first = first / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    second = np.cross(direction, first)
    second_norm = float(np.linalg.norm(second))
    second = second / second_norm if second_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return first, second


def _direction_context(
    signs: Mapping[str, tuple[int, float]] | None,
    config: FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把带方向的符号条目转成按节点下标对齐的方向先验数组。

    返回 ``(idx, weight, e1, e2)``：残差为 ``sqrt(w)·(v·e1)`` 与 ``sqrt(w)·(v·e2)``，
    两者平方和正好是 ``w·|v⊥|²`` —— 即"垂直于期望方向的偏差"的单侧二次惩罚。
    """
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    if not signs or config.depth_direction_weight <= 0:
        return empty

    indices: list[int] = []
    weights: list[float] = []
    firsts: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    base = math.sqrt(config.depth_direction_weight)
    for index, node in enumerate(NODES):
        entry = signs.get(node.key)
        if entry is None or node.parent is None:
            continue
        sign, confidence, direction = _entry_parts(entry)
        if direction is None or confidence < config.depth_direction_min_confidence:
            continue
        # 方向线段与符号先验必须同向：不同向说明两者矛盾，宁可不施加
        if sign != 0 and float(np.sign(direction[2])) != float(sign) and abs(direction[2]) > 1e-6:
            continue
        first, second = _orthonormal_basis(direction)
        indices.append(index)
        weights.append(base * min(1.0, confidence))
        firsts.append(first)
        seconds.append(second)
    if not indices:
        return empty
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(firsts, dtype=np.float64),
        np.asarray(seconds, dtype=np.float64),
    )


def _sign_context(
    signs: Mapping[str, tuple[int, float]] | None,
    config: FitConfig,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 ``{骨段 key: (符号, 置信度)}`` 转成按节点下标对齐的四个数组。

    权重取 ``sqrt(depth_sign_weight) * k * 置信度``：与 ``_flat_scale`` 同量纲
    （1 米深度误差 ↔ 同等归一化图像误差），于是两个先验之间的强弱可以直接比较。
    """
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
    )
    if not signs or config.depth_sign_weight <= 0 or scale <= 1e-9:
        return empty
    indices: list[int] = []
    values: list[float] = []
    weights: list[float] = []
    confidences: list[float] = []
    base = math.sqrt(config.depth_sign_weight) * scale
    for index, node in enumerate(NODES):
        entry = signs.get(node.key)
        if entry is None or node.parent is None:
            continue
        sign, confidence, direction = _entry_parts(entry)
        # 方向项与 hinge 互斥（§4）：同一帧同一骨段只走一套先验，否则两个先验会互相拉扯
        if direction is not None and config.depth_direction_weight > 0:
            if confidence >= config.depth_direction_min_confidence:
                continue
        if sign == 0 or confidence <= 0:
            continue
        weight = min(1.0, float(confidence))
        indices.append(index)
        values.append(1.0 if sign > 0 else -1.0)
        weights.append(base * weight)
        confidences.append(weight)
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(confidences, dtype=np.float64),
    )


def flip_signed_depth(
    positions: Mapping[str, np.ndarray],
    signs: Mapping[str, tuple[int, float]] | None,
    config: FitConfig,
) -> dict[str, np.ndarray]:
    """按符号先验把**观测位置**换到正确的深度分支上（等距镜像，2D 与骨长不变）。

    必须在 FK/遮挡判断之前做：遮挡检测比较的是"上一帧结果 vs 本帧观测"，
    如果本帧观测还停在镜像分支上，二者会被误判成"单帧突变"而触发保持逻辑，
    反而把姿态钉在错分支上。所以深度分支的修正在**读入这一帧时**一次做完，
    后续所有环节（方向初值、压平门控、yaw 参考、遮挡状态）看到的是同一个姿态。
    """
    if not signs or config.depth_sign_weight <= 0:
        return dict(positions)
    flipped = dict(positions)
    for node in NODES:  # NODES 保证父节点先于子节点，逐段翻转即为严格的等距镜像
        if node.parent is None:
            continue
        entry = signs.get(node.key)
        if entry is None:
            continue
        sign, confidence, _direction = _entry_parts(entry)
        if sign == 0 or confidence <= 0:
            continue
        parent = flipped.get(node.parent)
        child = flipped.get(node.key)
        if parent is None or child is None:
            continue
        dz = float(child[2] - parent[2])
        if dz == 0.0 or (dz > 0.0) == (sign > 0):
            continue
        mirrored = np.array([child[0], child[1], 2.0 * float(parent[2]) - float(child[2])])
        flipped[node.key] = mirrored
    return flipped


def _flip_depth_branch(
    d0: np.ndarray, sign_idx: np.ndarray, sign_value: np.ndarray
) -> np.ndarray:
    """按符号先验把初值换到正确的深度分支上。

    弱透视下 ``ΔZ`` 的正负是两个**投影完全相同、骨长完全相同**的镜像解，
    所以这里只翻 ``d0`` 的 z 分量再归一化：2D 残差与骨长一个字都不变，
    优化器却从一个正确分支出发，不必跨过退化流形爬过去。
    """
    if not len(sign_idx):
        return d0
    flipped = d0.copy()
    for position, index in enumerate(sign_idx):
        value = float(flipped[index, 2])
        target = float(sign_value[position])
        if value == 0.0 or (value > 0.0) == (target > 0.0):
            continue
        flipped[index, 2] = -value
        norm = float(np.linalg.norm(flipped[index]))
        if norm > 1e-9:
            flipped[index] /= norm
    return flipped


def _make_context(
    frame: PoseFrame,
    positions: dict[str, np.ndarray],
    lengths: BoneLengths,
    config: FitConfig,
    aspect: float,
    previous: np.ndarray | None,
    yaw_prior: tuple[float, float] | None,
    held: Mapping[int, tuple[np.ndarray, float]] | None = None,
    signs: Mapping[str, tuple[int, float]] | None = None,
) -> _FrameContext:
    keys = [node.key for node in NODES]
    index_of = {key: i for i, key in enumerate(keys)}
    parent = np.array([index_of.get(node.parent or "", 0) for node in NODES], dtype=np.int64)
    parent[0] = 0

    base = positions[PELVIS].copy()
    d0 = np.zeros((len(NODES), 3), dtype=np.float64)
    for i, node in enumerate(NODES):
        if node.parent is None:
            d0[i] = np.array([0.0, 1.0, 0.0])
            continue
        vector = positions[node.key] - positions[node.parent]
        norm = float(np.linalg.norm(vector))
        d0[i] = vector / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])

    length = np.array([lengths.get(node.key, node.mid) for node in NODES], dtype=np.float64)

    scale, u0, v0 = estimate_image_scale(frame, aspect)
    flat_scale = _flat_scale(NODES, positions, length, scale, config)
    sign_idx, sign_value, sign_weight, sign_confidence = _sign_context(signs, config, scale)
    dir_idx, dir_weight, dir_e1, dir_e2 = _direction_context(signs, config)
    # 压平先验会把不可观测的 Δz 往 0 拽，与"这块骨段的深度该往哪边"直接冲突：
    # 有符号先验的骨段按置信度让位，没有符号先验的骨段仍按原来的门控压平。
    if sign_idx.size and config.depth_sign_flat_suppression > 0:
        for position, index in enumerate(sign_idx):
            flat_scale[index] *= max(
                0.0, 1.0 - config.depth_sign_flat_suppression * float(sign_confidence[position])
            )
    # 有方向先验的骨段深度幅度由方向项说了算，压平先验必须整体让位
    if dir_idx.size:
        flat_scale[dir_idx] = 0.0
    # 压平目标：默认 0（"贴着正面平面"），但头部骨段不行 —— 鼻尖解剖上就在胸中点前方
    # 9~10 cm，压到 0 会把颅骨中心（两耳中点）顶到胸后方同样距离。
    # 非零目标只给头骨段（见下方残差）。
    flat_target = np.zeros(len(NODES), dtype=np.float64)
    if config.head_depth.enabled and flat_scale[index_of[HEAD]] > 0:
        # 世界坐标 -z = 靠近镜头，所以"前方 9 cm"的目标值是负的
        flat_target[index_of[HEAD]] = -float(config.head_depth.target_forward_meters)
    reproj_rows: list[int] = []
    obs_u: list[float] = []
    obs_v: list[float] = []
    weights: list[float] = []
    for i, node in enumerate(NODES):
        if node.index is None or node.key == PELVIS:
            continue
        norm = frame.landmarks[node.index]
        visibility = float(norm.visibility)
        if visibility < config.min_visibility:
            continue
        reproj_rows.append(i)
        obs_u.append(norm.x * aspect)
        obs_v.append(norm.y)
        weights.append(min(1.0, visibility))

    yaw_index = (index_of["left_shoulder"], index_of["right_shoulder"])
    yaw_half_width = lengths.shoulder_width / 2.0
    yaw = 0.0
    yaw_confidence = 0.0
    if yaw_prior is not None:
        yaw, yaw_confidence = yaw_prior
        if not _pair_visible(frame, LEFT_SHOULDER, RIGHT_SHOULDER, config.min_visibility):
            # 肩线被衣物/手臂挡住时改用髋线（yaw 的宽度比在髋线上同样有效）
            yaw_index = (index_of["left_hip"], index_of["right_hip"])
            yaw_half_width = lengths.hip_width / 2.0
            if not _pair_visible(frame, LEFT_HIP, RIGHT_HIP, config.min_visibility):
                yaw_confidence = 0.0

    # 任务 3.2：用 2D 宽度比估出的 yaw **替换**初值里由 world z 猜出来的朝向。
    # 做法是把所有初始方向整体绕竖直轴（y）转 Δ = yaw_2d - yaw_init，
    # 于是优化器从一个"肩线朝向已经正确"的点出发，只需要细调，不再需要先
    # 沿一个大概率错误的深度方向爬出来（这也是再拟合漂移的主要来源之一）。
    if yaw_prior is not None and yaw_confidence > 0:
        left = positions[NODES[yaw_index[0]].key]
        right = positions[NODES[yaw_index[1]].key]
        observed = math.atan2(-float(left[2] - right[2]), float(left[0] - right[0]))
        delta = _wrap_angle(yaw - observed)
        if abs(delta) > 1e-6:
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            rotated = d0.copy()
            rotated[:, 0] = d0[:, 0] * cos_d + d0[:, 2] * sin_d
            rotated[:, 2] = -d0[:, 0] * sin_d + d0[:, 2] * cos_d
            d0 = rotated

    # 深度符号先验：先把初值换到正确的深度分支上（只翻 z，2D 与骨长完全不变）。
    # 这一步比"让优化器自己爬过去"可靠得多：Δz 的符号在退化流形上是对称的，
    # 从错分支出发的局部最优同样满足 2D 残差，优化器没有理由离开它。
    d0 = _flip_depth_branch(d0, sign_idx, sign_value)

    # 遮挡保持：直接把初始方向换成"上一姿态随父关节旋转后"的方向，而不是让
    # 优化器从一个低置信误检的方向往回爬；参数 0 处即满足保持残差。
    # 放在符号翻转之后：被挡住的骨段没有可信 2D，理应沿用上一姿态而不是信任先验。
    if held:
        for index, (direction, _) in held.items():
            d0[index] = _unit(direction)

    e1 = np.zeros_like(d0)
    e2 = np.zeros_like(d0)
    for i in range(len(NODES)):
        direction = d0[i]
        reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        first = np.cross(direction, reference)
        first /= max(1e-9, float(np.linalg.norm(first)))
        e1[i] = first
        e2[i] = np.cross(direction, first)

    context = _FrameContext()
    context.aspect = aspect
    context.base = base
    context.initial = np.stack([positions[node.key] for node in NODES], axis=0)
    # 参数空间的偏移是无量纲的，乘上"方向偏移 1 个单位在图像上的位移"（约 k·骨长）
    # 之后才与重投影残差同量纲，权重才有可比性。
    context.prior_scale = (
        math.sqrt(config.depth_prior_weight) * max(1e-6, scale) * float(np.mean(length[1:]))
    )
    # 逐骨段的方向回归强度：默认全部等于 prior_scale，头部按 head_depth.prior_scale 打折。
    # 头部那一项是"朝 MediaPipe 输入方向回归"，而输入方向本身就带脸朝镜头的前伸先验，
    # 不把它关掉，压平先验再强也会被拉回一半（实测就是"直立时头稳定前探几厘米"）。
    prior_gain = np.full(
        2 * (len(NODES) - 1), context.prior_scale, dtype=np.float64
    )
    head_position = index_of.get(HEAD)
    if config.head_depth.enabled and head_position:
        offset = 2 * (head_position - 1)
        prior_gain[offset : offset + 2] *= float(config.head_depth.prior_scale)
    context.prior_gain = prior_gain
    context.d0 = d0
    context.e1 = e1
    context.e2 = e2
    held_indices = sorted(held or {})
    context.held_idx = np.asarray(held_indices, dtype=np.int64)
    context.held_dir = (
        np.stack([_unit(held[index][0]) for index in held_indices])
        if held_indices
        else np.zeros((0, 3), dtype=np.float64)
    )
    context.held_weight = (
        np.asarray([held[index][1] for index in held_indices], dtype=np.float64)
        if held_indices
        else np.zeros(0, dtype=np.float64)
    )
    context.length = length
    # FK 用的纯标量表（见 _forward_kinematics）：每个节点 (父节点, 骨长, d0, e1, e2)
    context.fk_terms = [
        (
            int(parent[i]),
            float(length[i]),
            *d0[i].tolist(),
            *e1[i].tolist(),
            *e2[i].tolist(),
        )
        for i in range(len(NODES))
    ]
    context.parent = parent
    context.flat_scale = flat_scale
    context.flat_target = flat_target
    context.sign_idx = sign_idx
    context.sign_value = sign_value
    context.sign_weight = sign_weight
    # 时域 z 残差原本是"深度没有观测，只能靠上一帧拉住"的替代品。一旦某骨段有了
    # 方向先验，它就不该再被上一帧拉走（否则动作起始帧的深度会被压回 0，
    # 表现为"踢腿慢半拍"）。
    # 注意必须排除**整条子树**：残差是加在关节绝对位置上的，只排除骨段自身时，
    # 被它带动的子节点（脚尖、手）仍会把父骨段一起拽回去，排除就失效了。
    temporal_rows = np.arange(len(NODES), dtype=np.int64)
    if config.temporal_axes == "z" and (sign_idx.size or dir_idx.size):
        excluded = [False] * len(NODES)
        signed = {int(index) for index in sign_idx} | {int(index) for index in dir_idx}
        for index in range(len(NODES)):
            ancestor = int(parent[index])
            excluded[index] = index in signed or (ancestor > 0 and excluded[ancestor])
        temporal_rows = np.flatnonzero([not flag for flag in excluded]).astype(np.int64)
    context.temporal_rows = temporal_rows
    context.dir_idx = dir_idx
    context.dir_weight = dir_weight
    context.dir_e1 = dir_e1
    context.dir_e2 = dir_e2
    # 深度稀疏项只作用于"没有方向先验"的骨段：两个项都在动深度，一起上等于自相矛盾
    sparse_scale = np.zeros(len(NODES), dtype=np.float64)
    if config.depth_sparsity_weight > 0:
        # 同样换成图像单位（每米深度 ↔ 多少归一化图像误差），才能和重投影直接比
        sparse_scale[1:] = math.sqrt(config.depth_sparsity_weight) * max(1e-6, scale)
        sparse_scale[dir_idx] = 0.0
    context.sparse_scale = sparse_scale
    context.scale = scale
    context.u0 = u0
    context.v0 = v0
    context.reproj_idx = np.asarray(reproj_rows, dtype=np.int64)
    context.obs_u = np.asarray(obs_u, dtype=np.float64)
    context.obs_v = np.asarray(obs_v, dtype=np.float64)
    context.weights = np.asarray(weights, dtype=np.float64)
    context.prev = previous
    context.temporal_scale = math.sqrt(config.temporal_weight)
    context.root_scale = math.sqrt(config.root_prior_weight)
    context.yaw = yaw
    context.yaw_confidence = yaw_confidence
    context.yaw_index = yaw_index
    context.yaw_limit = math.radians(config.max_yaw_deviation)
    # 单位统一：yaw 偏差 δ 在图像上的表现是"肩宽少掉 k·半肩宽·sinδ"，
    # 所以乘上 k·半肩宽，残差就与归一化坐标同量纲了（否则 1 rad 会盖过一切）。
    context.yaw_scale = math.sqrt(config.yaw_weight) * max(1e-6, scale * yaw_half_width)
    context.sparsity = _build_sparsity(context, config)
    return context


def _ancestor_chains(parent: np.ndarray) -> list[list[int]]:
    """每个节点的祖先链（含自身）。位置只受自己和祖先的参数影响，这是用好稀疏雅可比的关键。"""
    chains: list[list[int]] = []
    for i in range(len(NODES)):
        chain = [i]
        cursor = int(parent[i])
        while cursor != 0 and cursor not in chain:
            chain.append(cursor)
            cursor = int(parent[cursor])
        chains.append(chain)
    return chains


def _build_sparsity(context: _FrameContext, config: FitConfig) -> np.ndarray:
    """残差-参数的依赖矩阵。

    ``least_squares`` 拿到它之后可以用列分组一次性扰动多个参数列，
    把有限差分雅可比的函数求值次数从"参数个数"降到"独立列组数"——
    本骨架的列组约 8 组（35 个参数），单帧耗时因此降到约 1/4。
    """
    node_count = len(NODES)
    param_count = ROOT_PARAMS + 2 * (node_count - 1)
    chains = _ancestor_chains(context.parent)

    def columns_for(node: int) -> list[int]:
        cols = list(range(ROOT_PARAMS))
        for ancestor in chains[node]:
            if ancestor == 0:
                continue
            offset = ROOT_PARAMS + 2 * (ancestor - 1)
            cols.extend((offset, offset + 1))
        return cols

    rows: list[tuple[int, list[int]]] = []
    cursor = 0
    reproj_count = int(context.reproj_idx.size) if context.scale > 1e-9 else 0
    if reproj_count:
        for offset, node in enumerate(context.reproj_idx):
            cols = columns_for(int(node))
            rows.append((cursor + offset, cols))  # u
            rows.append((cursor + reproj_count + offset, cols))  # v
        cursor += 2 * reproj_count

    # 遮挡保持：每骨段 3 行，依赖自己和祖先的方向参数
    for offset, node in enumerate(context.held_idx):
        cols = columns_for(int(node))
        for axis in range(3):
            rows.append((cursor + offset * 3 + axis, cols))
    cursor += int(context.held_idx.size) * 3

    # 深度压平残差：每骨段一行，只依赖自己与祖先的方向参数（与重投影残差同列组）
    if float(np.max(context.flat_scale)) > 0:
        for node in range(1, node_count):
            rows.append((cursor + node - 1, columns_for(node)))
        cursor += node_count - 1

    # 深度符号（hinge）残差：每个带符号先验的骨段一行，依赖同一条方向参数链
    if context.sign_idx.size:
        for offset, node in enumerate(context.sign_idx):
            rows.append((cursor + offset, columns_for(int(node))))
        cursor += int(context.sign_idx.size)

    # 方向先验：每个骨段两行（两个垂直分量），依赖同一条方向参数链
    if context.dir_idx.size:
        for offset, node in enumerate(context.dir_idx):
            columns = columns_for(int(node))
            rows.append((cursor + 2 * offset, columns))
            rows.append((cursor + 2 * offset + 1, columns))
        cursor += 2 * int(context.dir_idx.size)

    # 深度稀疏项：每骨段一行（含被方向先验覆盖的骨段，它们权重为 0，但行留着更简单）
    if float(np.max(getattr(context, "sparse_scale", np.zeros(1)))) > 0:
        for node in range(1, node_count):
            rows.append((cursor + node - 1, columns_for(node)))
        cursor += node_count - 1

    if context.prior_scale > 0:
        for node in range(1, node_count):
            offset = ROOT_PARAMS + 2 * (node - 1)
            rows.append((cursor + 2 * (node - 1), [offset]))
            rows.append((cursor + 2 * (node - 1) + 1, [offset + 1]))
        cursor += 2 * (node_count - 1)

    if context.prev is not None and context.temporal_scale > 0:
        if config.temporal_axes == "z":
            for offset, node in enumerate(context.temporal_rows):
                rows.append((cursor + offset, columns_for(int(node))))
            cursor += int(context.temporal_rows.size)
        else:
            for node in range(node_count):
                cols = columns_for(node)
                for axis in range(3):
                    rows.append((cursor + node * 3 + axis, cols))
            cursor += node_count * 3

    if context.root_scale > 0:
        for axis in range(ROOT_PARAMS):
            rows.append((cursor + axis, list(range(ROOT_PARAMS))))
        cursor += ROOT_PARAMS

    if context.yaw_scale > 0 and context.yaw_confidence > 0:
        cols = sorted(
            set(columns_for(context.yaw_index[0])) | set(columns_for(context.yaw_index[1]))
        )
        rows.append((cursor, cols))
        cursor += 1

    if not config.hard_bones and config.bone_weight > 0:
        for node in range(1, node_count):
            rows.append((cursor + node - 1, columns_for(node)))
        cursor += node_count - 1

    total = max(1, cursor)
    matrix = np.zeros((total, param_count), dtype=np.int8)
    for row, cols in rows:
        matrix[row, cols] = 1
    return matrix


def _forward_kinematics(params: np.ndarray, context: _FrameContext) -> np.ndarray:
    """FK：沿每条骨段的固定长度累加方向。

    实现上用**纯 Python 标量**而不是逐节点的 numpy 运算：这个函数在有限差分
    雅可比里每帧要跑几百次，而节点只有 17 个 —— 标量版比 numpy 版快约 10 倍
    （每帧拟合的耗时几乎全在这里），数值结果逐位一致。
    """
    terms = context.fk_terms
    base = context.base
    values = params.tolist() if hasattr(params, "tolist") else [float(p) for p in params]
    count = len(terms)
    xs = [0.0] * count
    ys = [0.0] * count
    zs = [0.0] * count
    xs[0] = float(base[0]) + values[0]
    ys[0] = float(base[1]) + values[1]
    zs[0] = float(base[2])
    for i in range(1, count):
        parent, length, d0x, d0y, d0z, e1x, e1y, e1z, e2x, e2y, e2z = terms[i]
        offset = ROOT_PARAMS + 2 * (i - 1)
        a = values[offset]
        b = values[offset + 1]
        dx = d0x + a * e1x + b * e2x
        dy = d0y + a * e1y + b * e2y
        dz = d0z + a * e1z + b * e2z
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm < 1e-9:
            dx, dy, dz, norm = d0x, d0y, d0z, 1.0
        scale = length / norm
        xs[i] = xs[parent] + dx * scale
        ys[i] = ys[parent] + dy * scale
        zs[i] = zs[parent] + dz * scale
    return np.array([xs, ys, zs], dtype=np.float64).T


def _residuals(params: np.ndarray, context: _FrameContext, config: FitConfig) -> np.ndarray:
    positions = _forward_kinematics(params, context)
    parts: list[np.ndarray] = []

    rows = context.reproj_idx
    if rows.size and context.scale > 1e-9:
        predicted_u = context.scale * positions[rows, 0] + context.u0
        predicted_v = context.scale * positions[rows, 1] + context.v0
        weight = context.weights * config.reproj_weight
        parts.append((predicted_u - context.obs_u) * weight)
        parts.append((predicted_v - context.obs_v) * weight)

    # 遮挡保持：被挡住的骨段没有可靠 2D观测，用上一姿态的相对角度约束方向。
    # 这里只罚方向（不含长度）：长度由 FK 硬约束固定，父关节移动时子关节自然跟随。
    if context.held_idx.size:
        vectors = positions[context.held_idx] - positions[context.parent[context.held_idx]]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        directions = vectors / np.maximum(norms, 1e-9)
        error = (directions - context.held_dir) * context.held_weight[:, None]
        parts.append(error.ravel())

    # 深度压平：把每条骨段沿视线的分量压向**目标值**（默认 0）。权重按"该骨段有多贴合
    # 画面平面"门控，于是正面站直的小腿不会被解成屈膝，而正面伸向镜头的胳膊依然保留真实
    # 深度。头部骨段的目标是 −9 cm（鼻尖在胸中点前方），见 ``HeadDepthConfig``。
    if float(np.max(context.flat_scale)) > 0:
        local_z = positions[1:, 2] - positions[context.parent[1:], 2]
        parts.append((local_z - context.flat_target[1:]) * context.flat_scale[1:])

    # 深度符号先验（hinge）：只在"Δz 与先验反号"时施加惩罚。
    # 符号正确时残差为 0、梯度为 0 → 不抢 2D 重投影与骨长硬约束的任何自由度；
    # 幅度完全放行（|Δz| 本来就被骨长与 2D 投影长度定死），所以模型在跨域时
    # 幅度不准也无所谓 —— 这正是本方案能抗住 OOD 的原因。
    if context.sign_idx.size:
        rows = context.sign_idx
        local_z = positions[rows, 2] - positions[context.parent[rows], 2]
        wrong = np.maximum(0.0, -context.sign_value * local_z)
        parts.append(wrong * context.sign_weight)

    # 方向向量先验（§4）：要求骨段方向对齐"方向线段"，因此也能约束幅度。
    # 残差取"垂直于期望方向的两个分量"，平方和 = w·|v⊥|²；对齐时残差与梯度都为 0。
    if context.dir_idx.size:
        vectors = positions[context.dir_idx] - positions[context.parent[context.dir_idx]]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        units = vectors / np.maximum(norms, 1e-9)
        # 换成"图像单位"：偏离角 δ 在画面上的位移 ≈ δ·L·k。不乘这一步，
        # 方向项的量纲就与重投影不同，权重扫描会得到"稍微加大就压过 2D"的假象。
        gain = context.dir_weight * context.scale * context.length[context.dir_idx]
        along_first = np.einsum("ij,ij->i", units, context.dir_e1) * gain
        along_second = np.einsum("ij,ij->i", units, context.dir_e2) * gain
        parts.append(np.concatenate([along_first, along_second]))

    # 深度稀疏项（默认关闭）：把没有方向先验的骨段 Δz 往 0 拉，
    # 让"贴着正面平面的动作"不会被凭空解出深度。
    if float(np.max(context.sparse_scale)) > 0:
        local_z = positions[1:, 2] - positions[context.parent[1:], 2]
        parts.append(local_z * context.sparse_scale[1:])

    # 方向先验：2D 只约束"上下左右"，深度方向没有观测，必须靠先验选分支。
    # 注意它是加在**参数**（相对输入方向的偏移）上，而不是加在位置上：
    # 这样在不动点处（偏移回到 0）这项的梯度自动为 0，不会像"位置先验"那样
    # 每次都把解往新一轮的输入处拉，从而保证"再拟合一次"几乎不动。
    if context.prior_scale > 0:
        # 逐骨段的 gain：头部通常被 head_depth.prior_scale 置 0，不再朝 d0 回归，
        # 于是深度压平（或符号/方向先验）成为唯一决定头部深度的项。
        parts.append(params[ROOT_PARAMS:] * context.prior_gain)

    if context.prev is not None and context.temporal_scale > 0:
        delta = positions - context.prev
        if config.temporal_axes == "z":
            # 只压深度方向的帧间跳变：x/y 由 2D 说了算，不该被上一帧拖住。
            # 已有符号先验的骨段被排除（见 _make_context 的 temporal_rows）。
            if context.temporal_rows.size:
                parts.append(delta[context.temporal_rows, 2] * context.temporal_scale)
        else:
            parts.append(delta.ravel() * context.temporal_scale)

    if context.root_scale > 0:
        parts.append(params[:ROOT_PARAMS] * context.root_scale)

    if context.yaw_scale > 0 and context.yaw_confidence > 0:
        left, right = context.yaw_index
        fitted = _yaw_from_positions(positions, left, right)
        delta = _wrap_angle(fitted - context.yaw)
        penalty = np.array([delta], dtype=np.float64)
        # 超出允许偏差后线性加罚（软约束，不是硬卡死）
        overflow = abs(delta) - context.yaw_limit
        if overflow > 0:
            penalty = np.array([math.copysign(context.yaw_limit, delta) + 4.0 * overflow])
        parts.append(penalty * context.yaw_scale * context.yaw_confidence)

    if not config.hard_bones and config.bone_weight > 0:
        error = []
        for i, node in enumerate(NODES):
            if node.parent is None:
                continue
            current = float(np.linalg.norm(positions[i] - positions[context.parent[i]]))
            error.append((current - context.length[i]) * config.bone_weight)
        parts.append(np.asarray(error, dtype=np.float64))

    if not parts:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(parts)


def _residual_metrics(
    positions: np.ndarray, context: _FrameContext
) -> tuple[float, float]:
    """返回 ``(重投影 RMS（归一化坐标）, 骨长相对误差百分比)``。"""
    rows = context.reproj_idx
    reproj = 0.0
    if rows.size and context.scale > 1e-9:
        du = context.scale * positions[rows, 0] + context.u0 - context.obs_u
        dv = context.scale * positions[rows, 1] + context.v0 - context.obs_v
        reproj = float(np.sqrt(np.mean(du**2 + dv**2)))

    errors = []
    for i, node in enumerate(NODES):
        if node.parent is None:
            continue
        target = context.length[i]
        if target <= 1e-9:
            continue
        current = float(np.linalg.norm(positions[i] - positions[context.parent[i]]))
        errors.append(abs(current - target) / target)
    bone = float(np.mean(errors) * 100.0) if errors else 0.0
    return reproj, bone


class FrameFitter:
    """带状态的逐帧拟合器：自己记住上一帧结果，用于时域抑制。"""

    def __init__(self, lengths: BoneLengths, config: FitConfig | None = None) -> None:
        self.config = config or FitConfig()
        self.lengths = lengths
        self._previous: np.ndarray | None = None
        keys = [node.key for node in NODES]
        index_of = {key: i for i, key in enumerate(keys)}
        self._parent = np.array(
            [index_of.get(node.parent or "", 0) for node in NODES], dtype=np.int64
        )
        self._parent[0] = 0
        self._last_dirs: np.ndarray | None = None
        self._last_parent_dirs: np.ndarray | None = None
        self._hold_age = [0] * len(NODES)

    def _update_occlusion_state(self, array: np.ndarray, occluded: Sequence[bool]) -> None:
        """更新"上次可信方向"与连续遮挡帧数；被保持的关节不覆盖参考姿态。"""
        if self._last_dirs is None:
            self._last_dirs = np.zeros_like(array)
            self._last_parent_dirs = np.zeros_like(array)
        assert self._last_dirs is not None and self._last_parent_dirs is not None
        for i in range(1, len(NODES)):
            p = int(self._parent[i])
            has_reference = float(np.linalg.norm(self._last_dirs[i])) > 1e-9
            if occluded[i] and has_reference:
                self._hold_age[i] += 1
                continue
            self._hold_age[i] = 0
            self._last_dirs[i] = _unit(array[i] - array[p])
            if p != 0:
                self._last_parent_dirs[i] = _unit(array[p] - array[int(self._parent[p])])
            else:
                self._last_parent_dirs[i] = 0.0

    def _fallback(
        self,
        frame: PoseFrame,
        note: str,
        *,
        held: Mapping[int, tuple[np.ndarray, float]] | None = None,
        occluded: Sequence[bool] | None = None,
        signs: Mapping[str, tuple[int, float]] | None = None,
    ) -> tuple[PoseFrame, FrameFitReport]:
        """不做 2D 拟合时的兜底：尽量把骨长归一，而不是原样透传。

        原样透传会让这一帧的骨长继续沿用引擎的逐帧猜测，和左右相邻帧的标定骨长
        不一致，播放时表现为"某个肢体的长度突然弹一下"。
        """
        report = FrameFitReport(index=frame.index, note=note)
        if held:
            report.held_joints = [NODES[index].key for index in sorted(held)]
        flags = occluded or [False] * len(NODES)

        def commit(output: PoseFrame) -> PoseFrame:
            available = _node_positions(output.world_landmarks)
            if available:
                self._update_occlusion_state(_positions_array(available), flags)
            return output

        if not self.config.rigidify_failed_frames:
            return commit(frame.model_copy(deep=True)), report
        rigid = _rigidify(
            frame,
            self.lengths,
            self.config,
            self.config.aspect_ratio or DEFAULT_ASPECT,
            held=held,
            signs=signs,
        )
        if rigid is None:
            return commit(frame.model_copy(deep=True)), report
        report.rigidified = True
        report.note = f"{note}；已按标定骨长重建（骨长保持恒定）"
        return commit(rigid), report

    def fit(
        self,
        frame: PoseFrame,
        *,
        yaw: tuple[float, float] | None = None,
        signs: Mapping[str, tuple[int, float]] | None = None,
    ) -> tuple[PoseFrame, FrameFitReport]:
        config = self.config
        report = FrameFitReport(index=frame.index)
        aspect = config.aspect_ratio or DEFAULT_ASPECT

        if (
            not config.enabled
            or not frame.valid
            or len(frame.world_landmarks) < LANDMARK_COUNT
            or len(frame.landmarks) < LANDMARK_COUNT
        ):
            report.note = "跳过（未启用 / 帧无效 / 缺 world landmarks）"
            return frame.model_copy(deep=True), report

        if frame.mean_visibility(CORE_INDICES) < config.min_visibility:
            return self._fallback(
                frame, "跳过（核心关节平均可见度过低）", signs=signs
            )

        positions = _node_positions(frame.world_landmarks)
        if not positions:
            report.note = "跳过（世界坐标不完整）"
            return frame.model_copy(deep=True), report

        # 先把观测换到正确深度分支上，再进遮挡判断与 FK：否则"上一帧结果（已修正）vs
        # 本帧观测（未修正）"会被误判成单帧突变，触发保持逻辑反而钉住错分支。
        positions = flip_signed_depth(positions, signs, config)
        current_array = _positions_array(positions)
        held: dict[int, tuple[np.ndarray, float]] = {}
        occluded = [False] * len(NODES)
        if config.hold_occluded_limbs:
            held, occluded = _occlusion_targets(
                frame,
                current_array,
                self._previous,
                self._parent,
                self._last_dirs,
                self._last_parent_dirs,
                self._hold_age,
                config,
            )
        report.held_joints = [NODES[index].key for index in sorted(held)]

        context = _make_context(
            frame,
            positions,
            self.lengths,
            config,
            aspect,
            self._previous,
            yaw,
            held=held,
            signs=signs,
        )
        if context.scale <= 1e-9:
            return self._fallback(
                frame,
                "跳过（弱透视尺度估计失败）",
                held=held,
                occluded=occluded,
                signs=signs,
            )

        initial = np.zeros(ROOT_PARAMS + 2 * (len(NODES) - 1), dtype=np.float64)
        if _least_squares is None:  # pragma: no cover - 依赖缺失时降级
            # 没有 scipy 也能把骨长钳到标定值（沿观测方向重新量一次），
            # 只是不做"以 2D 一致性为目标"的优化。
            fitted = _forward_kinematics(initial, context)
            reproj, bone = _residual_metrics(fitted, context)
            report.fitted = True
            report.converged = True
            report.reproj_error_norm = round(reproj, 6)
            report.bone_length_error_pct = round(bone, 4)
            report.note = "未安装 scipy，仅做骨长钳制（不做最小二乘拟合）"
            self._previous = fitted
            self._update_occlusion_state(fitted, occluded)
            return _write_back(frame, fitted), report
        try:
            solution = _least_squares(
                _residuals,
                initial,
                args=(context, config),
                method="trf",
                loss=config.loss,
                max_nfev=config.max_nfev,
                jac_sparsity=context.sparsity,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
        except Exception as exc:  # pragma: no cover - 数值异常兜底
            logger.debug("第 %s 帧拟合失败：%s", frame.index, exc)
            return self._fallback(
                frame, f"求解失败：{exc}", held=held, occluded=occluded, signs=signs
            )

        fitted = _forward_kinematics(solution.x, context)
        reproj, bone = _residual_metrics(fitted, context)
        initial_reproj, _ = _residual_metrics(
            _forward_kinematics(initial, context), context
        )
        report.iterations = int(solution.nfev)
        report.cost = float(solution.cost)
        report.reproj_error_norm = round(reproj, 6)
        report.bone_length_error_pct = round(bone, 4)
        report.converged = bool(solution.success)
        if not solution.success:
            report.note = f"未完全收敛：{solution.message}"

        # 兜底：拟合后仍与 2D 差得远，说明这帧的 3D 与 2D 本身不自洽
        # （坏检测 / 极端遮挡 / 上游给的世界坐标尺度异常），不采用拟合结果，
        # 退回"观测方向 + 标定骨长"的重建（骨长仍然恒定）。
        # 注意门限留了 ``max_reproj_increase`` 的余量：压平先验会主动放弃一点
        # 平面内精度去换掉不可观测的深度，不能用"不许比输入差"的零容差判据。
        if reproj > config.max_reproj_error or reproj > initial_reproj + config.max_reproj_increase:
            return self._fallback(
                frame,
                f"2D 一致性不达标（{reproj:.4f} > {config.max_reproj_error:.4f}，"
                f"初始 {initial_reproj:.4f}）",
                held=held,
                occluded=occluded,
                signs=signs,
            )

        report.fitted = True
        self._previous = fitted
        self._update_occlusion_state(fitted, occluded)
        return _write_back(frame, fitted), report


def _max_drift(before: Sequence[PoseFrame], after: Sequence[PoseFrame]) -> float:
    """两轮结果之间的最大关节位移（米），用于判断迭代是否已到不动点。"""
    worst = 0.0
    for old, new in zip(before, after, strict=False):
        if len(old.world_landmarks) < LANDMARK_COUNT or len(new.world_landmarks) < LANDMARK_COUNT:
            continue
        for a, b in zip(old.world_landmarks, new.world_landmarks, strict=False):
            distance = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
            worst = max(worst, distance)
    return worst


def _write_back(frame: PoseFrame, positions: np.ndarray) -> PoseFrame:
    """把拟合后的节点位置写回 ``world_landmarks``（只动参与拟合的点）。"""
    copy = frame.model_copy(deep=True)
    for i, node in enumerate(NODES):
        if node.index is None:
            continue
        point = copy.world_landmarks[node.index]
        point.x = float(positions[i, 0])
        point.y = float(positions[i, 1])
        point.z = float(positions[i, 2])
    return copy


def _rigidify(
    frame: PoseFrame,
    lengths: BoneLengths,
    config: FitConfig,
    aspect: float,
    *,
    held: Mapping[int, tuple[np.ndarray, float]] | None = None,
    signs: Mapping[str, tuple[int, float]] | None = None,
) -> PoseFrame | None:
    """不做最小二乘，只把"观测方向 + 标定骨长"重新走一遍 FK。

    用在 IK 被跳过的**有效帧**上：方向仍来自本帧观测（有 2D 支撑），长度换成整段
    共用的标定值，于是整段视频的骨长严格恒定。输入骨长偏差过大（世界坐标已经崩了）
    时返回 ``None``，让调用方继续原样透传，避免把关节推离观测。
    """
    if len(frame.landmarks) < LANDMARK_COUNT or len(frame.world_landmarks) < LANDMARK_COUNT:
        return None
    positions = _node_positions(frame.world_landmarks)
    if not positions:
        return None
    # 保留符号先验：兜底路径的职责是"不做最小二乘"，不是"把已经选对的分支退回去"。
    positions = flip_signed_depth(positions, signs, config)
    context = _make_context(
        frame, positions, lengths, config, aspect, None, None, held=held, signs=signs
    )
    worst = 0.0
    for i, node in enumerate(NODES):
        if node.parent is None or (held and i in held):
            continue  # 被保持的骨段本来就不用本帧观测方向，不参与"输入是否已崩"判定
        target = float(context.length[i])
        child = positions.get(node.key)
        parent = positions.get(node.parent)
        if target <= 1e-9 or child is None or parent is None:
            continue
        current = float(np.linalg.norm(child - parent))
        worst = max(worst, abs(current - target))
    if worst > config.rigidify_max_joint_shift:
        return None
    params = np.zeros(ROOT_PARAMS + 2 * (len(NODES) - 1), dtype=np.float64)
    return _write_back(frame, _forward_kinematics(params, context))


def fit_frame(
    frame: PoseFrame,
    lengths: BoneLengths,
    config: FitConfig | None = None,
    *,
    yaw: tuple[float, float] | None = None,
    signs: Mapping[str, tuple[int, float]] | None = None,
) -> tuple[PoseFrame, FrameFitReport]:
    """单帧拟合（无时域项），便于单测与离线调试。"""
    config = config or FitConfig()
    if not config.enabled:
        return frame.model_copy(deep=True), FrameFitReport(index=frame.index, note="未启用")
    fitter = FrameFitter(lengths, config.model_copy(update={"temporal_weight": 0.0}))
    return fitter.fit(frame, yaw=yaw, signs=signs)


def fit_frames(
    frames: Sequence[PoseFrame],
    config: FitConfig | None = None,
    *,
    lengths: BoneLengths | None = None,
    yaws: Mapping[int, tuple[float, float]] | None = None,
    signs: Mapping[int, Mapping[str, tuple[int, float]]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[PoseFrame], FitReport]:
    """整段拟合。无效帧、缺 world landmarks 的帧原样透传。

    ``signs`` 是 ``{帧号: {骨段 key: (符号, 置信度)}}`` 的深度符号先验
    （由 ``postprocess.depth_sign`` 生成）。为空时行为与没有这一层完全一致。

    ``progress(done, total)`` 按趟累计帧数上报；``should_cancel`` 在每帧检查，
    让长视频的 IK 拟合可以被用户取消（否则 2 万帧会卡住整个任务）。
    """
    config = config or FitConfig()
    ordered = sorted(frames, key=lambda item: item.index)
    if not config.enabled:
        return [frame.model_copy(deep=True) for frame in ordered], FitReport(
            applied=False,
            frames=len(ordered),
            notes=["FitConfig.enabled=False，拟合已关闭（输出与改动前一致）"],
        )

    started = time.perf_counter()
    lengths = lengths or calibrate_bone_lengths(ordered, config)

    # 时域/深度先验都是"以上一步结果为中心"的近端项，因此单趟拟合只走到不动点附近。
    # 这里做少量迭代细化：第 2 趟起以第 1 趟的输出为中心，几毫秒就能收敛，
    # 收益是"对已拟合结果再拟合"成为不动点（幂等，改动 < 1 cm）。
    current = [frame.model_copy(deep=True) for frame in ordered]
    per_frame: list[FrameFitReport] = []
    passes = max(1, config.refine_passes)
    total_steps = max(1, passes * len(current))
    done_steps = 0
    report_every = max(1, total_steps // 100)
    for p in range(passes):
        if should_cancel is not None and should_cancel():
            raise AnalysisCancelledError("后处理已取消")
        fitter = FrameFitter(lengths, config)
        outputs: list[PoseFrame] = []
        reports: list[FrameFitReport] = []
        for frame in current:
            if should_cancel is not None and should_cancel():
                raise AnalysisCancelledError("后处理已取消")
            yaw = None if yaws is None else yaws.get(frame.index)
            frame_signs = None if signs is None else signs.get(frame.index)
            fitted, report = fitter.fit(frame, yaw=yaw, signs=frame_signs)
            outputs.append(fitted)
            reports.append(report)
            done_steps += 1
            if progress is not None and done_steps % report_every == 0:
                progress(done_steps, total_steps)
        drift = _max_drift(current, outputs)
        current = outputs
        per_frame = reports
        if drift <= config.refine_tolerance:
            done_steps = (p + 1) * len(current)
            if progress is not None:
                progress(done_steps, total_steps)
            break
    if progress is not None:
        progress(total_steps, total_steps)
    elapsed = time.perf_counter() - started

    result = current
    fitted_reports = [item for item in per_frame if item.fitted]
    notes: list[str] = list(lengths.notes)
    if yaws:
        notes.append(f"已使用 {len(yaws)} 帧的 2D yaw 作为拟合先验")
    if signs:
        used = sum(1 for frame in ordered if signs.get(frame.index))
        if used:
            notes.append(
                f"{used} 帧使用深度符号先验约束前后方向"
                f"（权重 {config.depth_sign_weight:.2f}，压平抑制 {config.depth_sign_flat_suppression:.2f}）"
            )
    skipped = len(per_frame) - len(fitted_reports)
    rigidified = sum(1 for item in per_frame if item.rigidified)
    held_frames = sum(1 for item in per_frame if item.held_joints)
    if config.hold_occluded_limbs and held_frames:
        notes.append(
            f"{held_frames} 帧有肢体被遮挡/突变，已按上一姿态的相对骨段角度保持"
            f"（上限 {config.occlusion_max_frames} 帧）"
        )
    if skipped:
        tail = (
            f"其中 {rigidified} 帧已按标定骨长重建（骨长保持恒定）"
            if rigidified
            else "原样透传"
        )
        notes.append(f"{skipped} 帧未参与 2D 拟合（无效 / 缺 world landmarks / 可见度不足），{tail}")
    if _least_squares is None:  # pragma: no cover - 依赖缺失
        notes.append("未安装 scipy，拟合被跳过")
    if passes > 1:
        notes.append(f"迭代细化 {p + 1} 趟（收敛阈值 {config.refine_tolerance * 100:.1f} cm）")
    if config.enabled and config.head_depth.enabled:
        notes.append(
            "头部骨段深度专项：压平权重 ×"
            f"{config.head_depth.flat_boost:.1f}、门控指数 {config.head_depth.flat_gate_power:.1f}、"
            f"朝输入方向回归 ×{config.head_depth.prior_scale:.2f}"
            "（正面机位下头的前后位置无观测，只能靠这个先验站在颈上方）"
        )

    return result, FitReport(
        applied=bool(fitted_reports),
        frames=len(ordered),
        frames_fitted=len(fitted_reports),
        frames_skipped=skipped,
        frames_rigidified=rigidified,
        frames_occlusion_held=held_frames,
        iterations=sum(item.iterations for item in fitted_reports),
        cost=round(float(np.mean([item.cost for item in fitted_reports])), 6)
        if fitted_reports
        else 0.0,
        reproj_error_norm=round(
            float(np.mean([item.reproj_error_norm for item in fitted_reports])), 6
        )
        if fitted_reports
        else 0.0,
        bone_length_error_pct=round(
            float(np.max([item.bone_length_error_pct for item in fitted_reports])), 4
        )
        if fitted_reports
        else 0.0,
        converged=all(item.converged for item in fitted_reports) if fitted_reports else True,
        milliseconds_per_frame=round(elapsed * 1000.0 / max(1, len(ordered)), 3),
        bone_lengths=lengths.joint_lengths(),
        per_frame=per_frame,
        notes=notes,
    )


def bone_length_cv(frames: Sequence[PoseFrame]) -> dict[str, float]:
    """各骨段长度的变异系数（std/mean），用于质检与基准对比。"""
    series: dict[str, list[float]] = {node.key: [] for node in BONE_SEGMENTS}
    for frame in sorted(frames, key=lambda item: item.index):
        if not frame.valid or len(frame.world_landmarks) < LANDMARK_COUNT:
            continue
        positions = _node_positions(frame.world_landmarks)
        if not positions:
            continue
        for key, value in _segment_lengths(positions).items():
            series[key].append(value)
    result: dict[str, float] = {}
    for key, values in series.items():
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if mean <= 1e-9:
            continue
        result[key] = float(np.std(values) / mean)
    return result


__all__ = [
    "BoneLengths",
    "FitConfig",
    "FitReport",
    "FrameFitReport",
    "HeadDepthConfig",
    "FrameFitter",
    "bone_length_cv",
    "calibrate_bone_lengths",
    "estimate_image_scale",
    "fit_frame",
    "flip_signed_depth",
    "fit_frames",
    "load_bone_lengths_override",
    "node_positions",
]
