"""导出 JSON Schema（任务 7.1）。

这一层是**交付契约**：iOS / macOS RealityKit 侧按此解析，任何不兼容变更必须
升大版本（``version`` 的 major）。字段命名统一 camelCase，与附录 A 的草案一致。

约定：
- 顶层：``version`` / ``generator`` / ``source`` / ``coordinateSystem`` / ``units`` /
  ``personHeightMeters`` / ``jointNames`` / ``frames`` / ``quality``（+ 可选 ``rootMotion``）。
- 每帧：``index`` / ``t``（秒）/ ``valid`` / ``joints`` / ``angles``。
- 坐标：米制、右手系、y 向上、-z 向前（``arkit-right-handed-y-up``）。
- 数值在导出时统一压到 4 位小数（任务 7.2），本层不负责取整。

``python -m app.backend.export.schema`` 会把 schema.json 写到包目录，作为存档。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.backend.skeleton.root_motion import Quaternion
from app.backend.skeleton.transform import COORDINATE_SYSTEM, UNITS, Vector3

SCHEMA_VERSION = "1.1.0"
GENERATOR = "DetectBodyInVideo 0.1.0"
SCHEMA_ID = "https://detect-body.local/schema/pose-export-1.1.0.json"

# 深度（Z）只是单目估计值 —— 这条常量随文档一起下发，消费方不必翻文档就能知道。
DEPTH_IS_ESTIMATED = True
SCHEMA_FILE = Path(__file__).with_name("schema.json")

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

# allow_inf_nan=False：NaN/Inf 会让 iOS 端解析直接崩，必须在写盘前拦住
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class JointOut(BaseModel):
    """关节点 + 置信度；joints 使用米，rawLandmarks 使用原始归一化坐标。"""

    model_config = ConfigDict(extra="forbid")

    x: FiniteFloat = 0.0
    y: FiniteFloat = 0.0
    z: FiniteFloat = 0.0
    visibility: float = Field(0.0, ge=0, le=1)


class SourceInfo(BaseModel):
    """源视频元信息，iOS 侧据此做时间轴与画幅对齐。"""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1, description="源文件名（不含目录）")
    fps: FiniteFloat = Field(gt=0)
    width: int = Field(gt=0, description="纠正旋转后的显示宽度")
    height: int = Field(gt=0)
    frames: int = Field(ge=0, description="源视频总帧数")
    durationSeconds: FiniteFloat | None = Field(None, ge=0)


class ExportFrame(BaseModel):
    """一帧姿态。``t`` 为相对视频起点的秒数。"""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    t: FiniteFloat = Field(ge=0)
    valid: bool = True
    interpolated: bool = False
    joints: dict[str, JointOut] = Field(
        default_factory=dict,
        description="髋中心相对坐标，单位米、Y 向上；不是相对各自父关节的局部坐标",
    )
    angles: dict[str, FiniteFloat] = Field(default_factory=dict)
    rawLandmarks: dict[str, JointOut] | None = Field(
        None,
        description="原始归一化 33 点，不是米：x/y 相对图像宽/高，y 向下，z 为髋相对深度且与 x 同尺度",
    )

    @model_validator(mode="after")
    def _check_angles(self) -> ExportFrame:
        for name, value in self.angles.items():
            if value < 0 or value > 180:
                raise ValueError(f"角度 {name}={value} 超出 0~180°")
        return self


class QualityOut(BaseModel):
    """数据质检报告，与 ``qualityReport.verdict`` 一起决定"能用 / 建议重录"。"""

    model_config = ConfigDict(extra="forbid")

    framesTotal: int = Field(0, ge=0)
    framesValid: int = Field(0, ge=0)
    validFrameRatio: float = Field(0.0, ge=0, le=1)
    interpolatedFrames: int = Field(0, ge=0)
    meanVisibility: float = Field(0.0, ge=0, le=1)
    meanScore: float = Field(0.0, ge=0, le=1)
    maxJump: FiniteFloat = Field(0.0, ge=0)
    maxJumpFrame: int | None = None
    # 1.1.0 新增（可选、有默认值，旧文件仍可读入）
    zJitterRms: FiniteFloat = Field(0.0, ge=0, description="相邻帧 z 的 RMS（米）")
    boneLengthCv: FiniteFloat = Field(0.0, ge=0, description="骨长变异系数最大值（0~1）")
    meanReprojError: FiniteFloat = Field(0.0, ge=0, description="3D 反投影与 2D 的平均误差（归一化）")
    depthConfidence: float = Field(0.0, ge=0, le=1, description="深度可信度 0~1")
    verdict: Literal["ready", "review", "rerecord"] = "rerecord"
    verdictLabel: str = ""
    issues: list[str] = Field(default_factory=list)
    smoothing: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RootMotionFrameOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    t: FiniteFloat = Field(ge=0)
    translation: Vector3 = Field(default_factory=Vector3)
    rotation: Quaternion = Field(default_factory=Quaternion)


class RootMotionOut(BaseModel):
    """根运动（任务 6.3）：位移挂 root 节点，旋转挂髋节点。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["world", "in_place"] = "world"
    totalTranslation: FiniteFloat = Field(0.0, ge=0)
    convention: str = ""
    frames: list[RootMotionFrameOut] = Field(default_factory=list)


class ExportDocument(BaseModel):
    """完整导出文档 = 附录 A 的正式定义。"""

    model_config = ConfigDict(extra="forbid")

    version: str = SCHEMA_VERSION
    generator: str = GENERATOR
    source: SourceInfo
    coordinateSystem: str = COORDINATE_SYSTEM
    units: str = UNITS
    depthIsEstimated: bool = Field(
        DEPTH_IS_ESTIMATED,
        description="恒为 true：z（前后方向）是单目估计值，绝对深度与前后位移不可信",
    )
    depthConfidence: float = Field(
        0.0,
        ge=0,
        le=1,
        description="整段视频的深度可信度 0~1；与 quality.depthConfidence 同值，方便消费方只读顶层",
    )
    warnings: list[str] = Field(
        default_factory=list, description="导出时的可信度提示（中文），消费方应展示给用户"
    )
    personHeightMeters: FiniteFloat = Field(
        1.70,
        gt=0.5,
        le=2.5,
        description=(
            "系统从世界坐标自动测量的参考高度（头中心→脚底，米），"
            "比真实身高偏小约 10%；仅作缩放核对，不参与骨骼坐标计算"
        ),
    )
    jointNames: list[str] = Field(min_length=1)
    frames: list[ExportFrame] = Field(default_factory=list)
    quality: QualityOut = Field(default_factory=QualityOut)
    rootMotion: RootMotionOut | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_document(self) -> ExportDocument:
        if not _VERSION_PATTERN.match(self.version):
            raise ValueError(f"version 必须形如 1.0.0，实际 {self.version!r}")
        if self.coordinateSystem != COORDINATE_SYSTEM:
            raise ValueError(
                f"coordinateSystem 必须是 {COORDINATE_SYSTEM}，实际 {self.coordinateSystem!r}"
            )
        if self.units != UNITS:
            raise ValueError(f"units 必须是 {UNITS}，实际 {self.units!r}")

        names = set(self.jointNames)
        if len(names) != len(self.jointNames):
            raise ValueError("jointNames 存在重复项")
        previous_t: float | None = None
        previous_index: int | None = None
        for frame in self.frames:
            unknown = set(frame.joints) - names
            if unknown:
                raise ValueError(f"第 {frame.index} 帧出现未声明的关节：{sorted(unknown)[:5]}")
            if previous_t is not None and frame.t < previous_t - 1e-6:
                raise ValueError(f"帧时间戳非单调递增：{previous_t} → {frame.t}")
            if previous_index is not None and frame.index <= previous_index:
                raise ValueError(f"帧号必须严格递增：{previous_index} → {frame.index}")
            previous_t, previous_index = frame.t, frame.index

        if self.rootMotion is not None:
            motion_indices = {item.index for item in self.rootMotion.frames}
            if motion_indices and not motion_indices <= {frame.index for frame in self.frames}:
                raise ValueError("rootMotion 引用了不存在的帧号")
        return self


def schema_payload() -> dict[str, object]:
    """生成可存档的 schema 描述（含版本号与坐标约定）。"""
    payload = ExportDocument.model_json_schema()
    payload["$id"] = SCHEMA_ID
    payload["title"] = "DetectBodyInVideo Pose Export"
    payload["x-schema-version"] = SCHEMA_VERSION
    payload["x-coordinate-system"] = COORDINATE_SYSTEM
    payload["x-units"] = UNITS
    payload["x-depth-is-estimated"] = DEPTH_IS_ESTIMATED
    payload["x-generated-by"] = GENERATOR
    return payload


def write_schema_json(path: Path | str = SCHEMA_FILE) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(schema_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_schema_json(path: Path | str = SCHEMA_FILE) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def major_version(version: str = SCHEMA_VERSION) -> int:
    """兼容性判断只认 major：同 major 视为向后兼容。"""
    return int(version.split(".", 1)[0])


def is_compatible(version: str) -> bool:
    return _VERSION_PATTERN.match(version) is not None and major_version(version) == major_version()


if __name__ == "__main__":  # pragma: no cover - 存档用命令行
    written = write_schema_json()
    print(f"schema 已写入 {written}（schemaVersion={SCHEMA_VERSION}）")


__all__ = [
    "COORDINATE_SYSTEM",
    "DEPTH_IS_ESTIMATED",
    "GENERATOR",
    "SCHEMA_FILE",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "UNITS",
    "ExportDocument",
    "ExportFrame",
    "JointOut",
    "QualityOut",
    "RootMotionFrameOut",
    "RootMotionOut",
    "SourceInfo",
    "is_compatible",
    "load_schema_json",
    "major_version",
    "schema_payload",
    "write_schema_json",
]
