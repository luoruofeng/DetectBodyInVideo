"""确定性样例数据（任务 7.3 / 7.4 的回归基准）。

``samples/expected_pose.json`` 由本模块生成，不依赖任何视频/模型权重：
用解析公式造 30 帧"抬手 + 迈步"的 33 点，走一遍真实的
后处理 → 骨骼映射 → 坐标转换 → 导出装配链路，因此能当 golden file 用。

重新生成：

    python -m app.backend.export.samples
"""

from __future__ import annotations

import math
from pathlib import Path

from app.backend.engine.base import LANDMARK_COUNT, Landmark, PoseFrame
from app.backend.export.builder import ExportConfig, build_export_document
from app.backend.export.schema import ExportDocument, SourceInfo
from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess
from app.backend.skeleton.bundle import SkeletonConfig, build_skeleton_bundle

SAMPLE_FRAMES = 30
SAMPLE_FPS = 30.0
SAMPLE_NAME = "synthetic_wave.mp4"

SAMPLES_DIR = Path(__file__).resolve().parents[3] / "samples"
SAMPLE_PATH = SAMPLES_DIR / "expected_pose.json"

# 归一化坐标（y 向下）的静态骨架；只有手腕/肘部随时间摆动
_BASE_LAYOUT: dict[int, tuple[float, float, float]] = {
    0: (0.500, 0.150, -0.02),  # nose
    1: (0.492, 0.142, -0.02),
    2: (0.488, 0.140, -0.02),
    3: (0.484, 0.142, -0.02),
    4: (0.508, 0.142, -0.02),
    5: (0.512, 0.140, -0.02),
    6: (0.516, 0.142, -0.02),
    7: (0.470, 0.148, -0.01),
    8: (0.530, 0.148, -0.01),
    9: (0.492, 0.162, -0.02),
    10: (0.508, 0.162, -0.02),
    11: (0.430, 0.280, 0.00),  # left shoulder
    12: (0.570, 0.280, 0.00),
    13: (0.410, 0.410, 0.00),  # left elbow
    14: (0.590, 0.410, 0.00),
    15: (0.400, 0.540, 0.00),  # left wrist
    16: (0.600, 0.540, 0.00),
    17: (0.395, 0.565, 0.00),
    18: (0.605, 0.565, 0.00),
    19: (0.392, 0.560, 0.00),
    20: (0.608, 0.560, 0.00),
    21: (0.385, 0.552, 0.00),
    22: (0.615, 0.552, 0.00),
    23: (0.452, 0.560, 0.00),  # left hip
    24: (0.548, 0.560, 0.00),
    25: (0.455, 0.750, 0.00),  # left knee
    26: (0.545, 0.750, 0.00),
    27: (0.455, 0.940, 0.00),  # left ankle
    28: (0.545, 0.940, 0.00),
    29: (0.455, 0.958, 0.00),
    30: (0.545, 0.958, 0.00),
    31: (0.455, 0.978, 0.00),  # left foot index
    32: (0.545, 0.978, 0.00),
}

VISIBILITY = 0.95


def _animated_layout(phase: float) -> dict[int, tuple[float, float, float]]:
    """抬手 + 轻微迈步；``phase`` 为 0~2π。"""
    layout = dict(_BASE_LAYOUT)
    swing = math.sin(phase)
    left_wrist_y = 0.540 - 0.075 * (swing + 1.0) / 2.0
    right_wrist_y = 0.540 - 0.075 * (1.0 - swing) / 2.0
    left_elbow_y = 0.410 - 0.030 * (swing + 1.0) / 2.0
    right_elbow_y = 0.410 - 0.030 * (1.0 - swing) / 2.0
    layout[13] = (layout[13][0], left_elbow_y, layout[13][2])
    layout[14] = (layout[14][0], right_elbow_y, layout[14][2])
    layout[15] = (layout[15][0], left_wrist_y, layout[15][2])
    layout[16] = (layout[16][0], right_wrist_y, layout[16][2])
    for index in (17, 19, 21):
        layout[index] = (layout[index][0], left_wrist_y + 0.025, layout[index][2])
    for index in (18, 20, 22):
        layout[index] = (layout[index][0], right_wrist_y + 0.025, layout[index][2])

    step = 0.012 * swing
    layout[23] = (layout[23][0] - step, layout[23][1], layout[23][2])
    layout[24] = (layout[24][0] + step, layout[24][1], layout[24][2])
    layout[25] = (layout[25][0] - step * 1.6, layout[25][1], layout[25][2])
    layout[26] = (layout[26][0] + step * 1.6, layout[26][1], layout[26][2])
    layout[27] = (layout[27][0] - step * 2.2, layout[27][1], layout[27][2])
    layout[28] = (layout[28][0] + step * 2.2, layout[28][1], layout[28][2])
    # 镜像成"面向镜头"的手性：MediaPipe 约定面向镜头时人物左侧落在图像右侧（x > 0.5）。
    # ``_BASE_LAYOUT`` 原本写反了（左肩在 x<0.5），与本文件"人物面向镜头、鼻尖 z 为负"
    # 的其它约定矛盾 —— 那会让 ``mapping._body_forward`` 判出"背对镜头"，把鼻尖前伸量
    # 加到反方向，进而让 head_joint 凭空前探。
    return {index: (1.0 - x, y, z) for index, (x, y, z) in layout.items()}


def synthetic_pose_frames(
    count: int = SAMPLE_FRAMES, fps: float = SAMPLE_FPS
) -> list[PoseFrame]:
    """生成确定性的 33 点序列（归一化 + 米制 world landmarks）。"""
    frames: list[PoseFrame] = []
    for index in range(count):
        phase = 2.0 * math.pi * index / max(1, count - 1)
        layout = _animated_layout(phase)
        landmarks: list[Landmark] = []
        world: list[Landmark] = []
        for point_index in range(LANDMARK_COUNT):
            x, y, z = layout.get(point_index, (0.5, 0.5, 0.0))
            landmarks.append(Landmark(x=x, y=y, z=z, visibility=VISIBILITY))
            # world landmarks：米制、y 向下、原点≈两髋中点
            world.append(
                Landmark(
                    x=(x - 0.5) * 0.62,
                    y=(y - 0.56) * 0.92,
                    z=z * 0.3,
                    visibility=VISIBILITY,
                )
            )
        frames.append(
            PoseFrame(
                index=index,
                timestamp_ms=round(index * 1000.0 / fps, 4),
                landmarks=landmarks,
                world_landmarks=world,
                score=VISIBILITY,
                valid=True,
                track_id=1,
            )
        )
    return frames


def build_sample_document() -> ExportDocument:
    """走完整后处理 + 骨骼链路，产出与线上导出完全同构的文档。"""
    pose_frames = synthetic_pose_frames()
    postprocess = run_postprocess(
        pose_frames,
        PostProcessConfig(stride=1, fps=SAMPLE_FPS),
    )
    skeleton = build_skeleton_bundle(
        postprocess.frames,
        SkeletonConfig(include_optional=True),
    )
    angles = {
        frame.index: values
        for frame, values in zip(postprocess.frames, postprocess.angles, strict=False)
    }
    return build_export_document(
        source=SourceInfo(
            file=SAMPLE_NAME,
            fps=SAMPLE_FPS,
            width=1080,
            height=1920,
            frames=len(pose_frames),
            durationSeconds=round(len(pose_frames) / SAMPLE_FPS, 4),
        ),
        skeleton=skeleton,
        quality=postprocess.quality,
        angles=angles,
        pose_frames=postprocess.frames,
        config=ExportConfig(
            granularity="full",
            include_raw_landmarks=True,
            round_digits=4,
            include_root_motion=True,
        ),
    )


def write_sample(path: Path | str = SAMPLE_PATH) -> Path:
    from app.backend.export.writer import write_document

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_document(build_sample_document(), target, mode="full", validate=True)
    return target


if __name__ == "__main__":  # pragma: no cover - 生成 golden file
    written = write_sample()
    print(f"样例已写入 {written}")


__all__ = [
    "SAMPLE_FPS",
    "SAMPLE_FRAMES",
    "SAMPLE_NAME",
    "SAMPLE_PATH",
    "build_sample_document",
    "synthetic_pose_frames",
    "write_sample",
]
