"""`docs/todo-kick-depth-accuracy.md` §2.1 的链路自检（无视频、无权重也能跑）。

它把"勾选深度方向修正后到底发生了什么"变成可打印的证据：

1. 后处理阶段出现 `lift` / 「修正前后方向」进度（UI 就是靠这条显示阶段名）；
2. 任务摘要 `depthSign.mode` 不再是 `off`，`flipsBefore → flipsAfter` 有下降；
3. 质检 notes 里能看到一致率/丢弃原因（`quality.py` 把 `DepthSignReport.notes` 汇入）。

用法::

    python scripts/check_depth_sign_chain.py
    python scripts/check_depth_sign_chain.py --mode model   # 无权重时应如实退回 temporal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import synth_pose  # noqa: E402

from app.backend.postprocess.pipeline import PostProcessConfig, run_postprocess  # noqa: E402


def _mirror_left_leg(frames: list, period: int = 6, width: int = 3) -> list:
    """故障注入：把部分帧的左腿沿髋部 z 平面镜像。

    2D 投影与骨长一个像素都不变，只有 ``Δz`` 反号 —— 这正是"腿翻面"的真实形态。
    """
    for frame in frames:
        if frame.index % period >= width:
            continue
        anchor = frame.world_landmarks[23].z
        for index in (25, 27, 29, 31):  # left knee / ankle / heel / foot_index
            point = frame.world_landmarks[index]
            frame.world_landmarks[index] = type(point)(
                x=point.x, y=point.y, z=2.0 * anchor - point.z, visibility=point.visibility
            )
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="深度方向修正链路自检（§2.1）")
    parser.add_argument("--mode", default="temporal", choices=["temporal", "model"])
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)

    frames = synth_pose.make_kick_frames(args.frames, peak_deg=80.0, ramp=0.4)
    frames = _mirror_left_leg(list(frames))

    stages: list[tuple[str, str]] = []

    def progress(stage: str, message: str, done: int, total: int) -> None:
        stages.append((stage, message))

    config = PostProcessConfig(
        depth_sign=PostProcessConfig().depth_sign.model_copy(
            update={"enabled": True, "mode": args.mode}
        )
    )
    result = run_postprocess(frames, config, progress=progress)

    report = result.depth_sign
    print("== 阶段进度 ==")
    for stage, message in stages:
        marker = "→" if "方向" in message else " "
        print(f" {marker} {stage}: {message}")

    print("\n== 摘要 ==")
    print(f" mode        = {report.mode}（期望 temporal / model，不能是 off）")
    print(f" segments    = {report.segments}")
    print(f" flipsBefore = {report.flips_before}")
    print(f" flipsAfter  = {report.flips_after}")
    print(f" lifter      = {report.lifter or '(未使用)'}")
    print(f" calibration = {report.calibration}  一致率={report.auto_calibrate_ratio}")
    print(f" swingLeg    = {report.swing_leg}")

    print("\n== 质检 notes ==")
    for note in result.quality.notes:
        print(f" · {note}")

    ok_stage = any(stage == "lift" for stage, _ in stages)
    ok_mode = report.mode != "off"
    ok_flips = report.flips_after <= report.flips_before
    ok_notes = any("方向" in note for note in result.quality.notes)
    print("\n== 判定 ==")
    for name, passed in (
        ("后处理出现 lift 阶段", ok_stage),
        ("mode 不再是 off", ok_mode),
        ("翻转次数未增加", ok_flips),
        ("质检 notes 里有方向说明", ok_notes),
    ):
        print(f" {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all((ok_stage, ok_mode, ok_flips, ok_notes)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
