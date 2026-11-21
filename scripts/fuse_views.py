"""双机位（正面 + 侧面）角度融合 CLI（``docs/todo-kick-depth-accuracy.md`` §6.2）。

它只做**角度级**融合，不做三角化：两段手机视频几乎没有同步时间戳，强行三角化
会把"不同时刻的两个人"解出一个不存在的 3D；而消费方要的是关节角（左右对称性、
ROM），角度本身就活在"该机位看得清的平面"里。

用法::

    python scripts/fuse_views.py --front samples/benchmark/BM-05-front-kick.mp4 \\
        --side samples/benchmark/BM-06-side-kick.mp4 \\
        --out out/fused-kick.json

    # 只跑一个机位也能用（会标记 degraded 并写一条"矢状面不可验证"的 note）
    python scripts/fuse_views.py --front BM-05-front-kick.mp4

输出 JSON 结构（消费方直接按 ``frames[i].angles`` 读）::

    {
      "views": ["front", "side"],
      "degraded": false,
      "frames": 240,
      "results": [{"index": 0, "angles": {"left_knee": 168.2, ...}}, ...],
      "notes": [...]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.core.logging import get_logger  # noqa: E402
from app.backend.multiview.fusion import (  # noqa: E402
    FusionConfig,
    ViewInput,
    fuse_views,
    view_from_frames,
)
from app.backend.pipeline import AnalysisOptions, analyze_video  # noqa: E402

logger = get_logger("scripts.fuse_views")


def _median_yaw(analysis) -> float:
    """取整段视频 yaw 的中位数作为机位朝向（正对镜头 0°，正侧面 90°）。"""
    values = [abs(estimate.degrees) for estimate in analysis.postprocess.yaw if estimate]
    if not values:
        return 0.0
    values.sort()
    return float(values[len(values) // 2])


def _phases(analysis) -> dict[str, int]:
    report = analysis.postprocess.kick_phases
    if not report.detected:
        return {}
    return dict(report.phases)


def build_view(name: str, path: str, *, override_yaw: float | None, options: AnalysisOptions) -> ViewInput:
    analysis = analyze_video(path, options=options)
    yaw = override_yaw if override_yaw is not None else _median_yaw(analysis)
    view = view_from_frames(
        name,
        analysis.postprocess.frames,
        yaw_deg=yaw,
        phases=_phases(analysis),
        fps=float(analysis.sampling.fps or 30.0),
    )
    logger.info(
        "机位 %s：%s 帧，yaw≈%.1f°，相位=%s",
        name,
        len(view.angles),
        yaw,
        view.phases or "未检测到",
    )
    return view


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双机位角度级融合（正面 + 侧面）")
    parser.add_argument("--front", default="", help="正面机位视频（额状面角度的主视角）")
    parser.add_argument("--side", default="", help="正侧面机位视频（矢状面角度的主视角）")
    parser.add_argument("--front-yaw", type=float, default=None, help="覆盖正面机位的 yaw（度）")
    parser.add_argument("--side-yaw", type=float, default=None, help="覆盖侧面机位的 yaw（度）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "out" / "fused-angles.json"))
    args = parser.parse_args(argv)

    if not args.front and not args.side:
        parser.print_help()
        print("\n至少给一个机位：--front 或 --side", file=sys.stderr)
        return 2

    options = AnalysisOptions()
    views: list[ViewInput] = []
    if args.front:
        views.append(
            build_view("front", args.front, override_yaw=args.front_yaw, options=options)
        )
    if args.side:
        views.append(build_view("side", args.side, override_yaw=args.side_yaw, options=options))

    fused = fuse_views(views, FusionConfig())
    payload = {
        "views": fused.views,
        "degraded": fused.degraded,
        "frames": fused.frames,
        "results": [
            {"index": index, "angles": angles} for index, angles in enumerate(fused.angles)
        ],
        "notes": fused.notes,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"融合结果已写入 {target}（{fused.frames} 帧，机位 {fused.views}）")
    if fused.degraded:
        print("注意：只有单机位，矢状面角度（踢腿幅度）不可验证", file=sys.stderr)
    for note in fused.notes:
        print(f"  · {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
