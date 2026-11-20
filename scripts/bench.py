"""推理链路基准测试（任务 9.2）。

用法：

    # 真实视频 + 真实引擎（推荐在目标机器上跑）
    python scripts/bench.py samples/dance.mp4 --engine auto --batch 1,4 --tiers lite,full

    # 无模型 / 无视频时验证基准工具本身（stub 引擎，只测链路不测模型）
    python scripts/bench.py --engine stub --synthetic --batch 1,4 --stride 2

输出：终端表格 + 写入 ``docs/benchmark.md``（替换 BENCH 标记之间的内容）。
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.core.settings import Settings, get_settings  # noqa: E402
from app.backend.engine.acceleration import build_hardware_report  # noqa: E402
from app.backend.engine.base import (  # noqa: E402
    LANDMARK_COUNT,
    EngineCapabilities,
    Landmark,
    PoseEngine,
    PoseFrame,
)
from app.backend.engine.registry import select_engine  # noqa: E402
from app.backend.media.probe import probe_video  # noqa: E402
from app.backend.media.sampling import SamplingConfig, build_analysis_plan  # noqa: E402
from app.backend.perf.config import RuntimeConfig  # noqa: E402
from app.backend.perf.runner import run_inference  # noqa: E402

BENCH_START = "<!-- BENCH:START -->"
BENCH_END = "<!-- BENCH:END -->"


class StubPoseEngine(PoseEngine):
    """不加载任何模型，只模拟"每帧一次矩阵运算"的开销，用于验证基准工具。"""

    name = "stub"
    display_name = "Stub Engine（仅基准工具自检）"

    def __init__(self, *, work: int = 64, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._work = work
        self._matrix = np.random.default_rng(0).random((work, work)).astype(np.float32)

    @classmethod
    def availability(cls, settings: Settings | None = None) -> tuple[bool, str]:
        return True, ""

    def prepare(self, device: str | None = None) -> None:
        self._prepared = True

    def infer(self, image, *, index: int = 0, timestamp_ms: float = 0.0):
        return self.infer_batch([(index, timestamp_ms, image)])[0]

    def infer_batch(self, frames):
        results = []
        for index, timestamp, _image in frames:
            # 模拟 CPU/GPU 工作量：work×work 矩阵连乘
            vector = self._matrix
            for _ in range(2):
                vector = vector @ self._matrix
            if not np.isfinite(vector).all():  # pragma: no cover - 防优化掉
                raise RuntimeError("stub 计算异常")
            landmarks = [
                Landmark(x=0.5, y=0.5, z=0.0, visibility=0.95)
                for _ in range(LANDMARK_COUNT)
            ]
            results.append(
                [
                    PoseFrame(
                        index=index,
                        timestamp_ms=timestamp,
                        landmarks=landmarks,
                        score=0.95,
                        valid=True,
                    )
                ]
            )
        return results

    def close(self) -> None:
        self._prepared = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            display_name=self.display_name,
            supports_batching=True,
            model_tier="stub",
            notes=["stub 引擎不代表任何真实模型性能"],
        )


def make_synthetic_video(directory: Path, frames: int = 60) -> Path:
    path = directory / "bench_synthetic.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    if not writer.isOpened():
        raise SystemExit("当前 OpenCV 缺少 mp4v 编码器，无法生成基准视频")
    for index in range(frames):
        frame = np.full((240, 320, 3), 24, dtype=np.uint8)
        x = 40 + (index * 3) % 240
        cv2.rectangle(frame, (x, 50), (x + 40, 230), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def machine_label() -> str:
    parts = [platform.system(), platform.machine(), f"Python {platform.python_version()}"]
    return " / ".join(part for part in parts if part)


def providers_label(settings: Settings) -> str:
    try:
        import onnxruntime as ort  # noqa: PLC0415

        return ", ".join(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return "onnxruntime 未安装"


def run_case(
    video: Path,
    *,
    engine_name: str,
    tier: str,
    batch: int,
    stride: int,
    max_frames: int | None,
    settings: Settings,
) -> dict[str, object]:
    probe = probe_video(video)
    plan = build_analysis_plan(video, probe, SamplingConfig(every_n_frames=stride))

    if engine_name == "stub":
        engine: PoseEngine = StubPoseEngine(settings=settings)
        engine.prepare()
        acceleration = build_hardware_report(settings).acceleration
        ep = "stub"
        effective_tier = "stub"
    else:
        acceleration = build_hardware_report(settings).acceleration.model_copy(
            update={"model_tier": tier}
        )
        try:
            engine, selection = select_engine(
                acceleration,
                settings=settings,
                preferred="auto" if engine_name == "auto" else engine_name,
                runtime=RuntimeConfig(batch_size=batch),
            )
        except Exception as exc:  # noqa: BLE001 - 基准不应因缺模型中断
            return {
                "engine": engine_name,
                "tier": tier,
                "batch": batch,
                "stride": stride,
                "fps": None,
                "frames": 0,
                "peak_mb": 0.0,
                "ep": "-",
                "error": str(exc),
            }
        ep = ", ".join(selection.capabilities.providers) or "unknown"
        effective_tier = selection.model_tier

    started = time.perf_counter()
    tracemalloc.start()
    try:
        outcome = run_inference(
            engine,
            video,
            probe,
            plan,
            config=RuntimeConfig(batch_size=batch),
            settings=settings,
            max_frames=max_frames,
        )
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        engine.close()
    elapsed = time.perf_counter() - started

    return {
        "engine": engine.name,
        "tier": effective_tier,
        "batch": outcome.batch_size,
        "stride": stride,
        "fps": round(outcome.processed / elapsed, 2) if elapsed > 0 else 0.0,
        "frames": outcome.processed,
        "peak_mb": round(peak / 1024**2, 2),
        "ep": ep,
        "error": "",
    }


def render_markdown(rows: list[dict[str, object]], *, label: str) -> str:
    lines = [
        BENCH_START,
        f"### 运行记录：{label}",
        "",
        f"- 机器：`{machine_label()}`",
        f"- ONNX EP：`{providers_label(get_settings())}`",
        "",
        "| 引擎 | 档位 | EP | batch | stride | 帧数 | fps | 内存峰值 | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        fps = "—" if row["fps"] is None else f"{row['fps']:.2f}"
        lines.append(
            "| {engine} | {tier} | {ep} | {batch} | {stride} | {frames} | {fps} | "
            "{peak_mb:.1f} MiB | {note} |".format(
                engine=row["engine"],
                tier=row["tier"],
                ep=row["ep"],
                batch=row["batch"],
                stride=row["stride"],
                frames=row["frames"],
                fps=fps,
                peak_mb=float(row["peak_mb"]),
                note=row["error"] or "ok",
            )
        )
    lines.extend(
        [
            "",
            "> 说明：stub 行只验证基准工具与生产/消费链路，不代表真实模型性能；"
            "真实结论请用目标机器 + 目标模型重跑。",
            BENCH_END,
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = "# 性能基准\n"
    if BENCH_START in text and BENCH_END in text:
        head, rest = text.split(BENCH_START, 1)
        _, tail = rest.split(BENCH_END, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DetectBodyInVideo 推理链路基准")
    parser.add_argument("video", nargs="?", type=Path, help="样例视频路径")
    parser.add_argument("--synthetic", action="store_true", help="自动生成合成视频")
    parser.add_argument("--engine", default="stub", choices=["stub", "auto", "mediapipe", "onnx"])
    parser.add_argument("--tiers", default="lite,full", help="模型档位（逗号分隔）")
    parser.add_argument("--batch", default="1,4", help="批大小列表（逗号分隔）")
    parser.add_argument("--stride", type=int, default=1, help="采样步长")
    parser.add_argument("--max-frames", type=int, default=0, help="最多测多少帧（0=全部）")
    parser.add_argument("--label", default="", help="运行标签（写入文档）")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs" / "benchmark.md"
    )
    parser.add_argument("--no-write", action="store_true", help="只打印，不写文档")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    temp_dir = None
    if args.video is not None:
        video = args.video
        if not video.is_file():
            raise SystemExit(f"视频不存在：{video}")
    elif args.synthetic:
        temp_dir = tempfile.TemporaryDirectory()
        video = make_synthetic_video(Path(temp_dir.name))
    else:
        raise SystemExit("请提供视频路径或加 --synthetic")

    batches = [int(item) for item in str(args.batch).split(",") if item.strip()]
    tiers = ["stub"] if args.engine == "stub" else [item for item in args.tiers.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    try:
        for tier in tiers:
            for batch in batches:
                print(f"→ engine={args.engine} tier={tier} batch={batch} ...", flush=True)
                row = run_case(
                    video,
                    engine_name=args.engine,
                    tier=tier,
                    batch=batch,
                    stride=max(1, args.stride),
                    max_frames=args.max_frames or None,
                    settings=settings,
                )
                rows.append(row)
                if row["fps"] is None:
                    print(f"   失败：{row['error']}")
                else:
                    print(
                        f"   {row['frames']} 帧 / {row['fps']} fps / "
                        f"峰值 {row['peak_mb']} MiB / EP {row['ep']}"
                    )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    label = args.label or datetime.now().strftime("%Y-%m-%d %H:%M")
    block = render_markdown(rows, label=label)
    print()
    print(block)
    if not args.no_write:
        write_markdown(args.output, block)
        print(f"\n已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
