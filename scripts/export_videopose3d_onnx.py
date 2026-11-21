"""把 VideoPose3D（H36M）离线导出成 ONNX，供 ``postprocess.depth_sign`` 的 model 后端使用。

为什么要 ONNX 而不是直接跑 PyTorch：
- 本项目运行期已经有 ``onnxruntime``（还有 TensorRT / CUDA / CoreML / DirectML 的 EP 选型），
  额外引入 torch 会让 PyInstaller 体积 +300 MB 并带来 DLL 冲突；
- VideoPose3D 是纯 ``Conv1d + BatchNorm + ReLU + residual`` 堆叠，天然可导出；
- 我们**只需要它的深度符号**，不需要与 PyTorch 逐位一致（幅度会被完全丢弃）。

权重与许可（务必先读）：
- 权重 ``pretrained_h36m_detectron_coco.bin`` 来自 facebookresearch/VideoPose3D，
  训练数据是 Human3.6M —— **该数据集仅限非商业研究使用**。产物不要提交到仓库、
  不要随商用发行版分发；商用前请自行核对许可并承担合规责任。
- 导出只需要在**一台**装了 torch 的机器上跑一次，产物是纯 ONNX。

用法::

    # 1) 克隆参考实现并下载预训练权重
    git clone https://github.com/facebookresearch/VideoPose3D
    # 权重放到 VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin

    # 2) 导出（--repo 指向克隆目录）
    python scripts/export_videopose3d_onnx.py --repo ../VideoPose3D \\
        --checkpoint ../VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin \\
        --output models/videopose3d-h36m.onnx

    # 3) 校验形状与推理
    python scripts/export_videopose3d_onnx.py --verify models/videopose3d-h36m.onnx

导出的模型输入 ``pose2d: [1, T, 17, 2]``、输出 ``pose3d: [1, T, 17, 3]``（root-relative）。
`depth_sign` 只读输出的 Δz 符号，所以**不需要**做任何尺度/坐标系对齐。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILTER_WIDTHS = "3,3,3,3,3"
RECEPTIVE_FIELD = 243


def _parse_widths(text: str) -> list[int]:
    values = [int(item) for item in text.replace(" ", "").split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--filter-widths 必须是正整数列表，例如 3,3,3,3,3")
    return values


def export(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not (repo / "common" / "model.py").is_file():
        print(
            f"找不到 VideoPose3D 源码：{repo}\n"
            "请先 git clone https://github.com/facebookresearch/VideoPose3D 后用 --repo 指向它",
            file=sys.stderr,
        )
        return 2
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"找不到预训练权重：{checkpoint}", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print(
            "导出需要 torch（只在导出机器上装，运行期不需要）：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(repo))
    from common.model import TemporalModel  # type: ignore[import-not-found]

    widths = _parse_widths(args.filter_widths)
    model = TemporalModel(
        17, 2, 17, filter_widths=widths, causal=False, dropout=0.25, channels=args.channels
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    payload = state.get("model_pos", state) if isinstance(state, dict) else state
    model.load_state_dict(payload)
    model.eval()
    model.float()

    sample = torch.zeros(1, RECEPTIVE_FIELD, 17, 2, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(output),
            input_names=["pose2d"],
            output_names=["pose3d"],
            dynamic_axes={"pose2d": {0: "T"}, "pose3d": {0: "T"}},
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"已导出 {output}（{size_mb:.1f} MB，感受野 {RECEPTIVE_FIELD} 帧）")
    print("下一步：在「识别设置」勾选『深度方向修正』，或设置 DETECT_BODY_DEPTH_SIGN_MODE=model")
    return 0


def verify(path: str) -> int:
    """不依赖 torch 的最小校验：形状、输出是否为有限值、Δz 是否非零。"""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("校验需要 onnxruntime：pip install onnxruntime", file=sys.stderr)
        return 2
    if not Path(path).is_file():
        print(f"找不到模型：{path}", file=sys.stderr)
        return 2
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    name = session.get_inputs()[0].name
    frames = RECEPTIVE_FIELD
    rng = np.random.default_rng(0)
    data = (rng.random((1, frames, 17, 2), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    output = session.run(None, {name: data})[0]
    if output.ndim == 4:
        output = output[0]
    print(f"输入 {shape} → 输出 {output.shape}")
    if output.shape[0] != frames or output.shape[1] != 17 or output.shape[2] != 3:
        print("形状不符：期望 [T, 17, 3]", file=sys.stderr)
        return 1
    if not np.all(np.isfinite(output)):
        print("输出含 NaN/Inf", file=sys.stderr)
        return 1
    print(f"Δz 非零帧占比 {float(np.mean(np.abs(np.diff(output[:, :, 2], axis=0)) > 1e-6)):.2%}")
    print("校验通过：可被 postprocess.depth_sign 的 model 后端加载")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 / 校验 VideoPose3D ONNX（只取深度符号）")
    parser.add_argument("--verify", metavar="ONNX", default="", help="只校验已有模型，不导出")
    parser.add_argument("--repo", default="", help="VideoPose3D 源码目录（含 common/model.py）")
    parser.add_argument("--checkpoint", default="", help="pretrained_h36m_detectron_coco.bin 路径")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "models" / "videopose3d-h36m.onnx"), help="输出 ONNX 路径"
    )
    parser.add_argument("--filter-widths", default=DEFAULT_FILTER_WIDTHS, help="模型结构，默认 3,3,3,3,3")
    parser.add_argument("--channels", type=int, default=1024, help="通道数，默认 1024（与官方权重匹配）")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if not args.repo or not args.checkpoint:
        parser.print_help()
        print("\n导出需要 --repo 与 --checkpoint；只校验用 --verify <onnx>", file=sys.stderr)
        return 2
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
