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

    # 1) 下载预训练权重（约 45 MB，写进 models/）并克隆参考实现
    python scripts/export_videopose3d_onnx.py --download-checkpoint
    git clone https://github.com/facebookresearch/VideoPose3D

    # 2) 导出（需要 torch，只在导出机器上装；运行期不需要）
    python scripts/export_videopose3d_onnx.py --repo ../VideoPose3D \\
        --output models/videopose3d-h36m.onnx

    # 3) 校验（不依赖 torch：形状、两个不同 T、数值有限性）
    python scripts/export_videopose3d_onnx.py --verify models/videopose3d-h36m.onnx

导出的模型输入 ``pose2d: [1, T, 17, 2]``、输出 ``pose3d: [1, T, 17, 3]``（root-relative，
时间轴 T 动态）。官方模型是 valid 卷积、本身只吃"感受野 243 帧"的窗口，导出时把
**两侧边缘填充 + 裁回原长**烘进了图里，所以运行期直接喂任意长度即可。
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

# 官方权重（facebookresearch/VideoPose3D README 上的下载地址，约 45 MB）。
# 只在导出机器上下载一次；产物（ONNX）不进仓库、不随发行版分发（H36M 仅限非商业研究）。
CHECKPOINT_FILENAME = "pretrained_h36m_detectron_coco.bin"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/video-pose-3d/{CHECKPOINT_FILENAME}"


def _parse_widths(text: str) -> list[int]:
    values = [int(item) for item in text.replace(" ", "").split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--filter-widths 必须是正整数列表，例如 3,3,3,3,3")
    return values


def download_checkpoint(target: Path) -> int:
    """把官方预训练权重下到本机（只下不导出：导出还需要 torch 与 VideoPose3D 源码）。"""
    import shutil
    import urllib.request

    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"权重已存在，跳过下载：{target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"正在下载 {CHECKPOINT_URL}")
    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:  # noqa: S310
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:  # pragma: no cover - 网络不可用
        partial.unlink(missing_ok=True)
        print(f"下载失败：{exc}\n请手动下载后放到 {target}", file=sys.stderr)
        return 2
    partial.replace(target)
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"已保存 {target}（{size_mb:.1f} MB）")
    print(
        "下一步：\n"
        "  git clone https://github.com/facebookresearch/VideoPose3D\n"
        f"  python scripts/export_videopose3d_onnx.py --repo ../VideoPose3D "
        f"--checkpoint {target} --output models/videopose3d-h36m.onnx"
    )
    return 0


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

    # 官方 ``TemporalModel`` 是 **valid 卷积**：输入 T 帧只会吐出 ``T − (RF − 1)`` 帧，
    # 官方 ``evaluate.py`` 因此先两侧各填 ``(RF−1)//2`` 帧（边缘复制）。我们在图里补上这一步，
    # 运行期才能直接喂 ``[1,T,17,2]`` 拿 ``[1,T,17,3]`` —— 否则默认后端得自己猜窗口，
    # 送 T=243 只会得到 1 帧输出。
    pad = (int(model.receptive_field()) - 1) // 2

    class _PaddedTemporal(torch.nn.Module):
        """把"两侧边缘填充 + 裁回原长度"烘进 ONNX，输入输出同长。"""

        def __init__(self, inner: torch.nn.Module, pad_frames: int) -> None:
            super().__init__()
            self.inner = inner
            self.pad_frames = pad_frames

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            frames = x.shape[1]
            if self.pad_frames > 0:
                head = x[:, :1].expand(-1, self.pad_frames, -1, -1)
                tail = x[:, -1:].expand(-1, self.pad_frames, -1, -1)
                x = torch.cat([head, x, tail], dim=1)
            return self.inner(x)[:, :frames]

    wrapper = _PaddedTemporal(model, pad).eval()
    # 用非感受野长度的样例导出，才能证明时间轴真是动态的（用 243 会把 T 烘死）
    sample = torch.zeros(1, 97, 17, 2, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 只把**时间轴**标成动态，批固定为 1：轴 0 是 batch、轴 1 才是 T，
    # 标错轴会导出成 ``[T,243,17,2]`` 这种"时间轴冻在首维"的图，运行期喂任意帧数都形状不符。
    kwargs: dict[str, object] = {
        "input_names": ["pose2d"],
        "output_names": ["pose3d"],
        "dynamic_axes": {"pose2d": {1: "T"}, "pose3d": {1: "T"}},
        "opset_version": 17,
        "do_constant_folding": True,
    }
    with torch.no_grad():
        try:
            # 优先走旧版导出器：单文件、无外部 data，与文档里"一个 .onnx 放进 models/"的约定一致
            torch.onnx.export(wrapper, sample, str(output), dynamo=False, **kwargs)
        except TypeError:  # pragma: no cover - 旧版 torch 没有 dynamo 参数
            torch.onnx.export(wrapper, sample, str(output), **kwargs)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"已导出 {output}（{size_mb:.1f} MB，感受野 {RECEPTIVE_FIELD} 帧，内部自动边缘填充）")
    print("下一步：在「识别设置」勾选『深度方向修正』，后端选「自动」即可；或用 --verify 校验")
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
    rng = np.random.default_rng(0)
    # 必须**换两个长度**各跑一次：只测一个长度的话，"时间轴被冻死"的导错图也能蒙混过关。
    for frames in (RECEPTIVE_FIELD, 64):
        data = (rng.random((1, frames, 17, 2), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
        output = session.run(None, {name: data})[0]
        if output.ndim == 4:
            output = output[0]
        print(f"输入 {shape}（T={frames}）→ 输出 {output.shape}")
        if output.shape[0] != frames or output.shape[1] != 17 or output.shape[2] != 3:
            print(
                f"形状不符：T={frames} 时期望 [T, 17, 3]，实际 {output.shape}"
                "（多半是 dynamic_axes 标错了轴：时间轴是 1，不是 0）",
                file=sys.stderr,
            )
            return 1
        if not np.all(np.isfinite(output)):
            print("输出含 NaN/Inf", file=sys.stderr)
            return 1
        if frames == RECEPTIVE_FIELD:
            share = float(np.mean(np.abs(np.diff(output[:, :, 2], axis=0)) > 1e-6))
            print(f"Δz 非零帧占比 {share:.2%}")
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
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help=f"只下载官方预训练权重（{CHECKPOINT_FILENAME}，约 45 MB，非商业）到 models/，不导出",
    )
    parser.add_argument("--filter-widths", default=DEFAULT_FILTER_WIDTHS, help="模型结构，默认 3,3,3,3,3")
    parser.add_argument("--channels", type=int, default=1024, help="通道数，默认 1024（与官方权重匹配）")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if args.download_checkpoint:
        return download_checkpoint(
            PROJECT_ROOT / "models" / CHECKPOINT_FILENAME
        )
    if not args.checkpoint:
        # 允许省略：之前用 --download-checkpoint 下过就直接用
        cached = PROJECT_ROOT / "models" / CHECKPOINT_FILENAME
        if cached.is_file():
            args.checkpoint = str(cached)
    if not args.repo or not args.checkpoint:
        parser.print_help()
        print(
            "\n导出需要 --repo 与 --checkpoint（可先用 --download-checkpoint 下权重）；"
            "只校验用 --verify <onnx>",
            file=sys.stderr,
        )
        return 2
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
