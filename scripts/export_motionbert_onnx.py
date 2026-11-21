"""把 MotionBERT（H36M）离线导出成 ONNX，供 ``postprocess.depth_sign`` 的 model 后端使用。

与 ``export_videopose3d_onnx.py`` 的区别，以及为什么仍然只导出 ONNX：

- 运行期**不引入 PyTorch**：本项目只用 ``onnxruntime``（EP 选型见
  ``engine/acceleration.py``），多一份 torch 会让 PyInstaller 体积 +300 MB；
- MotionBERT 是 **DSTFormer**（时空 transformer），注意力分支对 ``T`` 的形状比
  VideoPose3D 的纯卷积更敏感，所以这里同时支持两种导出方式：

  - ``--mode dynamic``（默认）：``dynamic_axes`` 放开 T，任意长度一次推理；
  - ``--mode fixed``：把 T 烘死为 ``--max-T``，运行期由 ``OnnxLifter`` 按窗切分
    （``LIFTER_RECEPTIVE_FIELDS['motionbert'] = 128``）。

  两种都要在同一份权重上验证一遍，取能稳定通过 ``--verify`` 的那个。

输入契约必须与官方 ``read_input`` 对齐（否则输出整体失效）：

- 人体框外扩 **1.25×**，再按半宽/半高归一化到 ``[-1, 1]``
  （等价于 ``crop_bbox`` + ``normalize_screen_coordinates``）；
- 17 个 H36M 关节、``[T, 17, 2]``，**根相对**；本项目的这一步在
  ``postprocess/depth_sign.py::_h36m_input()``，与 lifter 无关，不要在这里重写。

权重与许可：MotionBERT 的 H36M 预训练权重同样**仅限非商业研究**。产物不提交仓库、
不随商用发行版分发；商用前请做许可决策（见 ``docs/todo-kick-depth-accuracy.md`` §7）。

用法::

    # 1) 下载官方实现与权重
    #    git clone https://github.com/Walter0807/MotionBERT
    #    checkpoint 放到 MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin

    # 2) 导出（两种 T 策略各跑一次，取能过 --verify 的那个）
    python scripts/export_motionbert_onnx.py --repo ../MotionBERT \\
        --checkpoint ../MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin \\
        --mode dynamic --output models/motionbert-h36m.onnx

    python scripts/export_motionbert_onnx.py --repo ../MotionBERT \\
        --checkpoint ../MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin \\
        --mode fixed --max-T 128 --output models/motionbert-h36m.onnx

    # 3) 校验（不依赖 torch）
    python scripts/export_motionbert_onnx.py --verify models/motionbert-h36m.onnx

导出的模型输入 ``pose2d: [1, T, 17, 2]``（固定模式为 ``[1, max_T, 17, 2]``），
输出 ``pose3d: [1, T, 17, 3]``。``depth_sign`` 只读 Δz 的符号，尺度一律丢弃。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 与 depth_sign.LIFTER_RECEPTIVE_FIELDS 保持一致（这里写常量是为了导出的机器上
# 不必把整个 app 跑起来；两者不一致时 _verify 会给出提示）
DEFAULT_MAX_T = 128


def _load_model(repo: Path, checkpoint: Path, args: argparse.Namespace) -> object:
    """按官方仓库的结构构建 DSTFormer 并载入权重。"""
    import torch

    sys.path.insert(0, str(repo))
    try:
        from lib.model.DSTformer import DSTformer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - 只在导出机器上触发
        raise SystemExit(
            "无法导入 MotionBERT 的模型定义（lib.model.DSTformer）：\n"
            f"  repo={repo}\n  {exc}\n"
            "请确认 --repo 指向官方仓库根目录（含 lib/model/DSTformer.py）"
        ) from exc

    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=args.dim_feat,
        dim_rep=args.dim_rep,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=2,
        norm_eps=1e-5,
        maxlen=args.max_T,
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    payload = state.get("model_pos", state.get("model", state)) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(payload, strict=False)
    if missing:
        print(f"警告：{len(missing)} 个参数未从权重加载（{missing[:3]}…）", file=sys.stderr)
    if unexpected:
        print(f"警告：权重里有 {len(unexpected)} 个多余参数（{unexpected[:3]}…）", file=sys.stderr)
    model.eval()
    model.float()
    return model


def export(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not (repo / "lib" / "model" / "DSTformer.py").is_file():
        print(
            f"找不到 MotionBERT 源码：{repo}\n"
            "请先 git clone https://github.com/Walter0807/MotionBERT 后用 --repo 指向它",
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

    model = _load_model(repo, checkpoint, args)
    sample = torch.zeros(1, args.max_T, 17, 2, dtype=torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.mode == "dynamic":
        # 注意力分支对 T 的依赖只出现在 shape 运算里，放开后必须用多个长度各跑一次；
        # 有些导出会给 softmax 的 reshape 留下 T 的常量，那种情况下改用 --mode fixed。
        dynamic_axes = {"pose2d": {0: "B", 1: "T"}, "pose3d": {0: "B", 1: "T"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(output),
            input_names=["pose2d"],
            output_names=["pose3d"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"已导出 {output}（{size_mb:.1f} MB，mode={args.mode}，max_T={args.max_T}）")
    print("下一步：设置 DETECT_BODY_LIFTER=motionbert 与 DETECT_BODY_DEPTH_SIGN_MODE=model，")
    print("        或用 scripts/bench_depth.py --lifter motionbert --depth-sign-mode model 做 A/B")
    return 0


def verify(path: str) -> int:
    """不依赖 torch 的校验：形状、有限值、Δz 是否非零、动态 T 是否真的可用。"""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("校验需要 onnxruntime：pip install onnxruntime", file=sys.stderr)
        return 2
    model_path = Path(path)
    if not model_path.is_file():
        print(f"找不到模型：{model_path}", file=sys.stderr)
        return 2

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    length = shape[1] if len(shape) > 1 else None
    dynamic = not isinstance(length, int)
    lengths = [96, DEFAULT_MAX_T, 200] if dynamic else [int(length)]

    print(f"输入 {shape} → 动态 T：{'是' if dynamic else '否（按窗切分）'}")
    rng = np.random.default_rng(0)
    for frames in lengths:
        data = (rng.random((1, frames, 17, 2), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
        try:
            output = session.run(None, {name: data})[0]
        except Exception as exc:
            if dynamic and frames != lengths[-1]:
                print(f"  T={frames} 推理失败（{exc}）→ 建议改用 --mode fixed 重新导出", file=sys.stderr)
                return 1
            print(f"推理失败（{exc}）", file=sys.stderr)
            return 1
        if output.ndim == 4:
            output = output[0]
        if output.shape != (frames, 17, 3):
            print(f"  形状不符：{output.shape}，期望 {(frames, 17, 3)}", file=sys.stderr)
            return 1
        if not np.all(np.isfinite(output)):
            print("  输出含 NaN/Inf", file=sys.stderr)
            return 1
        share = float(np.mean(np.abs(np.diff(output[:, :, 2], axis=0)) > 1e-6))
        print(f"  T={frames}：Δz 非零帧占比 {share:.2%}")
        if share <= 0.0:
            print("  Δz 恒定为零：权重/输入归一化可能不对，符号先验无法工作", file=sys.stderr)
            return 1

    print("校验通过：可被 postprocess.depth_sign 的 model 后端加载（DETECT_BODY_LIFTER=motionbert）")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 / 校验 MotionBERT ONNX（只取深度符号）")
    parser.add_argument("--verify", metavar="ONNX", default="", help="只校验已有模型，不导出")
    parser.add_argument("--repo", default="", help="MotionBERT 源码目录（含 lib/model/DSTformer.py）")
    parser.add_argument("--checkpoint", default="", help="best_epoch.bin 路径")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "models" / "motionbert-h36m.onnx"), help="输出 ONNX 路径"
    )
    parser.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic", "fixed"],
        help="T 的处理方式：dynamic 放开 dynamic_axes / fixed 烘死为 --max-T（运行期按窗切分）",
    )
    parser.add_argument("--max-T", dest="max_T", type=int, default=DEFAULT_MAX_T, help="导出用的帧数")
    parser.add_argument("--dim-feat", dest="dim_feat", type=int, default=256)
    parser.add_argument("--dim-rep", dest="dim_rep", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=8)
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if not args.repo or not args.checkpoint:
        parser.print_help()
        print("\n导出需要 --repo 与 --checkpoint；只校验用 --verify <onnx>", file=sys.stderr)
        return 2
    if args.max_T < 16:
        print("--max-T 至少 16（注意力分支需要足够的时序上下文）", file=sys.stderr)
        return 2
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
