"""抬升器后端抽象与降级路径（``docs/todo-kick-depth-accuracy.md`` §3.4）。

四条硬约束，任何一条被破坏都会让"可选模型"变成"随时爆炸的可选依赖"：

1. **符号能还原**：给一段已知 3D 真值的序列，后端输出的 Δz 符号必须与真值一致；
2. **失败不抛异常**：权重缺失 / 输出含 NaN / 形状不符，一律退回 ``temporal``；
3. **镜像能自愈**：模型 z 轴整体取反时，``auto_calibrate`` 要判出来并整体翻转；
4. **EP 选型只有一处**：``engine/acceleration.py::onnx_execution_providers()``。
"""

from __future__ import annotations

import numpy as np
import pytest
import synth_pose

from app.backend.engine.acceleration import onnx_execution_providers
from app.backend.postprocess import depth_sign
from app.backend.postprocess.depth_sign import DepthSignConfig, OnnxLifter

ASPECT = synth_pose.ASPECT


# --------------------------------------------------------------------------- #
# 夹具：把真值骨架映射成 H36M 17 点
# --------------------------------------------------------------------------- #
def _h36m_truth(frames: list) -> np.ndarray:
    """``[T,17,3]``：H36M 顺序的**真值**世界坐标（未映射的槽位填 0）。"""
    from app.backend.skeleton.topology import node_map

    nodes = node_map()
    rows = []
    for frame in frames:
        row = np.zeros((17, 3), dtype=np.float64)
        for key, h36m_index in depth_sign.H36M_TO_NODE.items():
            node = nodes.get(key)
            if node is None or node.index is None:
                continue
            point = frame.world_landmarks[node.index]
            row[h36m_index] = (point.x, point.y, point.z)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class FakeLifter:
    """内存里的假抬升器：直接回放给定输出，或按需抛错。"""

    def __init__(
        self,
        output: np.ndarray | None = None,
        *,
        error: Exception | None = None,
        name: str = "motionbert",
        receptive_field: int = 128,
    ) -> None:
        self.name = name
        self.receptive_field = receptive_field
        self._output = output
        self._error = error
        self.loaded = False
        self.calls: list[tuple[int, int]] = []

    def availability(self) -> tuple[bool, str]:
        return True, ""

    def load(self) -> None:
        self.loaded = True

    def infer(self, sequence: np.ndarray) -> np.ndarray:
        self.calls.append(tuple(sequence.shape))
        if self._error is not None:
            raise self._error
        assert self._output is not None
        return self._output

    @property
    def providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def _use_lifter(monkeypatch: pytest.MonkeyPatch, lifter: FakeLifter) -> None:
    monkeypatch.setattr(depth_sign, "build_lifter_backend", lambda config: lifter)


# --------------------------------------------------------------------------- #
# 1. 符号还原
# --------------------------------------------------------------------------- #
def test_synthetic_truth_signs_are_recovered(monkeypatch):
    frames = synth_pose.make_kick_frames(45, peak_deg=70.0, ramp=0.45)
    truth = _h36m_truth(frames)
    lifter = FakeLifter(truth)
    _use_lifter(monkeypatch, lifter)

    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )

    assert lifter.loaded is True
    assert report.mode == "model", report.notes
    assert report.lifter == "motionbert"
    assert report.auto_calibrate_ratio is not None and report.auto_calibrate_ratio > 0.55
    assert report.calibration == "passed"
    assert signs

    # 每一帧下发的符号都必须与真值一致（真值 = 上面那张 H36M 表）
    from app.backend.skeleton.topology import node_map

    nodes = node_map()
    wrong = 0
    compared = 0
    for frame_index, entries in signs.items():
        row = truth[frame_index]
        for key, entry in entries.items():
            # §4 之后条目可能是 (符号, 置信度) 或 (符号, 置信度, 方向)，这里只关心符号
            sign = entry[0]
            h36m_index = depth_sign.H36M_TO_NODE[key]
            parent = nodes[key].parent
            parent_h36m = depth_sign.H36M_TO_NODE.get(parent or "")
            if parent_h36m is None:
                continue
            dz = row[h36m_index, 2] - row[parent_h36m, 2]
            if abs(dz) < 1e-6:
                continue
            compared += 1
            wrong += int((1 if dz > 0 else -1) != sign)
    assert compared > 0
    assert wrong == 0, f"{wrong}/{compared} 个符号与真值不符"


# --------------------------------------------------------------------------- #
# 2. 降级路径
# --------------------------------------------------------------------------- #
def _baseline_frames() -> list:
    return synth_pose.make_kick_frames(45, peak_deg=60.0, ramp=0.45)


def test_nan_output_falls_back_to_temporal(monkeypatch):
    lifter = FakeLifter(error=ValueError("抬升器输出含 NaN/Inf"))
    _use_lifter(monkeypatch, lifter)
    signs, report = depth_sign.estimate_depth_signs(
        _baseline_frames(), DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )
    assert report.mode == "temporal"
    assert report.applied is True  # 时序路径仍然给出符号
    assert any("推理失败" in note for note in report.notes), report.notes


def test_missing_weights_falls_back_to_temporal(monkeypatch):
    lifter = FakeLifter()
    lifter.availability = lambda: (False, "未找到 motionbert 权重（motionbert-h36m.onnx）")  # type: ignore[method-assign]
    _use_lifter(monkeypatch, lifter)
    signs, report = depth_sign.estimate_depth_signs(
        _baseline_frames(), DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )
    assert report.mode == "temporal"
    assert report.lifter == "motionbert"  # 仍然告诉调用方"尝试过哪个"
    assert any("权重" in note for note in report.notes), report.notes


def test_wrong_output_shape_is_rejected(monkeypatch):
    """形状不符必须被后端自身拒绝，而不是把错数据带进符号提取。"""

    class Session:
        def __init__(self, output: np.ndarray) -> None:
            self._output = output

        def run(self, *_args, **_kwargs):
            return [self._output]

        def get_inputs(self):
            return [type("Input", (), {"name": "pose2d", "shape": [1, "T", 17, 2]})()]

    lifter = OnnxLifter("videopose3d", "dummy.onnx")
    monkeypatch.setattr(lifter, "_session", Session(np.zeros((5, 17, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="形状不符"):
        lifter.infer(np.zeros((5, 17, 2), dtype=np.float32))

    nan = np.zeros((5, 17, 3), dtype=np.float32)
    nan[2, 3, 1] = np.nan
    monkeypatch.setattr(lifter, "_session", Session(nan))
    with pytest.raises(ValueError, match="NaN"):
        lifter.infer(np.zeros((5, 17, 2), dtype=np.float32))


def test_fixed_window_model_is_chunked():
    """把 T 烘死的导出（固定窗口）也要能吃下更长的序列。"""

    class EchoSession:
        def __init__(self, window: int) -> None:
            self.window = window

        def run(self, _outputs, feed):
            data = next(iter(feed.values()))
            assert data.shape[1] == self.window
            result = np.zeros((1, self.window, 17, 3), dtype=np.float32)
            result[0, :, :, 0] = data[0, :, :, 0]
            result[0, :, :, 2] = 0.5
            return [result]

        def get_inputs(self):
            return [type("Input", (), {"name": "pose2d", "shape": [1, 8, 17, 2]})()]

    lifter = OnnxLifter("videopose3d", "dummy.onnx")
    lifter._session = EchoSession(8)  # type: ignore[assignment]
    lifter._window = 8
    sequence = np.zeros((21, 17, 2), dtype=np.float32)
    sequence[:, :, 0] = np.arange(21, dtype=np.float32)[:, None]
    output = lifter.infer(sequence)
    assert output.shape == (21, 17, 3)
    assert np.allclose(output[:, 0, 0], np.arange(21))


# --------------------------------------------------------------------------- #
# 3. 镜像自校准
# --------------------------------------------------------------------------- #
def test_mirrored_lifter_output_is_inverted(monkeypatch):
    frames = synth_pose.make_kick_frames(45, peak_deg=70.0, ramp=0.45)
    truth = _h36m_truth(frames)
    mirrored = truth.copy()
    mirrored[:, :, 2] = -mirrored[:, :, 2]
    _use_lifter(monkeypatch, FakeLifter(mirrored))

    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )
    assert report.calibration == "inverted", report.notes
    assert report.mode == "model"
    assert report.auto_calibrate_ratio is not None and report.auto_calibrate_ratio < 0.3

    # 取反后必须回到真值分支：至少大腿（膝）方向的符号全对
    wrong = 0
    for frame_index, entries in signs.items():
        entry = entries.get("left_knee")
        if entry is None:
            continue
        knee = truth[frame_index][2, 2] - truth[frame_index][1, 2]
        if abs(knee) < 1e-6:
            continue
        wrong += int((1 if knee > 0 else -1) != entry[0])
    assert wrong == 0


def test_random_output_is_discarded(monkeypatch):
    """一致率落在 0.3~0.55 说明模型只是噪声：判定无效并退回时序，而不是硬用。

    这里不是"随机数碰运气"：把 5/9 的帧整体翻转 z（每帧要么全对要么全错），
    一致率就稳定落在 ≈4/9 ≈ 0.44（中段），可以确定性地触发"判定无效"分支。
    """
    frames = _baseline_frames()
    truth = _h36m_truth(frames)
    mixed = truth.copy()
    flip = [position for position in range(len(frames)) if position % 9 < 5]
    mixed[flip, :, 2] *= -1.0
    _use_lifter(monkeypatch, FakeLifter(mixed))
    signs, report = depth_sign.estimate_depth_signs(
        frames, DepthSignConfig(enabled=True, mode="model"), aspect=ASPECT
    )
    assert report.mode == "temporal"
    assert report.calibration == "discarded", (report.auto_calibrate_ratio, report.notes)
    assert report.auto_calibrate_ratio is not None
    assert 0.3 <= report.auto_calibrate_ratio < 0.55
    assert signs  # 时序路径仍给出结果


# --------------------------------------------------------------------------- #
# 4. EP 选型只有一处
# --------------------------------------------------------------------------- #
def test_execution_providers_are_ordered_with_cpu_last():
    providers = onnx_execution_providers(
        ["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"]
    )
    assert providers[-1] == "CPUExecutionProvider"
    assert providers[0] == "CUDAExecutionProvider"
    assert "DmlExecutionProvider" in providers
    # 未知 provider 不应被加进来
    assert set(providers) <= {
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    }
