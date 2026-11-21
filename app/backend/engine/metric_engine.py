"""米制（metric）增强引擎的**契约与降级路径**（``docs/todo-kick-depth-accuracy.md`` §7）。

这个文件故意**不实现推理内核**：

- 米制 3D 抬升（VideoPose3D / MotionBERT / AGFormer 等）的权重许可都要求
  "非商业 / 研究用途"，商用发行版不能内置（决策见 ``docs/packaging.md`` §7）；
- 没有权重与许可结论之前写"看起来能跑"的推理代码只会掩盖问题。

所以这里只把三件事准备好：

1. **契约**：接口与 ``PoseEngine`` 完全一致（``prepare`` / ``infer`` / ``probabilities``
   等），将来接入任何米制模型都不用改调用方；
2. **可用性探测**：缺 ``torch`` / 缺权重时 ``availability()`` 返回 False，
   ``prepare()`` 抛 ``EngineUnavailableError``，由 ``registry.select_engine()``
   自动回落到 MediaPipe（这是"可选增强"不拖垮主链的关键）；
3. **注册点**：只有 ``DETECT_BODY_ENABLE_METRIC_ENGINE=1`` 时才注册，
   因此默认构建里它既不占显存也不影响选型顺序。

启用步骤（源码模式）::

    pip install -e ".[metric]"
    set DETECT_BODY_ENABLE_METRIC_ENGINE=1
    set DETECT_BODY_METRIC_MODEL=models/metric-h36m.onnx

注意：即使注册成功，``infer()`` 仍会抛 ``EngineError("推理内核未实现")`` —— 这是
刻意的，保证"声明了但没接完"的状态不会被误当成可用引擎。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings
from app.backend.engine.base import (
    EngineCapabilities,
    EngineError,
    EngineUnavailableError,
    PoseEngine,
    PoseFrame,
    engine_tier,
)

logger = get_logger("engine.metric")

METRIC_MODEL_FILENAME = "metric-h36m.onnx"


def _deps_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401, PLC0415 - 仅探测
    except Exception as exc:  # noqa: BLE001 - 缺包 / 二进制不兼容
        return False, f"未安装 metric 依赖（{exc}）"
    return True, ""


def resolve_metric_model(settings: Settings | None) -> str | None:
    """显式配置 → ``models_dir`` → 打包资源根 ``models/``；都没有返回 None。"""
    explicit = getattr(settings, "metric_model_path", None) if settings else None
    if explicit:
        path = Path(explicit)
        return str(path) if path.is_file() else None
    if settings is None:
        return None
    candidates = [
        Path(settings.models_dir) / METRIC_MODEL_FILENAME,
        Path(settings.static_dir).parents[2] / "models" / METRIC_MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


class MetricPoseEngine(PoseEngine):
    """米制增强引擎的占位实现：契约齐全、推理未接。"""

    name = "metric"
    display_name = "米制增强（可选，未随发行版分发）"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        plan: object | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(settings=settings, plan=plan, model_path=model_path)
        self.tier = engine_tier(settings, plan)
        self._model: str | None = None

    @classmethod
    def availability(
        cls, settings: Settings | None = None, plan: object | None = None
    ) -> tuple[bool, str]:
        if not getattr(settings, "enable_metric_engine", False):
            return False, "米制引擎未启用（DETECT_BODY_ENABLE_METRIC_ENGINE=1 才注册）"
        ok, reason = _deps_available()
        if not ok:
            return False, reason
        if resolve_metric_model(settings) is None:
            return False, f"未找到米制模型权重（{METRIC_MODEL_FILENAME}）"
        return True, "米制引擎已注册，但推理内核尚未接入（仅契约与降级路径）"

    def prepare(self, device: str | None = None) -> None:
        if self._prepared:
            return
        ok, reason = _deps_available()
        if not ok:
            raise EngineUnavailableError(
                "米制引擎依赖缺失",
                details={"reason": reason, "hint": 'pip install -e ".[metric]"'},
            )
        model = self.model_path or resolve_metric_model(self.settings)
        if model is None:
            raise EngineUnavailableError(
                "米制引擎缺少权重",
                details={
                    "hint": f"把权重放到 models/{METRIC_MODEL_FILENAME}，"
                    "或在设置里指定 DETECT_BODY_METRIC_MODEL"
                },
            )
        self._model = model
        self._prepared = True
        self._notes.append("米制引擎已加载契约；推理内核未实现，实际分析会回落 MediaPipe")

    def infer(
        self, image: np.ndarray, *, index: int = 0, timestamp_ms: float = 0.0
    ) -> list[PoseFrame]:
        raise EngineError(
            "米制引擎的推理内核尚未实现",
            details={
                "reason": "这是刻意的：没有许可结论之前不接入权重",
                "hint": "把 engine 设为 auto / mediapipe；或先完成 docs/packaging.md §7 的决策",
            },
        )

    def close(self) -> None:
        """占位引擎没有会话要释放；保留实现以满足 ``PoseEngine`` 的抽象契约。"""
        self._model = None
        self._prepared = False

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            display_name=self.display_name,
            multi_person=False,
            world_landmarks=True,
            hands=False,
            face=False,
            providers=["metric-unavailable"],
            model_tier=self.tier,
            device=None,
            notes=self._notes + [f"model={self._model or '(未加载)'}"],
        )


__all__ = ["METRIC_MODEL_FILENAME", "MetricPoseEngine", "resolve_metric_model"]
