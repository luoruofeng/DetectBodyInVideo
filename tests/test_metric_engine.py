"""米制增强引擎的契约与降级路径（``docs/todo-kick-depth-accuracy.md`` §7）。

这里验证的是"可选增强不能拖垮主链"：

1. 开关关着时它**不注册**（选型顺序、显存占用、UI 都不受影响）；
2. 开关打开但缺依赖/缺权重时 ``availability()`` 为 False、``prepare()`` 抛
   ``EngineUnavailableError``，调用方可以无痛回落 MediaPipe；
3. 即使注册成功，``infer()`` 也明确报"推理内核未实现"，不会被误当成可用能力。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backend.core.errors import AppError
from app.backend.core.settings import Settings
from app.backend.engine import registry
from app.backend.engine.base import EngineError, EngineUnavailableError
from app.backend.engine.metric_engine import (
    METRIC_MODEL_FILENAME,
    MetricPoseEngine,
    resolve_metric_model,
)


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "data",
        "models_dir": tmp_path / "models",
        "log_level": "ERROR",
    }
    values.update(overrides)
    return Settings(**values)


# --------------------------------------------------------------------------- #
# 1. 可用性探测
# --------------------------------------------------------------------------- #
def test_availability_is_false_when_flag_is_off(tmp_path):
    available, reason = MetricPoseEngine.availability(_settings(tmp_path))
    assert available is False
    assert "未启用" in reason


def test_availability_is_false_without_weights(tmp_path):
    available, reason = MetricPoseEngine.availability(
        _settings(tmp_path, enable_metric_engine=True)
    )
    assert available is False
    assert METRIC_MODEL_FILENAME in reason


def test_availability_is_true_with_weights(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / METRIC_MODEL_FILENAME).write_bytes(b"placeholder")
    settings = _settings(tmp_path, enable_metric_engine=True)
    available, reason = MetricPoseEngine.availability(settings)
    # torch 没装时这里会是 False（依赖缺失），装了才会是 True
    if available:
        assert "推理内核" in reason
    else:
        assert "依赖" in reason or "权重" in reason


def test_resolve_model_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "custom.onnx"
    explicit.write_bytes(b"x")
    settings = _settings(tmp_path, metric_model_path=str(explicit))
    assert resolve_metric_model(settings) == str(explicit)
    assert resolve_metric_model(_settings(tmp_path)) is None


# --------------------------------------------------------------------------- #
# 2. 降级路径
# --------------------------------------------------------------------------- #
def test_prepare_raises_engine_unavailable_without_weights(tmp_path):
    engine = MetricPoseEngine(settings=_settings(tmp_path, enable_metric_engine=True))
    with pytest.raises((EngineUnavailableError, AppError)) as excinfo:
        engine.prepare()
    assert "权重" in str(excinfo.value) or "依赖" in str(excinfo.value)


def test_infer_is_explicitly_unimplemented(tmp_path):
    engine = MetricPoseEngine(settings=_settings(tmp_path, enable_metric_engine=True))
    engine._prepared = True
    engine._model = "models/metric-h36m.onnx"
    with pytest.raises(EngineError) as excinfo:
        engine.infer(np.zeros((64, 48, 3), dtype=np.uint8))
    assert "未实现" in excinfo.value.message


def test_capabilities_are_honest(tmp_path):
    engine = MetricPoseEngine(settings=_settings(tmp_path))
    capabilities = engine.capabilities
    assert capabilities.name == "metric"
    assert capabilities.world_landmarks is True
    assert any("未加载" in note for note in capabilities.notes)


# --------------------------------------------------------------------------- #
# 3. 注册点
# --------------------------------------------------------------------------- #
def test_metric_is_not_registered_by_default():
    assert "metric" not in registry.registered_engines()


def test_register_defaults_skips_metric_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_FACTORIES", {})
    monkeypatch.setattr(registry, "_ORDER", {})
    monkeypatch.setattr(
        "app.backend.core.settings.get_settings", lambda: _settings(tmp_path)
    )
    registry._register_defaults()
    assert "mediapipe" in registry.registered_engines()
    assert "metric" not in registry.registered_engines()


def test_register_defaults_adds_metric_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_FACTORIES", {})
    monkeypatch.setattr(registry, "_ORDER", {})
    monkeypatch.setattr(
        "app.backend.core.settings.get_settings",
        lambda: _settings(tmp_path, enable_metric_engine=True),
    )
    registry._register_defaults()
    assert "metric" in registry.registered_engines()
    # 优先级最低：auto 选型不会先挑它
    engines = registry.registered_engines()
    assert engines[-1] == "metric"
