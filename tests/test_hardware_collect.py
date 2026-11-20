"""硬件采集冒烟测试：真实机器上不能抛异常，关键字段要有合理取值。"""

from __future__ import annotations

from app.backend.engine.acceleration import build_hardware_report
from app.backend.engine.hardware import collect_system_info


def test_collect_system_info_is_robust() -> None:
    info = collect_system_info()
    assert info.platform in {"Windows", "Darwin", "Linux"}
    assert info.cpu.logical_cores >= 1
    assert info.cpu.physical_cores >= 1
    assert info.cpu.model
    assert isinstance(info.onnx_providers, list)
    assert isinstance(info.notes, list)


def test_hardware_report_structure() -> None:
    report = build_hardware_report()
    plan = report.acceleration
    assert plan.preferred_provider
    assert plan.execution_providers
    assert plan.execution_providers[-1] == "CPUExecutionProvider"
    assert 1 <= plan.threads <= 64
    assert plan.model_tier in {"lite", "full", "heavy"}
    assert plan.reasons
