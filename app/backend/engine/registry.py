"""引擎注册表与自动选型（任务 4.1 / 4.3 的"失败自动降级"）。

选型策略：
1. 显式指定（``settings.pose_engine`` / 调用方参数）时优先用它；
2. ``auto``：GPU 可用 → 先试 ONNX（RTMPose/RTMO），否则先 MediaPipe；
3. 任一环节不可用/初始化失败，记录原因并降级到下一个，全部失败才抛
   ``EngineUnavailableError``（附完整原因链，供 UI 展示）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.backend.core.errors import AppError
from app.backend.core.logging import get_logger
from app.backend.core.settings import Settings, get_settings
from app.backend.engine.base import (
    EngineCapabilities,
    EngineSelection,
    EngineUnavailableError,
    PoseEngine,
)
from app.backend.engine.mediapipe_engine import MediaPipePoseEngine
from app.backend.engine.onnx_engine import OnnxPoseEngine

logger = get_logger("engine.registry")

EngineFactory = Callable[..., PoseEngine]

_FACTORIES: dict[str, EngineFactory] = {}
_ORDER: dict[str, int] = {}


def register_engine(name: str, factory: EngineFactory, *, priority: int = 100) -> None:
    _FACTORIES[name] = factory
    _ORDER[name] = priority


def registered_engines() -> list[str]:
    return sorted(_FACTORIES, key=lambda item: _ORDER.get(item, 100))


def create_engine(name: str, **kwargs: Any) -> PoseEngine:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise EngineUnavailableError(
            f"未注册的推理引擎：{name}", details={"registered": registered_engines()}
        )
    return factory(**kwargs)


def describe_engines(settings: Settings | None = None) -> list[dict[str, Any]]:
    """给 UI 的引擎清单：谁可用、缺什么。"""
    rows: list[dict[str, Any]] = []
    for name in registered_engines():
        engine_class = getattr(_FACTORIES[name], "engine_class", None)
        available, reason = (True, "")
        if engine_class is not None:
            try:
                available, reason = engine_class.availability(settings)
            except Exception as exc:  # noqa: BLE001 - 探测不允许抛
                available, reason = False, str(exc)
        rows.append({"name": name, "available": available, "reason": reason})
    return rows


def _candidates(
    requested: str,
    plan: object | None,
    settings: Settings,
) -> list[str]:
    registered = [item for item in registered_engines()]
    if requested in {"mediapipe", "onnx"}:
        others = [item for item in registered if item != requested]
        return [requested, *others]

    gpu = bool(getattr(plan, "gpu_accelerated", False))
    preferred = "onnx" if (gpu and "onnx" in registered) else "mediapipe"
    ordered = [preferred, *[item for item in registered if item != preferred]]
    return ordered


def select_engine(
    plan: object | None = None,
    *,
    settings: Settings | None = None,
    preferred: str | None = None,
    runtime: object | None = None,
    **engine_kwargs: Any,
) -> tuple[PoseEngine, EngineSelection]:
    """按 ``AccelerationPlan`` 选引擎并完成 ``prepare()``；失败自动降级。"""
    settings = settings or get_settings()
    requested = (preferred or settings.pose_engine or "auto").lower()
    if requested not in {"auto", "mediapipe", "onnx"}:
        requested = "auto"

    reasons: list[str] = []
    candidates = _candidates(requested, plan, settings)
    for index, name in enumerate(candidates):
        if name not in _FACTORIES:
            reasons.append(f"{name}：未注册")
            continue
        engine_class = getattr(_FACTORIES[name], "engine_class", None)
        if engine_class is not None:
            available, reason = engine_class.availability(settings)
            if not available:
                reasons.append(f"{name} 不可用：{reason}")
                logger.info("跳过引擎 %s：%s", name, reason)
                continue

        engine: PoseEngine | None = None
        try:
            engine = create_engine(name, settings=settings, plan=plan, **engine_kwargs)
            engine.apply_runtime(runtime)
            engine.prepare()
        except AppError as exc:
            reasons.append(f"{name} 初始化失败：{exc.message}")
            logger.warning("引擎 %s 初始化失败：%s", name, exc.message)
            if engine is not None:
                engine.close()
            continue
        except Exception as exc:  # noqa: BLE001 - 第三方库异常类型不可控
            reasons.append(f"{name} 初始化异常：{exc}")
            logger.warning("引擎 %s 初始化异常", name, exc_info=True)
            if engine is not None:
                engine.close()
            continue

        selection = EngineSelection(
            engine=engine.name,
            display_name=engine.display_name,
            requested=requested,
            degraded=index > 0,
            reasons=reasons + engine.notes,
            capabilities=engine.capabilities,
            model_tier=engine.capabilities.model_tier,
        )
        logger.info(
            "选定引擎 %s（请求 %s，降级 %s）", engine.name, requested, selection.degraded
        )
        return engine, selection

    raise EngineUnavailableError(
        "没有可用的姿态推理引擎",
        details={
            "requested": requested,
            "candidates": candidates,
            "reasons": reasons,
            "hint": "安装 mediapipe，或把 RTMPose/RTMO 的 ONNX 权重放入 models/ 目录",
        },
    )


def _register_defaults() -> None:
    register_engine("mediapipe", _named_factory(MediaPipePoseEngine, "mediapipe"), priority=20)
    register_engine("onnx", _named_factory(OnnxPoseEngine, "onnx"), priority=10)


def _named_factory(engine_class: type[PoseEngine], name: str) -> EngineFactory:
    def factory(**kwargs: Any) -> PoseEngine:
        return engine_class(**kwargs)

    factory.engine_class = engine_class  # type: ignore[attr-defined]
    factory.engine_name = name  # type: ignore[attr-defined]
    return factory


def capabilities_of(engine: PoseEngine | None) -> EngineCapabilities | None:
    return engine.capabilities if engine is not None else None


_register_defaults()
