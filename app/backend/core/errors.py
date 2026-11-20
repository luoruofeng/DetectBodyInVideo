"""业务异常：统一带 code / message / details，由 FastAPI 异常处理器转成 JSON。"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有可预期错误的基类；前端据 ``code`` 决定展示方式。"""

    status_code: int = 400
    code: str = "app_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class AnalysisCancelledError(AppError):
    """任务被用户取消（分析 / 导出）；取消后必须立刻释放线程与显存。"""

    status_code = 409
    code = "analysis_cancelled"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"
