"""
exceptions.py — 自定义业务异常与全局异常处理器
业务代码只需 raise 对应的异常类，全局处理器统一转换为标准 ResponseModel 格式。
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .error_codes import ERROR_HTTP_STATUS, ERROR_MESSAGES, ErrorCode

log = logging.getLogger("api")


class APIException(Exception):
    """所有业务异常的基类。"""

    def __init__(self, error_code: ErrorCode, message: str | None = None):
        self.error_code = error_code
        self.message = message or ERROR_MESSAGES.get(error_code, "未知错误")
        self.http_status = ERROR_HTTP_STATUS.get(error_code, 500)
        super().__init__(self.message)


class ParamError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.PARAM_ERROR, message)


class AuthFailedError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.AUTH_FAILED, message)


class DocNotFoundError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.DOC_NOT_FOUND, message)


class SessionNotFoundError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.SESSION_NOT_FOUND, message)


class ModelCallError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.MODEL_CALL_FAILED, message)


class ServiceNotReadyError(APIException):
    def __init__(self, message: str | None = None):
        super().__init__(ErrorCode.SERVICE_NOT_READY, message)


def _error_body(error_code: int, message: str, request: Request) -> dict:
    return {
        "code": int(error_code),
        "message": message,
        "data": None,
        "request_id": getattr(request.state, "request_id", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def register_exception_handlers(app: FastAPI) -> None:
    """在 FastAPI app 上注册全局异常处理器，务必在路由注册前调用一次。"""

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        log.warning(f"[{getattr(request.state, 'request_id', '?')}] APIException: "
                    f"code={exc.error_code} message={exc.message}")
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_body(exc.error_code, exc.message, request),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # pydantic/FastAPI 参数校验失败（如 query 为空、top_k 超出范围）统一归为 PARAM_ERROR
        first_error = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first_error.get("loc", []) if p != "body")
        detail = f"{loc}: {first_error.get('msg', '参数校验失败')}" if loc else "参数校验失败"
        log.warning(f"[{getattr(request.state, 'request_id', '?')}] 参数校验失败: {detail}")
        return JSONResponse(
            status_code=ERROR_HTTP_STATUS[ErrorCode.PARAM_ERROR],
            content=_error_body(ErrorCode.PARAM_ERROR, detail, request),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # 兜底：任何未被上面两类捕获的异常都不能让服务器返回裸露的 500 堆栈
        log.error(f"[{getattr(request.state, 'request_id', '?')}] 未处理异常: {exc!r}", exc_info=exc)
        return JSONResponse(
            status_code=ERROR_HTTP_STATUS[ErrorCode.INTERNAL_ERROR],
            content=_error_body(ErrorCode.INTERNAL_ERROR, "服务内部错误", request),
        )
