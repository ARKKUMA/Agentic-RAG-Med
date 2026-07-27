"""
error_codes.py — 统一错误码枚举
约定：0 表示成功；非 0 均为业务错误码，四位数字，首位表示错误类别：
  1xxx 参数/请求错误   2xxx 认证/鉴权错误   3xxx 资源不存在   4xxx 下游服务调用失败   5xxx 内部错误
"""

from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    SUCCESS = 0

    # 1xxx —— 参数/请求错误
    PARAM_ERROR = 1001
    QUERY_TOO_LONG = 1002
    INVALID_TOP_K = 1003

    # 2xxx —— 认证/鉴权错误
    AUTH_FAILED = 2001
    AUTH_EXPIRED = 2002

    # 3xxx —— 资源不存在
    DOC_NOT_FOUND = 3001
    SESSION_NOT_FOUND = 3002

    # 4xxx —— 下游服务调用失败
    MODEL_CALL_FAILED = 4001
    RETRIEVAL_FAILED = 4002

    # 5xxx —— 内部错误
    INTERNAL_ERROR = 5000
    SERVICE_NOT_READY = 5001


# 错误码 -> 默认提示文案（可在抛出异常时用具体信息覆盖）
ERROR_MESSAGES: dict[int, str] = {
    ErrorCode.SUCCESS: "success",
    ErrorCode.PARAM_ERROR: "请求参数错误",
    ErrorCode.QUERY_TOO_LONG: "查询内容过长",
    ErrorCode.INVALID_TOP_K: "top_k 超出允许范围",
    ErrorCode.AUTH_FAILED: "认证失败",
    ErrorCode.AUTH_EXPIRED: "认证已过期",
    ErrorCode.DOC_NOT_FOUND: "文档不存在",
    ErrorCode.SESSION_NOT_FOUND: "会话不存在或已过期",
    ErrorCode.MODEL_CALL_FAILED: "模型调用失败",
    ErrorCode.RETRIEVAL_FAILED: "检索服务调用失败",
    ErrorCode.INTERNAL_ERROR: "服务内部错误",
    ErrorCode.SERVICE_NOT_READY: "服务尚未就绪，请稍后重试",
}

# 错误码 -> 对应的 HTTP 状态码
ERROR_HTTP_STATUS: dict[int, int] = {
    ErrorCode.PARAM_ERROR: 422,
    ErrorCode.QUERY_TOO_LONG: 422,
    ErrorCode.INVALID_TOP_K: 422,
    ErrorCode.AUTH_FAILED: 401,
    ErrorCode.AUTH_EXPIRED: 401,
    ErrorCode.DOC_NOT_FOUND: 404,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.MODEL_CALL_FAILED: 502,
    ErrorCode.RETRIEVAL_FAILED: 502,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.SERVICE_NOT_READY: 503,
}
