"""
models.py — 统一响应模型、分页模型、问答接口请求/响应 schema
"""

from __future__ import annotations

import time
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    """所有接口的统一响应外壳。code=0 表示成功，非 0 对应 error_codes.ErrorCode。"""

    code: int = 0
    message: str = "success"
    data: T | None = None
    request_id: str | None = None
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    @classmethod
    def ok(cls, data: T | None = None, request_id: str | None = None) -> "ResponseModel[T]":
        return cls(code=0, message="success", data=data, request_id=request_id)


class PageInfo(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class PaginatedData(BaseModel, Generic[T]):
    items: list[T]
    page_info: PageInfo


# ══════════════════════════════════════════════════════════════════
# 问答接口
# ══════════════════════════════════════════════════════════════════

class QARequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题，非空，最长 2000 字符")
    top_k: int = Field(default=8, ge=1, le=20, description="最终返回结果数量，1-20")
    fusion_strategy: Literal["rrf", "weighted", "simple"] = Field(default="rrf", description="多路检索融合策略")
    session_id: str | None = Field(default=None, description="会话 ID；传入则关联历史对话，不传则新建/不关联")
    run_evaluation: bool = Field(default=True, description="是否执行证据评估阶段")
    run_review: bool = Field(default=True, description="是否执行批判性审查阶段")


class SourceItem(BaseModel):
    rank: int
    chunk_id: str
    journal: str | None = None
    pub_year: int | None = None
    pmc_id: str | None = None
    relevance_score: float


class QAResponseData(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str | None = None
    total_time_seconds: float
    citation_retry_attempts: int
    format_check_pass: bool


class SessionHistoryTurn(BaseModel):
    query: str
    answer: str
    timestamp: float


class SessionHistoryData(BaseModel):
    session_id: str
    turns: list[SessionHistoryTurn]
