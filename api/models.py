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
    created_at: float
    last_active: float
    turn_count: int
    turns: list[SessionHistoryTurn]


# ══════════════════════════════════════════════════════════════════
# 会话管理接口
# ══════════════════════════════════════════════════════════════════

class SessionCreateData(BaseModel):
    session_id: str
    created_at: float


class SessionDeleteData(BaseModel):
    session_id: str
    deleted: bool


# ══════════════════════════════════════════════════════════════════
# 运营统计接口
# ══════════════════════════════════════════════════════════════════

class QAStatsData(BaseModel):
    total_calls: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_latency_seconds: float


class CorpusStatsData(BaseModel):
    total_documents: int
    total_chunks: int
    index_size_bytes: int | None = None
    incremental_update_count: int


class ComponentHealthItem(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None
    latency_seconds: float | None = None


class OperationalStatsData(BaseModel):
    qa: QAStatsData
    corpus: CorpusStatsData
    components: list[ComponentHealthItem]
    active_sessions: int


# ══════════════════════════════════════════════════════════════════
# 文档管理接口
# ══════════════════════════════════════════════════════════════════

class DocumentIn(BaseModel):
    """文档写入模型（供未来文档录入/更新接口使用；当前 Part 2 仅开放只读查询）。"""

    doc_id: str = Field(..., description="文档唯一标识，对应 PMC ID")
    title: str = Field(..., description="文章标题")
    abstract: str | None = Field(default=None, description="摘要")
    journal: str | None = Field(default=None, description="期刊名称")
    pub_date: str | None = Field(default=None, description="发表日期/年份")
    pmid: str | None = Field(default=None, description="PubMed ID")
    doi: str | None = Field(default=None, description="DOI")
    article_type: str | None = Field(default=None, description="文章类型")


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    abstract: str | None = None
    journal: str | None = None
    pub_date: str | None = None
    pmid: str | None = None
    doi: str | None = None
    article_type: str | None = None
    chunk_count: int
