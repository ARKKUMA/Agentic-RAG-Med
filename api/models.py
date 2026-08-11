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
    run_evaluation: bool = Field(default=True, description="是否执行证据评估阶段（仅 agent_mode=False 时生效）")
    run_review: bool = Field(default=True, description="是否执行批判性审查阶段（仅 agent_mode=False 时生效）")
    agent_mode: bool = Field(
        default=False,
        description="false（默认）：直通原有单轮 RAG 流水线，行为与响应结构完全不变；"
                     "true：进入 Agent 执行链路（LangGraph 状态机），响应额外带 execution_trace 字段",
    )
    where_filter: dict | None = Field(
        default=None,
        description="显式 ChromaDB 元数据过滤条件（仅 agent_mode=True 时生效）；"
                     "不传时两种模式都会退化为按查询文本自动提取的过滤条件（如提及的年份范围）",
    )


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
    # agent_mode=True 时本周未执行格式校验阶段（这是明确未完成的工作，不是"默认通过"）——
    # 用 None 如实表示"未评估"，而不是伪造一个 True；RAG 模式（agent_mode=False）行为、
    # 取值范围（True/False，从不为 None）与升级前完全一致，原有调用方不受影响。
    format_check_pass: bool | None
    agent_mode: bool = False
    execution_trace: list[dict] | None = Field(
        default=None, description="仅 agent_mode=True 时非空：LangGraph 各节点的执行轨迹（步骤/耗时/工具调用详情等）",
    )


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
