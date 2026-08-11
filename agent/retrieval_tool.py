"""
agent/retrieval_tool.py — 检索工具封装
把 retrieval.pipeline.RetrievalPipeline.retrieve() 包装成符合 ToolRegistry
规范的工具：参数用 pydantic 模型声明，handler 是薄封装，已知的瞬时性异常
（连接失败/超时）重新抛出为 RetryableError 供 ToolDispatcherEngine 识别，
其它异常保持原样（调度引擎按不可重试处理）。

这是 AGENT_ARCHITECTURE.md"工具抽象"设计落地的第一个真实工具——检索不再是
流水线里硬编码的一步，而是注册进 ToolRegistry 的普通工具，未来新增工具
（术语查询、单位换算等）按同样的模式接入即可，不需要改动 Agent 主循环。
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .tool_dispatcher import RetryableError
from .tool_registry import ToolRegistry, ToolSpec

log = logging.getLogger("agent.retrieval_tool")


class RetrievalToolParams(BaseModel):
    query: str = Field(..., min_length=1, description="检索查询文本")
    top_k: int = Field(default=8, ge=1, le=20)
    fusion_strategy: Literal["rrf", "weighted", "simple"] = "rrf"
    where_filter: dict | None = Field(
        default=None, description="显式 ChromaDB 元数据过滤条件（第 2 周新增，透传给 RetrievalPipeline.retrieve）",
    )


def register_retrieval_tool(registry: ToolRegistry, retrieval_pipeline, max_retries: int = 2) -> None:
    """把已构建好的 RetrievalPipeline 实例注册为名为 "retrieval" 的工具。"""

    def handler(
        query: str, top_k: int = 8, fusion_strategy: str = "rrf", where_filter: dict | None = None,
    ) -> dict:
        try:
            return retrieval_pipeline.retrieve(
                query, top_k=top_k, fusion_strategy=fusion_strategy, where_filter=where_filter,
            )
        except (ConnectionError, TimeoutError) as e:
            # 向量库连接失败/超时——归类为可重试
            raise RetryableError(f"检索工具连接异常：{e}") from e

    registry.register(ToolSpec(
        name="retrieval",
        handler=handler,
        param_schema=RetrievalToolParams,
        description="基于 PMC 文献语料的多路检索（向量 + BM25 融合 + 重排序）",
        max_retries=max_retries,
    ))
