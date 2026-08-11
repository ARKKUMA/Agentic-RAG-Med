"""
dependencies.py — FastAPI 依赖注入
所有单例（RAG 流水线、会话管理器、统计器、健康检查器、文档索引、第 2 周新增的
Agent 状态机 + AgentMemory）都在 app 启动时（见 main.py 的 lifespan）构建一次
并挂在 app.state 上，避免每次请求重新加载模型或重新扫描语料——Agent 状态机复用
与 RAG 流水线完全相同的 RetrievalPipeline/LLMGenerator 实例，不重复加载
BGE/ChromaDB/BM25/reranker/Ollama 连接。
"""

from __future__ import annotations

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from agent.memory import AgentMemory
from generation import MedicalGenerationPipeline

from .document_index import DocumentIndex
from .exceptions import ServiceNotReadyError
from .health_checker import HealthChecker
from .session import SessionManager
from .stats import StatsTracker


def get_generation_pipeline(request: Request) -> MedicalGenerationPipeline:
    pipeline = getattr(request.app.state, "generation_pipeline", None)
    if pipeline is None or not getattr(request.app.state, "ready", False):
        raise ServiceNotReadyError("RAG 流水线尚未初始化完成")
    return pipeline


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_stats_tracker(request: Request) -> StatsTracker:
    return request.app.state.stats_tracker


def get_health_checker(request: Request) -> HealthChecker:
    return request.app.state.health_checker


def get_document_index(request: Request) -> DocumentIndex:
    return request.app.state.document_index


def get_agent_graph(request: Request) -> CompiledStateGraph:
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None or not getattr(request.app.state, "ready", False):
        raise ServiceNotReadyError("Agent 状态机尚未初始化完成")
    return graph


def get_agent_memory(request: Request) -> AgentMemory:
    return request.app.state.agent_memory
