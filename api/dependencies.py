"""
dependencies.py — FastAPI 依赖注入
所有单例（RAG 流水线、会话管理器、统计器、健康检查器、文档索引）都在
app 启动时（见 main.py 的 lifespan）构建一次并挂在 app.state 上，
避免每次请求重新加载模型或重新扫描语料。
"""

from __future__ import annotations

from fastapi import Request

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
