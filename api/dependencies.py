"""
dependencies.py — FastAPI 依赖注入
RAG 流水线（含 BGE 模型、ChromaDB、reranker、Ollama 连接）在 app 启动时
（见 main.py 的 lifespan）构建一次并挂在 app.state 上，避免每次请求重新加载模型。
"""

from __future__ import annotations

from fastapi import Request

from generation import MedicalGenerationPipeline

from .exceptions import ServiceNotReadyError
from .session import SessionManager


def get_generation_pipeline(request: Request) -> MedicalGenerationPipeline:
    pipeline = getattr(request.app.state, "generation_pipeline", None)
    if pipeline is None or not getattr(request.app.state, "ready", False):
        raise ServiceNotReadyError("RAG 流水线尚未初始化完成")
    return pipeline


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager
