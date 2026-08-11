"""
main.py — FastAPI 应用入口
应用骨架：统一响应格式 / 错误码 / 全局异常处理 / 请求日志 / 健康检查。
RAG 流水线（BGE embedder + ChromaDB + BM25 + reranker + Ollama）、会话管理器、
统计器、健康检查器、文档索引均在 lifespan 启动阶段构建一次，挂在 app.state 上
供各请求复用。

运行：
    $env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000

配置通过 .env 文件管理（见 config.py / .env.example），环境变量优先级更高。
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from agent import ToolDispatcherEngine, ToolRegistry, build_agent_graph, register_retrieval_tool
from agent.memory import AgentMemory
from generation import GenerationCache, LLMGenerator, MedicalGenerationPipeline
from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline

from .config import settings
from .document_index import DocumentIndex
from .exceptions import register_exception_handlers
from .health_checker import HealthChecker
from .middleware import RequestLoggingMiddleware
from .models import ResponseModel
from .routers import documents, qa, sessions, stats
from .session import SessionManager
from .stats import StatsTracker

LOG_DIR = Path("d:/Rag-Med/logs")
API_LOG_PATH = LOG_DIR / "api_service.log"
REQUEST_LOG_PATH = LOG_DIR / "api_requests.jsonl"
GENERATION_LOG_PATH = LOG_DIR / "api_generation.jsonl"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(API_LOG_PATH, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


log = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    log.info("正在初始化 RAG 流水线（首次加载模型需要数秒）…")
    t0 = time.time()
    try:
        embedder = BGEEmbedder(device="cpu", log=log)
        vector_index = PMCVectorIndex(
            db_dir=settings.rag_db_dir, collection_name=settings.rag_collection, embedder=embedder, log=log,
        )

        bm25_index = (
            BM25Index.load(settings.rag_bm25_cache, log=log)
            if Path(settings.rag_bm25_cache).exists()
            else BM25Index(log=log).build_from_chroma(vector_index.collection)
        )

        retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)

        gen_cache = GenerationCache(log=log)
        llm = LLMGenerator(
            model_name=settings.rag_llm_model, base_url=settings.rag_llm_base_url, log=log, cache=gen_cache,
        )

        generation_pipeline = MedicalGenerationPipeline(
            retrieval_pipeline=retrieval_pipeline, llm=llm, log=log, log_path=GENERATION_LOG_PATH,
        )

        log.info("正在构建文档级索引（扫描一次 ChromaDB 集合）…")
        document_index = DocumentIndex(log=log)
        document_index.build_from_collection(vector_index.collection)

        session_manager = SessionManager(
            ttl_seconds=settings.session_ttl_seconds, max_turns=settings.session_max_turns,
        )

        # ── Agent 状态机（第 2 周新增）───────────────────────────────
        # 复用同一个 retrieval_pipeline / llm 实例，不重复加载模型；
        # AgentMemory 连接不上 Redis 时只记 warning、自动降级为直通，不阻塞启动。
        log.info("正在组装 Agent 状态机（复用已加载的检索/生成组件）…")
        agent_registry = ToolRegistry(log=log)
        register_retrieval_tool(agent_registry, retrieval_pipeline)
        agent_dispatcher = ToolDispatcherEngine(agent_registry, log=log)
        agent_memory = AgentMemory(session_manager, redis_url=settings.agent_redis_url, log=log)
        agent_graph = build_agent_graph(
            retrieval_dispatcher=agent_dispatcher,
            llm=llm,
            session_context_fn=session_manager.build_context_prefix,
            memory=agent_memory,
        )

        app.state.generation_pipeline = generation_pipeline
        app.state.session_manager = session_manager
        app.state.stats_tracker = StatsTracker()
        app.state.health_checker = HealthChecker(
            llm_base_url=settings.rag_llm_base_url, vector_index=vector_index, bm25_index=bm25_index, log=log,
        )
        app.state.document_index = document_index
        app.state.agent_graph = agent_graph
        app.state.agent_memory = agent_memory
        app.state.ready = True
        log.info(f"RAG 流水线初始化完成，耗时 {time.time() - t0:.1f}s，服务就绪")
    except Exception as e:
        log.error(f"RAG 流水线初始化失败：{e}", exc_info=e)
        raise

    yield

    log.info("服务正在关闭")


app = FastAPI(
    title="PMC Medical RAG API",
    description="基于 PMC oa_comm 医学文献的检索增强生成服务",
    version="0.2.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.add_middleware(RequestLoggingMiddleware, log_path=REQUEST_LOG_PATH, logger=logging.getLogger("api.requests"))


@app.get("/health", response_model=ResponseModel[dict], tags=["health"])
async def health_check():
    """
    健康检查：liveness（进程是否存活，能响应即代表存活）+ readiness
    （RAG 流水线是否已加载完成，未就绪时返回 ready=False 但不报错，
    供负载均衡器/编排系统区分"正在启动"与"已崩溃"）。
    """
    ready = getattr(app.state, "ready", False)
    return ResponseModel.ok(data={"status": "ok", "ready": ready})


app.include_router(qa.router)
app.include_router(sessions.router)
app.include_router(stats.router)
app.include_router(documents.router)
