"""
main.py — FastAPI 应用入口
应用骨架：统一响应格式 / 错误码 / 全局异常处理 / 请求日志 / 健康检查。
RAG 流水线（BGE embedder + ChromaDB + BM25 + reranker + Ollama）在 lifespan
启动阶段构建一次，挂在 app.state 上供各请求复用，避免每次请求重新加载模型。

运行：
    $env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000

环境变量（均有默认值，默认指向小规模 test_dir_mode 集合，避免误跑全量 pmc_full
导致的内存问题——详见 README"已知限制"一节）：
    RAG_DB_DIR       ChromaDB 持久化目录
    RAG_COLLECTION   集合名称
    RAG_BM25_CACHE   BM25 索引 pickle 缓存路径
    RAG_LLM_MODEL    Ollama 模型名称
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from generation import GenerationCache, LLMGenerator, MedicalGenerationPipeline
from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline

from .exceptions import register_exception_handlers
from .middleware import RequestLoggingMiddleware
from .models import ResponseModel
from .routers import qa
from .session import SessionManager

# ── 配置 ──────────────────────────────────────────────────────────
DB_DIR = os.environ.get("RAG_DB_DIR", r"d:\Rag-Med\pipeline_output\chroma_db")
COLLECTION = os.environ.get("RAG_COLLECTION", "test_dir_mode")
BM25_CACHE = os.environ.get("RAG_BM25_CACHE", r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b-instruct")

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
        vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)

        bm25_index = (
            BM25Index.load(BM25_CACHE, log=log)
            if os.path.exists(BM25_CACHE)
            else BM25Index(log=log).build_from_chroma(vector_index.collection)
        )

        retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)

        gen_cache = GenerationCache(log=log)
        llm = LLMGenerator(model_name=LLM_MODEL, log=log, cache=gen_cache)

        generation_pipeline = MedicalGenerationPipeline(
            retrieval_pipeline=retrieval_pipeline, llm=llm, log=log, log_path=GENERATION_LOG_PATH,
        )

        app.state.generation_pipeline = generation_pipeline
        app.state.session_manager = SessionManager()
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
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.add_middleware(RequestLoggingMiddleware, log_path=REQUEST_LOG_PATH, logger=logging.getLogger("api.requests"))


@app.get("/health", response_model=ResponseModel[dict])
async def health_check():
    """
    健康检查：liveness（进程是否存活，能响应即代表存活）+ readiness
    （RAG 流水线是否已加载完成，未就绪时返回 ready=False 但不报错，
    供负载均衡器/编排系统区分"正在启动"与"已崩溃"）。
    """
    ready = getattr(app.state, "ready", False)
    return ResponseModel.ok(data={"status": "ok", "ready": ready})


app.include_router(qa.router)
