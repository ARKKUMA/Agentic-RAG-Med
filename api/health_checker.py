"""
health_checker.py — 各组件健康状态检查
覆盖任务书要求的三类组件：LLM（Ollama）/ 向量库（ChromaDB）/ 数据库
（本系统没有独立关系型数据库，"数据库"角色由 BM25 关键词索引承担——
会话存储是进程内内存，不是持久化数据库，在 detail 字段中如实说明）。
"""

from __future__ import annotations

import logging
import time

import requests


class HealthChecker:
    def __init__(self, llm_base_url: str, vector_index, bm25_index, log: logging.Logger | None = None):
        self.llm_base_url = llm_base_url.rstrip("/")
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.log = log or logging.getLogger("health_checker")

    def check_llm(self) -> dict:
        t0 = time.time()
        try:
            resp = requests.get(f"{self.llm_base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            n_models = len(resp.json().get("models", []))
            return {
                "name": "llm", "status": "ok",
                "detail": f"Ollama 可达，已拉取模型数={n_models}",
                "latency_seconds": round(time.time() - t0, 4),
            }
        except Exception as e:
            return {
                "name": "llm", "status": "down", "detail": str(e),
                "latency_seconds": round(time.time() - t0, 4),
            }

    def check_vector_store(self) -> dict:
        t0 = time.time()
        try:
            count = self.vector_index.collection.count()
            status = "ok" if count > 0 else "degraded"
            return {
                "name": "vector_store", "status": status,
                "detail": f"ChromaDB 集合 {self.vector_index.collection_name} 向量数={count:,}",
                "latency_seconds": round(time.time() - t0, 4),
            }
        except Exception as e:
            return {
                "name": "vector_store", "status": "down", "detail": str(e),
                "latency_seconds": round(time.time() - t0, 4),
            }

    def check_database(self) -> dict:
        """本系统的"数据库"角色：BM25 关键词索引。会话数据为进程内存，非持久化。"""
        t0 = time.time()
        try:
            n = len(self.bm25_index)
            status = "ok" if n > 0 else "degraded"
            return {
                "name": "database", "status": status,
                "detail": f"BM25 关键词索引文档数={n:,}（会话存储为进程内内存，非持久化数据库）",
                "latency_seconds": round(time.time() - t0, 4),
            }
        except Exception as e:
            return {
                "name": "database", "status": "down", "detail": str(e),
                "latency_seconds": round(time.time() - t0, 4),
            }

    def check_all(self) -> list[dict]:
        return [self.check_llm(), self.check_vector_store(), self.check_database()]
