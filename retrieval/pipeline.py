"""
pipeline.py — 完整检索流水线
查询理解与增强 -> 多路检索（向量 + BM25 融合）-> 交叉编码器多准则重排序
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

from .multipath_retriever import MultiPathRetriever
from .query_processor import MedicalQueryProcessor
from .reranker import DEFAULT_CRITERIA_WEIGHTS, MedicalReranker


class RetrievalPipeline:
    """
    端到端检索流水线：

      1. MedicalQueryProcessor — 查询理解与增强（缩写展开/同义词扩展/实体识别/过滤条件）
      2. MultiPathRetriever    — 向量检索 + BM25 关键词检索，融合召回
      3. MedicalReranker       — 交叉编码器多准则重排序（相关性/时效性/权威性）
    """

    def __init__(
        self,
        vector_index,
        bm25_index,
        query_processor: MedicalQueryProcessor | None = None,
        reranker: MedicalReranker | None = None,
        log: logging.Logger | None = None,
        log_path: str | Path | None = None,
    ):
        self.log = log or logging.getLogger("retrieval_pipeline")
        self.vector_index = vector_index
        self.query_processor = query_processor or MedicalQueryProcessor()
        self.retriever = MultiPathRetriever(vector_index, bm25_index, log=self.log)
        self.reranker = reranker or MedicalReranker(log=self.log)

        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── JSONL 运行日志（每次 retrieve() 追加一行，记录完整可追溯信息）──
    def _log_run(self, query: str, fusion_strategy: str, top_k: int, out: dict) -> None:
        if not self.log_path:
            return
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "fusion_strategy": fusion_strategy,
            "top_k": top_k,
            "query_info": {
                "cleaned": out["query_info"].cleaned,
                "entities": out["query_info"].entities,
                "abbreviations": out["query_info"].abbreviations,
                "filters": out["query_info"].filters,
                "imrad_hint": out["query_info"].imrad_hint,
            },
            "fused_candidates": out["fused_candidates"],
            "results": [
                {
                    "final_rank": r["final_rank"],
                    "chunk_id": r["chunk_id"],
                    "final_score": r["final_score"],
                    "relevance_score": r["relevance_score"],
                    "recency_score": r["recency_score"],
                    "authority_score": r["authority_score"],
                    "fused_score": r["fused_score"],
                    "sources": r["sources"],
                    "journal": r["metadata"].get("journal"),
                    "pub_year": r["metadata"].get("pub_year"),
                    "text_preview": (r.get("text") or "")[:200],
                }
                for r in out["results"]
            ],
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 融合结果的 text 字段可能是截断预览，重排序前补回完整正文 ──
    def _fetch_full_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        if not chunk_ids:
            return {}
        got = self.vector_index.collection.get(ids=chunk_ids, include=["documents"])
        return dict(zip(got["ids"], got["documents"]))

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_k_vector: int = 20,
        top_k_keyword: int = 20,
        fusion_strategy: str = "rrf",
        rerank_weights: dict | None = None,
        where_filter: dict | None = None,
    ) -> dict:
        """
        执行完整检索流水线。

        Args:
            query: 用户自然语言查询（中/英文均可）
            top_k: 最终返回结果数量
            top_k_vector / top_k_keyword: 两路召回各自的返回数量
            fusion_strategy: 'rrf' / 'weighted' / 'simple'
            rerank_weights: 重排序多准则权重，默认 DEFAULT_CRITERIA_WEIGHTS
            where_filter: 显式 ChromaDB 过滤条件；缺省时用查询处理器提取的过滤条件

        Returns:
            {
              "query_info": ProcessedQuery,
              "fused_candidates": int,   # 融合后的候选总数
              "results": [ {chunk_id, text, metadata, sources,
                            vector_score, keyword_score, fused_score,
                            relevance_score, recency_score, authority_score,
                            final_score, final_rank}, ... ]
            }
        """
        query_info = self.query_processor.process(query)

        fused = self.retriever.retrieve(
            query_info,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            fusion_strategy=fusion_strategy,
            where_filter=where_filter,
        )

        full_texts = self._fetch_full_texts([c["chunk_id"] for c in fused])
        for c in fused:
            c["text"] = full_texts.get(c["chunk_id"], c["text"])

        reranked = self.reranker.rerank(
            query_info.cleaned or query_info.original,
            fused,
            top_k=top_k,
            criteria_weights=rerank_weights or DEFAULT_CRITERIA_WEIGHTS,
        )

        out = {
            "query_info": query_info,
            "fused_candidates": len(fused),
            "results": reranked,
        }
        self._log_run(query, fusion_strategy, top_k, out)
        return out
