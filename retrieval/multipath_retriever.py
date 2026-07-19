"""
multipath_retriever.py — 多路检索器（MultiPathRetriever）
向量检索（语义，ChromaDB + BGE）与 BM25 关键词检索融合，提升检索召回与排序质量。

融合策略：
  - simple  : 简单合并去重（基本方法，忽略排名信息，仅按命中路数排序）
  - rrf     : Reciprocal Rank Fusion，score = Σ 1/(k + rank)，学术检索常用，
              不依赖不同检索路径分数量纲是否可比
  - weighted: 加权融合，向量检索权重更高（语义检索通常更贴合医学查询意图）
"""

from __future__ import annotations

import logging

from .bm25_index import BM25Index
from .query_processor import ProcessedQuery

RRF_K = 60  # RRF 融合常数（学术检索常用默认值，越大排名差异的影响越平滑）
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3


def _new_record(chunk_id: str, text: str, metadata: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata,
        "sources": [],
        "vector_score": None,
        "vector_rank": None,
        "keyword_score": None,
        "keyword_rank": None,
    }


class MultiPathRetriever:
    """
    融合向量检索（语义相似）与 BM25 检索（关键词精确匹配）的多路召回器。

    向量检索路径：擅长语义相似、同义表达；对罕见专有名词/缩写召回较弱。
    关键词检索路径：擅长精确术语/缩写/专有名词匹配；不理解语义相似。
    二者互补，融合后综合提升召回率与排序质量。
    """

    def __init__(
        self,
        vector_index,
        bm25_index: BM25Index,
        log: logging.Logger | None = None,
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.log = log or logging.getLogger("multipath_retriever")

    # ── 向量检索 ──────────────────────────────────────────────────
    def _vector_search(
        self,
        query_info: ProcessedQuery,
        top_k: int,
        where_filter: dict | None,
    ) -> list[dict]:
        # 注意：query_info.vector_query 已带 BGE 指令前缀，而 PMCVectorIndex.query()
        # 内部会再次添加自己的前缀。若直接传 vector_query 会造成双重前缀，
        # 因此这里改用 cleaned 原文，前缀统一交给 PMCVectorIndex 处理。
        text = query_info.cleaned or query_info.original
        result = self.vector_index.query(text, n_results=top_k, where_filter=where_filter)
        return result["results"]

    # ── 关键词检索 ────────────────────────────────────────────────
    def _keyword_search(self, query_info: ProcessedQuery, top_k: int) -> list[dict]:
        return self.bm25_index.search(query_info.keyword_query, top_k=top_k)

    # ── 合并两路结果为统一记录（不做打分，供各融合策略复用）──────
    def _merge(self, vector_hits: list[dict], keyword_hits: list[dict]) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        for h in vector_hits:
            cid = h["chunk_id"]
            rec = merged.setdefault(cid, _new_record(cid, h.get("text_preview", ""), h.get("metadata", {})))
            rec["sources"].append("vector")
            rec["vector_score"] = h["similarity"]
            rec["vector_rank"] = h["rank"]
        for h in keyword_hits:
            cid = h["chunk_id"]
            rec = merged.setdefault(cid, _new_record(cid, h.get("text", h.get("text_preview", "")), h.get("metadata", {})))
            rec["sources"].append("keyword")
            rec["keyword_score"] = h["bm25_score"]
            rec["keyword_rank"] = h["rank"]
            if h.get("text"):
                rec["text"] = h["text"]  # BM25 语料本身是全文，优先用它填充预览
        return merged

    # ── 融合策略：简单合并去重 ────────────────────────────────────
    @staticmethod
    def _score_simple(merged: dict[str, dict]) -> list[dict]:
        fused = list(merged.values())
        for item in fused:
            item["fused_score"] = float(len(item["sources"]))  # 命中路数即分数，忽略排名细节
        fused.sort(key=lambda x: (
            -x["fused_score"],
            -(x["vector_score"] or 0.0),
            -(x["keyword_score"] or 0.0),
        ))
        return fused

    # ── 融合策略：RRF ─────────────────────────────────────────────
    @staticmethod
    def _score_rrf(merged: dict[str, dict], k: int = RRF_K) -> list[dict]:
        fused = list(merged.values())
        for item in fused:
            score = 0.0
            if item["vector_rank"] is not None:
                score += 1.0 / (k + item["vector_rank"])
            if item["keyword_rank"] is not None:
                score += 1.0 / (k + item["keyword_rank"])
            item["fused_score"] = score
        fused.sort(key=lambda x: -x["fused_score"])
        return fused

    # ── 融合策略：加权融合 ────────────────────────────────────────
    @staticmethod
    def _score_weighted(
        merged: dict[str, dict],
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> list[dict]:
        kw_scores = [item["keyword_score"] for item in merged.values() if item["keyword_score"] is not None]
        if kw_scores:
            kw_lo, kw_hi = min(kw_scores), max(kw_scores)
            kw_span = (kw_hi - kw_lo) or 1.0
        else:
            kw_lo, kw_span = 0.0, 1.0

        fused = list(merged.values())
        for item in fused:
            v_norm = item["vector_score"] or 0.0  # 向量相似度已在 [0,1]（cosine, L2 归一化）
            if item["keyword_score"] is not None:
                k_norm = (item["keyword_score"] - kw_lo) / kw_span
            else:
                k_norm = 0.0
            item["fused_score"] = vector_weight * v_norm + keyword_weight * k_norm
        fused.sort(key=lambda x: -x["fused_score"])
        return fused

    # ── 主入口 ────────────────────────────────────────────────────
    def retrieve(
        self,
        query_info: ProcessedQuery,
        top_k_vector: int = 20,
        top_k_keyword: int = 20,
        fusion_strategy: str = "rrf",
        where_filter: dict | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        执行多路检索并融合。

        Args:
            query_info: MedicalQueryProcessor.process() 输出的查询信息
            top_k_vector: 向量检索返回数量
            top_k_keyword: 关键词检索返回数量
            fusion_strategy: 融合策略 ('rrf', 'weighted', 'simple')
            where_filter: 显式传入的 ChromaDB where 过滤条件；缺省时使用
                          query_info.filters（查询处理器从文本中提取的过滤条件）
            top_k: 融合后截断返回数量；None 表示返回全部融合结果

        Returns:
            按 fused_score 降序排列的候选列表，每项含 chunk_id / text / metadata /
            sources / vector_score / vector_rank / keyword_score / keyword_rank /
            fused_score / fused_rank
        """
        where = where_filter if where_filter is not None else (query_info.filters or None)

        vector_hits = self._vector_search(query_info, top_k_vector, where)
        keyword_hits = self._keyword_search(query_info, top_k_keyword)
        merged = self._merge(vector_hits, keyword_hits)

        if fusion_strategy == "simple":
            fused = self._score_simple(merged)
        elif fusion_strategy == "weighted":
            fused = self._score_weighted(merged)
        elif fusion_strategy == "rrf":
            fused = self._score_rrf(merged)
        else:
            raise ValueError(f"未知融合策略: {fusion_strategy!r}（可选 'rrf' / 'weighted' / 'simple'）")

        if top_k:
            fused = fused[:top_k]
        for i, item in enumerate(fused, start=1):
            item["fused_rank"] = i
        return fused
