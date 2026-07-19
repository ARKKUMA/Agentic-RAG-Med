"""
bm25_index.py — BM25 关键词检索索引
基于 rank_bm25.BM25Okapi，为 PMC chunk 语料构建/加载倒排索引，
用于 MultiPathRetriever 的关键词检索路径（擅长专有名词/缩写/精确术语，
弥补向量检索对罕见术语召回不足的问题）。
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# 英文停用词（医学文献高频但无检索区分度的词）
STOPWORDS: set[str] = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those", "with",
    "by", "from", "as", "it", "its", "we", "our", "their", "which", "study", "studies",
    "results", "result", "using", "used", "between", "among", "also", "than", "then",
    "not", "no", "yes", "have", "has", "had", "such", "can", "may", "might", "one",
    "two", "during", "after", "before", "into", "over", "significant", "significantly",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")


def tokenize(text: str) -> list[str]:
    """英文分词：小写化 + 正则提取 token + 停用词过滤。语料与查询须用同一分词函数。"""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """
    BM25 关键词检索索引封装。

    注意：BM25Okapi 将全部语料分词结果保留在内存中，适合原型/中小规模集合
    （当前测试集合 ~1854 chunks）。若扩展到百万级全量语料，需改用
    Elasticsearch / Whoosh 等磁盘倒排索引，本类接口可保持不变。
    """

    def __init__(self, log: logging.Logger | None = None):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.log = log or logging.getLogger("bm25_index")

    # ── 构建 ──────────────────────────────────────────────────────
    def build_from_records(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> "BM25Index":
        self.chunk_ids = list(chunk_ids)
        self.documents = list(documents)
        self.metadatas = list(metadatas) if metadatas is not None else [{} for _ in self.chunk_ids]

        self.log.info(f"BM25: 分词 {len(self.documents):,} 条文档…")
        tokenized_corpus = [tokenize(doc) for doc in self.documents]

        self.log.info("BM25: 构建索引…")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.log.info(f"BM25: 索引构建完成，文档数={len(self.chunk_ids):,}")
        return self

    def build_from_chroma(self, collection, page_size: int = 5000) -> "BM25Index":
        """从 ChromaDB collection 分页拉取全部文档 + 元数据，构建 BM25 索引。"""
        total = collection.count()
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids.extend(batch["ids"])
            docs.extend(batch["documents"])
            metas.extend(batch["metadatas"])
            offset += page_size
        return self.build_from_records(ids, docs, metas)

    def build_from_dataframe(
        self,
        df,
        text_col: str = "text",
        id_col: str = "chunk_id",
    ) -> "BM25Index":
        meta_cols = [c for c in df.columns if c not in (text_col, id_col)]
        metadatas = df[meta_cols].to_dict("records")
        return self.build_from_records(df[id_col].tolist(), df[text_col].tolist(), metadatas)

    # ── 持久化（避免每次重新分词构建）──────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunk_ids": self.chunk_ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                },
                f,
            )
        self.log.info(f"BM25 索引已保存：{path}")

    @classmethod
    def load(cls, path: str | Path, log: logging.Logger | None = None) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(log=log)
        idx.bm25 = data["bm25"]
        idx.chunk_ids = data["chunk_ids"]
        idx.documents = data["documents"]
        idx.metadatas = data["metadatas"]
        return idx

    # ── 查询 ──────────────────────────────────────────────────────
    def search(self, query_text: str, top_k: int = 20) -> list[dict]:
        """
        返回 [{rank, chunk_id, bm25_score, text, text_preview, metadata}, ...]，
        按 bm25_score 降序，过滤掉 0 分（无词汇重叠）的结果。
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 索引尚未构建，请先调用 build_from_* 方法")

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            text = self.documents[idx] or ""
            results.append(
                {
                    "rank": rank,
                    "chunk_id": self.chunk_ids[idx],
                    "bm25_score": round(score, 6),
                    "text": text,
                    "text_preview": text[:200],
                    "metadata": self.metadatas[idx],
                }
            )
        return results

    def __len__(self) -> int:
        return len(self.chunk_ids)
