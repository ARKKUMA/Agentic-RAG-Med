"""
context_assembler.py — 上下文组装器
将 RetrievalPipeline 的检索结果转换为可直接注入 LLM 提示词的上下文文本：
去重（Jaccard 相似性）-> 按相关性排序并兼顾来源多样性 -> 按 token 预算截断
（在句子边界收尾）-> 附加统计元数据。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

DEFAULT_TOKENIZER = "cl100k_base"
DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_MAX_PER_SOURCE = 2       # 单个来源（doc_id）最多优先选入的 chunk 数
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # Jaccard 相似度阈值，超过视为重复


@dataclass
class DocumentChunk:
    text: str
    metadata: dict[str, Any]
    relevance_score: float
    source: str        # 来源文档标识（doc_id/pmc_id），用于多样性去重
    chunk_id: str


class ContextAssembler:
    """
    上下文组装器：把多路检索 + 重排序后的候选列表整理为注入 LLM 的上下文字符串。
    """

    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_per_source: int = DEFAULT_MAX_PER_SOURCE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        log: logging.Logger | None = None,
    ):
        self.encoding = tiktoken.get_encoding(tokenizer_name)
        self.max_context_tokens = max_context_tokens
        self.max_per_source = max_per_source
        self.similarity_threshold = similarity_threshold
        self.log = log or logging.getLogger("context_assembler")

    # ── token 估算 ────────────────────────────────────────────────
    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    # ── 格式转换：检索结果 dict -> DocumentChunk ──────────────────
    @staticmethod
    def _to_document_chunks(retrieved_docs: list[dict]) -> list[DocumentChunk]:
        chunks = []
        for d in retrieved_docs:
            meta = d.get("metadata", {}) or {}
            score = d.get("final_score", d.get("fused_score", 0.0)) or 0.0
            source = meta.get("doc_id") or meta.get("pmc_id") or (d.get("chunk_id", "") or "")[:8]
            chunks.append(DocumentChunk(
                text=d.get("text", "") or "",
                metadata=meta,
                relevance_score=float(score),
                source=str(source),
                chunk_id=d.get("chunk_id", ""),
            ))
        return chunks

    # ── Jaccard 相似性去重 ────────────────────────────────────────
    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        set_a = set(re.findall(r"\w+", a.lower()))
        set_b = set(re.findall(r"\w+", b.lower()))
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _deduplicate(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """按相关性降序遍历，剔除与已保留 chunk 高度相似（Jaccard ≥ threshold）的重复内容。"""
        ordered = sorted(chunks, key=lambda c: -c.relevance_score)
        unique: list[DocumentChunk] = []
        for c in ordered:
            is_dup = any(
                self._jaccard_similarity(c.text, u.text) >= self.similarity_threshold
                for u in unique
            )
            if not is_dup:
                unique.append(c)
        return unique

    # ── 相关性排序 + 来源多样性 ───────────────────────────────────
    def _select_with_diversity(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        按相关性降序排列；若某来源（doc_id）已达到 max_per_source，
        其后续 chunk 不丢弃，而是移到队列末尾（降低优先级，为其它来源让路）。
        """
        ordered = sorted(chunks, key=lambda c: -c.relevance_score)
        source_counts: dict[str, int] = {}
        primary, deprioritized = [], []
        for c in ordered:
            cnt = source_counts.get(c.source, 0)
            if cnt < self.max_per_source:
                primary.append(c)
            else:
                deprioritized.append(c)
            source_counts[c.source] = cnt + 1
        return primary + deprioritized

    # ── 句子边界截断 ──────────────────────────────────────────────
    @staticmethod
    def _truncate_at_boundary(text: str, tail_window_ratio: float = 0.1) -> str:
        """在文本末尾 tail_window_ratio 区间内寻找最后一个句号，避免从句子中间截断。"""
        if not text:
            return text
        window_start = int(len(text) * (1 - tail_window_ratio))
        tail = text[window_start:]
        cut = max(tail.rfind("。"), tail.rfind(". "), tail.rfind(".\n"))
        if cut == -1:
            return text  # 找不到句号边界，宁可保留完整文本也不做生硬截断
        cut_at = window_start + cut + 1
        return text[:cut_at].rstrip()

    # ── 来源统计 ──────────────────────────────────────────────────
    @staticmethod
    def _analyze_sources(chunks: list[DocumentChunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    # ── 主入口 ────────────────────────────────────────────────────
    def assemble(self, retrieved_docs: list[dict], query: str = "") -> dict:
        """
        组装最终注入 LLM 的上下文。

        Args:
            retrieved_docs: RetrievalPipeline.retrieve()["results"] 或等价的候选列表
            query: 原始查询（预留，供未来查询相关的组装策略使用）

        Returns:
            {"context_text": str, "metadata": dict, "selected_chunks": list[DocumentChunk]}
        """
        total_retrieved = len(retrieved_docs)
        all_chunks = self._to_document_chunks(retrieved_docs)
        unique_chunks = self._deduplicate(all_chunks)
        diversified = self._select_with_diversity(unique_chunks)

        selected: list[DocumentChunk] = []
        parts: list[str] = []
        used_tokens = 0

        for c in diversified:
            idx = len(selected) + 1
            # 保留原始引用编号（即使后续证据评估阶段会剔除部分来源，正文与参考文献列表的编号也不能错位）
            c.metadata = {**c.metadata, "_citation_id": idx}
            header = (
                f"[来源 {idx} | {c.metadata.get('journal', '?')} "
                f"{c.metadata.get('pub_year', '?')} | "
                f"{c.metadata.get('section_title') or c.metadata.get('imrad_type', '?')}]\n"
            )
            block = header + c.text.strip() + "\n"
            block_tokens = self.estimate_tokens(block)

            if used_tokens + block_tokens > self.max_context_tokens:
                remaining = self.max_context_tokens - used_tokens
                if remaining > 50:  # 剩余预算太小则放弃，不塞入无意义的碎片
                    approx_chars = max(0, int(len(block) * (remaining / block_tokens)))
                    truncated = self._truncate_at_boundary(block[:approx_chars])
                    if truncated.strip():
                        parts.append(truncated)
                        selected.append(c)
                        used_tokens += self.estimate_tokens(truncated)
                break

            parts.append(block)
            selected.append(c)
            used_tokens += block_tokens

        final_context = "\n".join(parts).strip()

        context_metadata = {
            "total_chunks_retrieved": total_retrieved,
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected),
        }

        return {
            "context_text": final_context,
            "metadata": context_metadata,
            "selected_chunks": selected,
        }
