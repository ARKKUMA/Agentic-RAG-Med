"""
reranker.py — 交叉编码器重排序器（BAAI/bge-reranker-base）
对 MultiPathRetriever 输出的初步候选进行多准则重排序：
  - relevance（相关性，0.6）: 交叉编码器 (query, passage) 打分 — 基础要求
  - recency  （时效性，0.25）: 按发表年份指数衰减 — 医学证据时效性重要
  - authority（权威性，0.15）: 按期刊权威性权重表 — 高影响力期刊加分
最终分数 = Σ weight_i * score_i，各子分数均归一化到 [0, 1]。
"""

from __future__ import annotations

import datetime
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

DEFAULT_CRITERIA_WEIGHTS: dict[str, float] = {
    "relevance": 0.6,   # 相关性 - 基础要求
    "recency":   0.25,  # 时效性 - 提取年份，随年份指数衰减
    "authority": 0.15,  # 权威性 - 按照期刊添加不同权重
}

# 期刊权威性权重表（示例，可按需扩展）；未列出的期刊使用 DEFAULT_JOURNAL_WEIGHT
JOURNAL_WEIGHTS: dict[str, float] = {
    "nature": 1.0, "science": 1.0, "cell": 1.0,
    "new england journal of medicine": 1.0, "nejm": 1.0,
    "the lancet": 1.0, "lancet": 1.0,
    "jama": 0.95, "nature medicine": 0.95,
    "bmj": 0.9, "annals of internal medicine": 0.9,
    "plos medicine": 0.75, "circulation": 0.8,
    "scientific reports": 0.6, "plos one": 0.55,
}
DEFAULT_JOURNAL_WEIGHT = 0.5

RECENCY_HALF_LIFE_YEARS = 8  # 时效性指数衰减半衰期


class MedicalReranker:
    """
    对初步检索候选进行多准则重排序。

    交叉编码器（cross-encoder）同时输入 (query, passage) 对，比双塔向量检索的
    余弦相似度更精确地建模相关性，但计算成本更高，因此只用于候选集的二次精排
    （通常几十条），而非全库检索。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        batch_size: int = 32,
        log: logging.Logger | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.log = log or logging.getLogger("reranker")

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        self.log.info(f"加载重排序模型：{model_name}  设备：{self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ── 相关性打分（交叉编码器）───────────────────────────────────
    @torch.no_grad()
    def _score_relevance(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            pairs = [[query, t] for t in batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(logits).cpu().tolist()
            scores.extend(probs)
        return scores

    # ── 时效性打分 ────────────────────────────────────────────────
    @staticmethod
    def _score_recency(pub_year, current_year: int | None = None) -> float:
        """指数衰减：score = 0.5 ** (age / half_life)。年份缺失/无效给中性分 0.5。"""
        if not pub_year:
            return 0.5
        try:
            pub_year = int(pub_year)
        except (TypeError, ValueError):
            return 0.5
        if pub_year <= 0:
            return 0.5
        current_year = current_year or datetime.date.today().year
        age = max(0, current_year - pub_year)
        return 0.5 ** (age / RECENCY_HALF_LIFE_YEARS)

    # ── 权威性打分 ────────────────────────────────────────────────
    @staticmethod
    def _score_authority(journal: str | None) -> float:
        if not journal:
            return DEFAULT_JOURNAL_WEIGHT
        return JOURNAL_WEIGHTS.get(journal.strip().lower(), DEFAULT_JOURNAL_WEIGHT)

    # ── 主入口 ────────────────────────────────────────────────────
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        criteria_weights: dict | None = None,
    ) -> list[dict]:
        """
        对候选列表进行多准则重排序。

        Args:
            query: 原始查询文本（交叉编码器不需要 BGE 指令前缀）
            candidates: [{chunk_id, text, metadata, ...}, ...]，text 应为完整正文
                        （而非截断预览），否则相关性打分会受截断影响
            top_k: 返回数量
            criteria_weights: {"relevance":.., "recency":.., "authority":..}

        Returns:
            按 final_score 降序排列的前 top_k 条，附加 relevance_score /
            recency_score / authority_score / final_score / final_rank 字段
        """
        if not candidates:
            return []
        weights = criteria_weights or DEFAULT_CRITERIA_WEIGHTS

        texts = [c.get("text", "") for c in candidates]
        relevance_scores = self._score_relevance(query, texts)

        reranked = []
        for cand, rel in zip(candidates, relevance_scores):
            meta = cand.get("metadata", {})
            recency = self._score_recency(meta.get("pub_year"))
            authority = self._score_authority(meta.get("journal"))
            final = (
                weights["relevance"] * rel
                + weights["recency"] * recency
                + weights["authority"] * authority
            )
            item = dict(cand)
            item.update({
                "relevance_score": round(float(rel), 6),
                "recency_score":   round(float(recency), 6),
                "authority_score": round(float(authority), 6),
                "final_score":     round(float(final), 6),
            })
            reranked.append(item)

        reranked.sort(key=lambda x: -x["final_score"])
        for i, item in enumerate(reranked, start=1):
            item["final_rank"] = i
        return reranked[:top_k]
