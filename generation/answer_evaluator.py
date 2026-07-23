"""
answer_evaluator.py — 生成答案多维度评估
四个维度：
  a) 文本相似性   — 与参考答案的 ROUGE-1/2/L（需要参考答案）
  b) 关键信息召回 — 正则提取医学关键信息，计算生成答案对参考答案的覆盖率（需要参考答案）
  c) 幻觉检测     — 无依据绝对化表述信号计数 -> 风险分数（无需参考答案）
  d) 可读性       — 平均句子长度等基础可读性统计（无需参考答案）
"""

from __future__ import annotations

import logging
import re

from rouge import Rouge

# ══════════════════════════════════════════════════════════════════
# b) 关键信息正则模式（中英双语）
# ══════════════════════════════════════════════════════════════════
KEY_INFO_PATTERNS: dict[str, re.Pattern] = {
    "percentage": re.compile(r"\d+(?:\.\d+)?\s?%"),
    "dosage": re.compile(
        r"\d+(?:\.\d+)?\s?(?:mg/kg/day|mg/kg|mg/day|mcg|μg|mg|g|ml|IU|单位|片|粒)\b"
        r"|剂量|dosage|dose\b",
        re.IGNORECASE,
    ),
    "time_range": re.compile(
        r"\d+(?:\.\d+)?\s?(?:years?|months?|weeks?|days?|年|个月|月|周|天)\b"
        r"|\d{4}\s?[-–—至]\s?\d{4}",
        re.IGNORECASE,
    ),
    "safety": re.compile(
        r"风险|副作用|不良反应|risk|side effect|adverse (?:reaction|event)", re.IGNORECASE
    ),
    "treatment": re.compile(
        r"建议|治疗方案|治疗|方案|recommend(?:ation)?|treatment|therapy", re.IGNORECASE
    ),
    "mechanism": re.compile(r"机制|原理|作用机制|mechanism|pathway", re.IGNORECASE),
}

# ══════════════════════════════════════════════════════════════════
# c) 幻觉信号正则模式（中英双语）
# ══════════════════════════════════════════════════════════════════
HALLUCINATION_SIGNALS: list[tuple[str, re.Pattern]] = [
    ("unattributed_claim", re.compile(
        r"研究表明|研究发现|studies show|research shows|research indicates", re.IGNORECASE
    )),
    ("unqualified_proof", re.compile(
        r"已被证明|证实|has been proven|proven to|demonstrated to", re.IGNORECASE
    )),
    ("absolute_100", re.compile(r"\b100\s?%")),
    ("overclaiming", re.compile(
        r"完全(?:安全|有效|无害)|completely (?:safe|effective|harmless)"
        r"|totally (?:safe|effective|harmless)|absolutely (?:safe|effective)",
        re.IGNORECASE,
    )),
]

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?.]+")
_CJK_RE = re.compile(r"[一-鿿]")


def extract_key_info(text: str) -> dict[str, list[str]]:
    """按 KEY_INFO_PATTERNS 从文本中提取各类别的关键信息片段。"""
    return {cat: pattern.findall(text) for cat, pattern in KEY_INFO_PATTERNS.items()}


def _normalize_matches(items: list[str]) -> set[str]:
    return {re.sub(r"\s+", " ", s.strip().lower()) for s in items if s.strip()}


def _pretokenize_for_rouge(text: str) -> str:
    """rouge 库按空格分词计算 n-gram 重叠；中文没有空格，需先按字切分才有意义。"""
    text = text.strip()
    if _CJK_RE.search(text):
        return " ".join(list(text))
    return text


class AnswerEvaluator:
    """
    生成答案质量评估器。reference（参考/标准答案）可选：
    提供时计算全部四个维度；不提供时仅计算幻觉检测与可读性（这两项本就无需参考答案）。
    """

    def __init__(self, log: logging.Logger | None = None):
        self.rouge = Rouge()
        self.log = log or logging.getLogger("answer_evaluator")

    # ── a) 文本相似性（ROUGE）────────────────────────────────────
    def evaluate_similarity(self, generated: str, reference: str) -> dict:
        hyp = _pretokenize_for_rouge(generated)
        ref = _pretokenize_for_rouge(reference)
        if not hyp or not ref:
            return {m: {"r": 0.0, "p": 0.0, "f": 0.0} for m in ("rouge-1", "rouge-2", "rouge-l")}
        try:
            return self.rouge.get_scores(hyp, ref, avg=True)
        except (ValueError, ZeroDivisionError) as e:
            self.log.warning(f"ROUGE 计算失败，返回零分：{e}")
            return {m: {"r": 0.0, "p": 0.0, "f": 0.0} for m in ("rouge-1", "rouge-2", "rouge-l")}

    # ── b) 关键信息召回率 ─────────────────────────────────────────
    def evaluate_key_info_recall(self, generated: str, reference: str) -> dict:
        """
        recall = overlap / gt_matches —— 生成答案覆盖了多少参考答案中的关键信息。
        overlap 按归一化后的字符串精确匹配计算（大小写/多余空白已忽略）。
        """
        gt_info = extract_key_info(reference)
        gen_info = extract_key_info(generated)

        per_category: dict[str, dict] = {}
        total_gt = 0
        total_overlap = 0
        for cat in KEY_INFO_PATTERNS:
            gt_set = _normalize_matches(gt_info[cat])
            gen_set = _normalize_matches(gen_info[cat])
            overlap = len(gt_set & gen_set)
            per_category[cat] = {
                "gt_matches": sorted(gt_set),
                "gen_matches": sorted(gen_set),
                "overlap": overlap,
                "recall": round(overlap / len(gt_set), 4) if gt_set else None,
            }
            total_gt += len(gt_set)
            total_overlap += overlap

        return {
            "overall_recall": round(total_overlap / total_gt, 4) if total_gt else None,
            "total_gt_matches": total_gt,
            "total_overlap": total_overlap,
            "per_category": per_category,
        }

    # ── c) 幻觉检测 ───────────────────────────────────────────────
    def evaluate_hallucination_risk(self, generated: str) -> dict:
        """信号越多风险越高：risk_score = min(1.0, 信号总数 / 5)。"""
        breakdown: dict[str, dict] = {}
        total = 0
        for name, pattern in HALLUCINATION_SIGNALS:
            matches = pattern.findall(generated)
            breakdown[name] = {"count": len(matches), "matches": matches[:10]}
            total += len(matches)

        risk_score = round(min(1.0, total / 5), 4)
        risk_level = "high" if risk_score >= 0.6 else ("medium" if risk_score >= 0.2 else "low")

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "total_signals": total,
            "signal_breakdown": breakdown,
        }

    # ── d) 可读性 ─────────────────────────────────────────────────
    @staticmethod
    def evaluate_readability(generated: str) -> dict:
        text = generated.strip()
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        n_sentences = len(sentences) or 1
        total_chars = len(text)
        total_words = len(text.split())
        return {
            "n_sentences": len(sentences),
            "avg_sentence_length_chars": round(total_chars / n_sentences, 2),
            "avg_sentence_length_words": round(total_words / n_sentences, 2),
            "total_chars": total_chars,
            "total_words": total_words,
        }

    # ── 主入口 ────────────────────────────────────────────────────
    def evaluate(self, generated: str, reference: str | None = None) -> dict:
        """
        综合评估入口。

        Args:
            generated: 待评估的生成答案
            reference: 参考/标准答案；None 时跳过 ROUGE 与关键信息召回

        Returns:
            {"rouge": dict|None, "key_info_recall": dict|None,
             "hallucination": dict, "readability": dict}
        """
        result = {
            "hallucination": self.evaluate_hallucination_risk(generated),
            "readability": self.evaluate_readability(generated),
        }
        if reference:
            result["rouge"] = self.evaluate_similarity(generated, reference)
            result["key_info_recall"] = self.evaluate_key_info_recall(generated, reference)
        else:
            result["rouge"] = None
            result["key_info_recall"] = None
        return result
