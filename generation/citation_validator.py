"""
citation_validator.py — 引用编号校验（CitationValidator）
从生成文本中用正则提取所有 [来源 N] 引用编号，检查是否都落在提供文献的编号
范围内。发现无效引用（编号不存在）或缺失引用（有实质内容却一个引用都没给）时，
提供可用于触发重试的修正指令，以及重试仍失败后的兜底修正（直接剔除无效标记）。
"""

from __future__ import annotations

import re

_CITATION_RE = re.compile(r"\[来源\s*(\d+)\]")
_BOUNDARY_MARKERS = ("无法回答", "cannot be answered")


class CitationValidator:
    """校验生成文本中的 [来源 N] 引用是否都指向真实存在的来源编号。"""

    @staticmethod
    def extract_citations(text: str) -> list[int]:
        return [int(n) for n in _CITATION_RE.findall(text)]

    def validate(self, text: str, valid_ids: set[int]) -> dict:
        """
        Args:
            text: 待校验的生成文本
            valid_ids: 当前上下文中真实存在的引用编号集合（ContextAssembler
                       分配的 _citation_id）

        Returns:
            {"cited_ids": [...], "invalid_ids": [...], "has_invalid": bool,
             "missing_citations": bool, "pass": bool}
        """
        cited = self.extract_citations(text)
        unique_cited = sorted(set(cited))
        invalid = sorted(set(cited) - valid_ids)

        # 命中知识边界声明（"无法回答"）时不应被当作"缺失引用"扣分——
        # 拒答本身就是正确行为，不需要引用
        is_boundary_response = any(m in text for m in _BOUNDARY_MARKERS)
        missing = len(cited) == 0 and bool(valid_ids) and not is_boundary_response

        return {
            "cited_ids": unique_cited,
            "invalid_ids": invalid,
            "has_invalid": len(invalid) > 0,
            "missing_citations": missing,
            "is_boundary_response": is_boundary_response,
            "pass": len(invalid) == 0 and not missing,
        }

    @staticmethod
    def build_retry_instruction(validation: dict) -> str:
        """生成一段附加到重试 prompt 末尾的修正指令。"""
        parts: list[str] = []
        if validation["invalid_ids"]:
            bad = ", ".join(f"[来源 {i}]" for i in validation["invalid_ids"])
            parts.append(
                f"CORRECTION NEEDED: your previous answer cited {bad}, but these source "
                f"numbers do not exist in the provided sources. Remove or correct these "
                f"citations — only cite numbers that were actually given to you."
            )
        if validation["missing_citations"]:
            parts.append(
                "CORRECTION NEEDED: your previous answer did not cite any sources despite "
                "sources being available. Every factual claim must include a [来源 N] citation."
            )
        return "\n".join(parts)

    @staticmethod
    def strip_invalid_citations(text: str, valid_ids: set[int]) -> str:
        """兜底修正：重试次数用尽仍未通过时，直接删除无效引用标记，不让虚假编号留在文本里。"""
        def _replace(m: re.Match) -> str:
            n = int(m.group(1))
            return m.group(0) if n in valid_ids else ""
        return _CITATION_RE.sub(_replace, text)
