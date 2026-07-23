"""
format_checker.py — 输出格式与术语规范检查器（FormatChecker）
对应任务书 1.d：
  - 医学缩写首次出现是否给出全称（复用 retrieval/medical_vocab.py 的缩写表）
  - 回答是否包含必需的章节标题（核心答案/证据总结/参考文献，双语）
  - 参考文献部分是否完整（至少包含标题、期刊、年份）
"""

from __future__ import annotations

import logging
import re

from retrieval.medical_vocab import ABBREVIATIONS

from .prompt_templates import SECTION_HEADERS

_WORD_RE = re.compile(r"\b[a-zA-Z][\w-]*\b")
# 章节标题行两侧常见的 markdown 装饰符（"## "、"**...**"、末尾冒号等），
# 判断"是否为标题行"前先剥离，避免把正文里偶然出现的同名词组（如叙述句中的
# "no references were found"）误判为章节标题
_HEADER_DECORATION_RE = re.compile(r"^[#*\s]+|[#*\s:：]+$")
# 与 MedicalGenerationPipeline._postprocess() 生成的引用行格式一一对应：
# "[N] {title} — {journal} ({year}), {pmc_id}"
_REF_ENTRY_RE = re.compile(
    r"\[(?P<id>[^\]]+)\]\s*(?P<title>[^—]+?)\s*—\s*(?P<journal>[^(]+?)\s*\((?P<year>\d{4})\)"
)
# medical_vocab.ABBREVIATIONS 里少数 key 恰好和常见英文单词同形（最典型是
# "or"，统计学里是 odds ratio 的缩写，但同时是英语里最常见的连词之一）。
# 按原样小写匹配会导致几乎所有英文句子都被误判为"使用了未展开的缩写"，
# 因此这里显式排除这类已知的假阳性 key。
_COMMON_WORD_FALSE_POSITIVES = {"or"}


class FormatChecker:
    def __init__(self, log: logging.Logger | None = None):
        self.log = log or logging.getLogger("format_checker")

    # ── 缩写全称检查 ──────────────────────────────────────────────
    @staticmethod
    def check_abbreviations(text: str) -> dict:
        """
        检测文本中出现的已知医学缩写（对照 medical_vocab.ABBREVIATIONS），
        判断其全称是否也在文本某处出现。不强制要求"紧邻首次出现处"，
        只要求全称确实写出过，兼容"先给全称后用缩写"的常见写法。
        """
        text_lower = text.lower()
        words = {w.lower() for w in _WORD_RE.findall(text)}
        found_abbrs = sorted((words & ABBREVIATIONS.keys()) - _COMMON_WORD_FALSE_POSITIVES)

        with_full_form: list[str] = []
        missing_full_form: list[str] = []
        for abbr in found_abbrs:
            full_forms = ABBREVIATIONS[abbr]
            if any(ff.lower() in text_lower for ff in full_forms):
                with_full_form.append(abbr)
            else:
                missing_full_form.append(abbr)

        return {
            "abbreviations_found": found_abbrs,
            "with_full_form": with_full_form,
            "missing_full_form": missing_full_form,
            "pass": len(missing_full_form) == 0,
        }

    # ── 必需章节检查 ──────────────────────────────────────────────
    @staticmethod
    def check_required_sections(text: str, language: str = "zh") -> dict:
        """
        逐行剥离 markdown 装饰符后与目标标题做整行精确匹配（大小写不敏感），
        而非在全文里做子串搜索——否则正文叙述中偶然出现的同名词组
        （如 "no references were found"）会被误判为"已包含该章节"。
        """
        headers = SECTION_HEADERS.get(language, SECTION_HEADERS["zh"])
        required = list(headers.values())
        stripped_lines = [_HEADER_DECORATION_RE.sub("", ln).strip().lower() for ln in text.splitlines()]
        present = {name: name.lower() in stripped_lines for name in required}
        missing = [name for name, ok in present.items() if not ok]
        return {"required": required, "present": present, "missing": missing, "pass": len(missing) == 0}

    # ── 参考文献完整性检查 ────────────────────────────────────────
    @staticmethod
    def check_references_completeness(text: str, language: str = "zh") -> dict:
        """
        定位"参考文献/References"标题后的文本，逐条解析每一行 "[N] ..." 条目，
        检查该行是否包含标题、期刊、四位年份（_REF_ENTRY_RE 局部匹配即可，
        不要求匹配到行尾——条目末尾的 PMC id 等内容不影响"是否完整"的判断）。
        """
        header = SECTION_HEADERS.get(language, SECTION_HEADERS["zh"])["references"]
        m = re.search(re.escape(header), text, re.IGNORECASE)
        if not m:
            return {"n_entries": 0, "incomplete_entries": [], "pass": False, "reason": "未找到参考文献章节"}

        refs_block = text[m.end():]
        entries = re.findall(r"\[[^\]]+\]\s*[^\n\[]+", refs_block)
        incomplete = [e.strip() for e in entries if not _REF_ENTRY_RE.search(e)]

        return {
            "n_entries": len(entries),
            "n_complete": len(entries) - len(incomplete),
            "incomplete_entries": incomplete,
            "pass": len(entries) > 0 and len(incomplete) == 0,
        }

    # ── 主入口 ────────────────────────────────────────────────────
    def check(self, text: str, language: str = "zh") -> dict:
        abbr = self.check_abbreviations(text)
        sections = self.check_required_sections(text, language)
        refs = self.check_references_completeness(text, language)
        return {
            "abbreviations": abbr,
            "sections": sections,
            "references": refs,
            "overall_pass": abbr["pass"] and sections["pass"] and refs["pass"],
        }
