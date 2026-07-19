"""
query_processor.py — 医学查询理解与增强
输入：用户自然语言查询（中/英文）
输出：ProcessedQuery，包含向量检索查询、关键词查询、识别实体、过滤条件
"""

import re
import json
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .medical_vocab import (
    ABBREVIATIONS,
    SYNONYMS,
    COMPILED_PATTERNS,
    TIME_PATTERNS,
    STUDY_DESIGN_HINTS,
    ZH_TO_EN,
)

# BGE 模型检索任务的查询前缀（官方推荐）
BGE_QUERY_PREFIX = "Represent this question for searching relevant passages: "


# ══════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessedQuery:
    original:       str                      # 原始输入
    cleaned:        str                      # 基础清洗后
    entities:       dict[str, list[str]]     # 识别的实体 {type: [term, ...]}
    abbreviations:  dict[str, list[str]]     # 展开的缩写 {abbr: [full_form, ...]}
    synonyms:       dict[str, list[str]]     # 同义词扩展 {term: [synonym, ...]}
    vector_query:   str                      # 向量检索用（带 BGE 前缀）
    keyword_query:  str                      # 关键词检索用（原词 + 扩展词）
    expanded_terms: list[str]                # 所有扩展词（去重）
    filters:        dict                     # ChromaDB where 过滤条件
    imrad_hint:     str | None               # 推断的 imrad_type 倾向

    def summary(self) -> str:
        lines = [
            f"原始查询    : {self.original}",
            f"清洗后      : {self.cleaned}",
            f"识别实体    : {self.entities}",
            f"缩写展开    : {self.abbreviations}",
            f"同义词扩展  : {dict(list(self.synonyms.items())[:3])}{'...' if len(self.synonyms)>3 else ''}",
            f"扩展词（前5）: {self.expanded_terms[:5]}",
            f"向量查询    : {self.vector_query[:80]}...",
            f"关键词查询  : {self.keyword_query[:80]}...",
            f"过滤条件    : {self.filters}",
            f"IMRaD 倾向  : {self.imrad_hint}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 主处理器
# ══════════════════════════════════════════════════════════════════

class MedicalQueryProcessor:
    """
    医学查询理解与增强处理器。
    所有方法均为纯函数（无副作用），可在无 GPU 环境下使用。

    Args:
        log_path: JSONL 日志文件路径，None 则不记录日志。
    """

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, result: 'ProcessedQuery') -> None:
        """将一条 ProcessedQuery 以 JSON 行追加写入日志文件。"""
        if not self.log_path:
            return
        record = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'original':       result.original,
            'cleaned':        result.cleaned,
            'entities':       result.entities,
            'abbreviations':  result.abbreviations,
            'synonyms':       result.synonyms,
            'expanded_terms': result.expanded_terms,
            'vector_query':   result.vector_query,
            'keyword_query':  result.keyword_query,
            'filters':        result.filters,
            'imrad_hint':     result.imrad_hint,
        }
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # ── 0. 中文词汇替换 ───────────────────────────────────────────
    def _translate_zh(self, query: str) -> str:
        """将中文医学词汇替换为英文，便于后续实体识别与同义词扩展。"""
        for zh, en in ZH_TO_EN.items():
            query = query.replace(zh, en)
        return query

    # ── 1. 基础清洗 ───────────────────────────────────────────────
    def _clean(self, query: str) -> str:
        """去除多余空白、统一标点、转小写（保留数字与连字符）。"""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)             # 合并连续空白
        query = re.sub(r'["""]', '"', query)           # 统一引号
        query = re.sub(r"[''']", "'", query)           # 统一单引号
        query = re.sub(r'[。？！，、；：]', ' ', query) # 中文标点转空格
        query = re.sub(r'\s+', ' ', query).strip()
        return query

    # ── 2. 实体识别 ───────────────────────────────────────────────
    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """用正则模式识别医学实体，返回 {entity_type: [matched_terms]}。"""
        entities: dict[str, list[str]] = {}
        for etype, pattern in COMPILED_PATTERNS.items():
            matches = [m.group(0).lower() for m in pattern.finditer(text)]
            if matches:
                entities[etype] = list(dict.fromkeys(matches))  # 去重保序
        return entities

    # ── 3. 缩写展开 ───────────────────────────────────────────────
    def _expand_abbreviations(self, text: str) -> dict[str, list[str]]:
        """
        识别并展开文本中的医学缩写。
        匹配规则：单词边界，不区分大小写。
        """
        found: dict[str, list[str]] = {}
        words = re.findall(r"[\w'-]+", text.lower())
        for word in words:
            if word in ABBREVIATIONS:
                found[word] = ABBREVIATIONS[word]
        return found

    # ── 4. 同义词扩展 ─────────────────────────────────────────────
    def _expand_synonyms(
        self,
        text: str,
        abbr_expansions: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """
        对文本中出现的词条（含缩写展开结果）查找同义词。
        返回 {匹配词: [同义词列表]}。
        """
        found: dict[str, list[str]] = {}
        text_lower = text.lower()

        # 在原始文本中直接查找
        for term, syns in SYNONYMS.items():
            if term.lower() in text_lower:
                found[term] = syns

        # 对缩写展开的全称也做同义词查找
        for full_forms in abbr_expansions.values():
            for form in full_forms:
                if form in SYNONYMS:
                    found[form] = SYNONYMS[form]

        return found

    # ── 5. 过滤条件提取 ───────────────────────────────────────────
    def _extract_filters(self, text: str) -> dict:
        """
        从查询文本中提取 ChromaDB where 过滤条件。
        当前支持：时间范围（pub_year）、研究设计（imrad_type）。
        """
        filters: dict = {}

        # 时间范围
        for pattern, year_fn in TIME_PATTERNS:
            m = pattern.search(text)
            if m:
                start, end = year_fn(m)
                if start == end:
                    filters['pub_year'] = {'$eq': start}
                else:
                    # ChromaDB 要求每个字段表达式只能有一个操作符，
                    # 范围条件需用 $and 组合两个单操作符表达式
                    filters['$and'] = [
                        {'pub_year': {'$gte': start}},
                        {'pub_year': {'$lte': end}},
                    ]
                break  # 只取第一个时间条件

        return filters

    # ── 6. IMRaD 倾向推断 ─────────────────────────────────────────
    def _infer_imrad(self, text: str) -> str | None:
        """根据查询中的关键词推断用户偏好的 IMRaD 章节。"""
        text_lower = text.lower()
        votes: dict[str, int] = {}
        for keyword, section in STUDY_DESIGN_HINTS.items():
            if keyword in text_lower:
                votes[section] = votes.get(section, 0) + 1
        return max(votes, key=votes.get) if votes else None

    # ── 7. 构建检索查询 ───────────────────────────────────────────
    def _build_vector_query(self, cleaned: str) -> str:
        """向量检索查询：在清洗后的原始查询前加 BGE 指令前缀。"""
        return BGE_QUERY_PREFIX + cleaned

    def _build_keyword_query(
        self,
        cleaned: str,
        abbr_expansions: dict[str, list[str]],
        synonyms: dict[str, list[str]],
    ) -> str:
        """
        关键词检索查询：原始查询 + 缩写展开全称 + 同义词，空格分隔。
        用于 BM25 / 全文检索时补充召回。
        """
        extra: list[str] = []
        for forms in abbr_expansions.values():
            extra.extend(forms)
        for syns in synonyms.values():
            extra.extend(syns)
        # 去重，过滤掉已在原文中出现的词
        cleaned_lower = cleaned.lower()
        unique_extra = [
            t for t in dict.fromkeys(extra)
            if t.lower() not in cleaned_lower
        ]
        return (cleaned + ' ' + ' '.join(unique_extra)).strip()

    # ── 主入口 ────────────────────────────────────────────────────
    def process(self, query: str) -> ProcessedQuery:
        """
        处理医学查询，返回 ProcessedQuery。

        Args:
            query: 用户输入的自然语言查询（中英文均可）

        Returns:
            ProcessedQuery 包含所有增强信息
        """
        # Step 0: 中文词汇替换（替换后再清洗，保留中文原文用于 vector_query）
        translated = self._translate_zh(query)

        # Step 1: 基础清洗
        cleaned = self._clean(translated)

        # Step 2: 实体识别
        entities = self._extract_entities(cleaned)

        # Step 3: 缩写展开
        abbr_expansions = self._expand_abbreviations(cleaned)

        # Step 4: 同义词扩展（基于原文 + 缩写展开）
        synonyms = self._expand_synonyms(cleaned, abbr_expansions)

        # Step 5: 提取过滤条件
        filters = self._extract_filters(cleaned)

        # Step 6: IMRaD 倾向
        imrad_hint = self._infer_imrad(cleaned)

        # Step 7: 构建检索查询（向量查询保留原始中文，语义更准确）
        vector_query  = self._build_vector_query(query)
        keyword_query = self._build_keyword_query(cleaned, abbr_expansions, synonyms)

        # 汇总所有扩展词（去重保序）
        all_extra: list[str] = []
        for forms in abbr_expansions.values():
            all_extra.extend(forms)
        for syns in synonyms.values():
            all_extra.extend(syns)
        expanded_terms = list(dict.fromkeys(all_extra))

        result = ProcessedQuery(
            original       = query,
            cleaned        = cleaned,
            entities       = entities,
            abbreviations  = abbr_expansions,
            synonyms       = synonyms,
            vector_query   = vector_query,
            keyword_query  = keyword_query,
            expanded_terms = expanded_terms,
            filters        = filters,
            imrad_hint     = imrad_hint,
        )
        self._log(result)
        return result
