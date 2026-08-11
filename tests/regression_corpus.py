"""
tests/regression_corpus.py — 全量功能回归测试用例集（数据驱动语料生成器）

为什么是"语料 + 参数化运行器"而不是 1000 个手写 def test_：
  - 1000 个手写方法会是几万行、大量重复、无法维护；
  - 更重要的是，现有真实端到端测试每条要真实调一次 LLM（5-10s，GPU 绑定），
    1000 条全真实跑一遍要数小时且不稳定——那不是好工程。
  因此这里把"用例"沉淀成一份结构化数据集（每条 = 输入 + 独立推导出的期望），
  由 tests/test_regression_suite.py 用 unittest.subTest 逐条跑，全部针对
  **纯逻辑函数**（过滤条件拆分、缓存键、查询理解、上下文组装、引用/格式校验、
  工具调度、会话管理、Agent 状态），不加载任何 GPU 模型、不调用 LLM——
  1000+ 条能在几分钟内全量跑完。真实调模型的冒烟用例单独放在
  tests/test_regression_llm_smoke.py，不混进这 1000 条。

期望值尽量"独立推导"而不是"用被测函数自己算一遍"（否则就成了循环验证、
只能锁定当前行为而抓不到回归）：
  - 过滤条件拆分：期望由构造输入时"这个 clause 用的是不是 range 操作符"直接决定；
  - 缓存键：期望是"两个输入相等 <=> 键相等"这种关系，而非某个绝对哈希值；
  - 范围比较：构造 value 明确处于 threshold 的上/下/相等位置，期望用显式真值表；
  - 查询理解：缩写展开的期望来自词表本身（独立事实源），验证的是"检测+透传"链路。

生成是确定性的（固定顺序、固定构造规则），因此 build_corpus() 每次输出一致。
直接运行本文件会把语料落盘成 tests/regression_corpus.jsonl 供人工查看/版本化：
    PYTHONUTF8=1 python tests/regression_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.medical_vocab import ABBREVIATIONS

CORPUS_PATH = Path(__file__).resolve().parent / "regression_corpus.jsonl"

# 与 pmc_vector_index._CURRENT_YEAR 无关的固定上界不在这里断言（见 query_time_filter
# 只断言可独立确定的边界），因此本文件不依赖任何"当前年份"常量。


def _case(cid: str, category: str, inp: dict, expect: dict) -> dict:
    return {"id": cid, "category": category, "input": inp, "expect": expect}


# ══════════════════════════════════════════════════════════════════
# 1. where_filter 拆分（ChromaDB range 过滤绕行的核心逻辑）
# ══════════════════════════════════════════════════════════════════

_RANGE_OPS = ["$gte", "$gt", "$lte", "$lt"]
_EQ_FIELDS = [("journal", "PLoS Biology"), ("chunk_type", "body"), ("imrad_type", "methods")]
_RANGE_FIELDS = ["pub_year", "token_count"]


def gen_where_filter_split() -> list[dict]:
    cases: list[dict] = []
    n = 0

    # 单等值条件：全部进 safe，range 为空
    for field, val in _EQ_FIELDS:
        cases.append(_case(
            f"wfs_eq_{n}", "where_filter_split",
            {"where_filter": {field: val}},
            {"range_pairs": [], "safe_is_none": False},
        ))
        n += 1

    # $in / $ne 也算安全（非 range）
    for field, val in _EQ_FIELDS:
        cases.append(_case(
            f"wfs_in_{n}", "where_filter_split",
            {"where_filter": {field: {"$in": [val, "other"]}}},
            {"range_pairs": [], "safe_is_none": False},
        ))
        n += 1
        cases.append(_case(
            f"wfs_ne_{n}", "where_filter_split",
            {"where_filter": {field: {"$ne": val}}},
            {"range_pairs": [], "safe_is_none": False},
        ))
        n += 1

    # 单 range 条件：进 range，safe 为 None
    for field in _RANGE_FIELDS:
        for op in _RANGE_OPS:
            cases.append(_case(
                f"wfs_range_{n}", "where_filter_split",
                {"where_filter": {field: {op: 2010}}},
                {"range_pairs": [[field, op]], "safe_is_none": True},
            ))
            n += 1

    # $and 两条 range（同字段区间）
    for field in _RANGE_FIELDS:
        for lo_op in ["$gte", "$gt"]:
            for hi_op in ["$lte", "$lt"]:
                cases.append(_case(
                    f"wfs_and_range_{n}", "where_filter_split",
                    {"where_filter": {"$and": [{field: {lo_op: 2000}}, {field: {hi_op: 2020}}]}},
                    {"range_pairs": sorted([[field, lo_op], [field, hi_op]]), "safe_is_none": True},
                ))
                n += 1

    # $and 混合：等值 + range（等值进 safe，range 绕行）
    for field, val in _EQ_FIELDS:
        for rfield in _RANGE_FIELDS:
            for op in _RANGE_OPS:
                cases.append(_case(
                    f"wfs_and_mix_{n}", "where_filter_split",
                    {"where_filter": {"$and": [{field: val}, {rfield: {op: 2015}}]}},
                    {"range_pairs": [[rfield, op]], "safe_is_none": False},
                ))
                n += 1

    # $and 两个等值（全 safe）
    for i in range(len(_EQ_FIELDS)):
        for j in range(len(_EQ_FIELDS)):
            if i == j:
                continue
            (fa, va), (fb, vb) = _EQ_FIELDS[i], _EQ_FIELDS[j]
            cases.append(_case(
                f"wfs_and_eq_{n}", "where_filter_split",
                {"where_filter": {"$and": [{fa: va}, {fb: vb}]}},
                {"range_pairs": [], "safe_is_none": False},
            ))
            n += 1

    # 空/None 过滤
    cases.append(_case(f"wfs_none_{n}", "where_filter_split", {"where_filter": None},
                       {"range_pairs": [], "safe_is_none": True}))
    n += 1
    # 空 dict：函数按 falsy 原样返回（返回 {} 而非 None），safe 不是 None
    cases.append(_case(f"wfs_empty_{n}", "where_filter_split", {"where_filter": {}},
                       {"range_pairs": [], "safe_is_none": False}))
    n += 1

    # 多字段等值 + 多字段 range 的混合（三 clause $and）
    for field, val in _EQ_FIELDS:
        for op in _RANGE_OPS:
            cases.append(_case(
                f"wfs_and3_{n}", "where_filter_split",
                {"where_filter": {"$and": [{field: val}, {"pub_year": {"$gte": 2005}}, {"token_count": {op: 500}}]}},
                {"range_pairs": sorted([["pub_year", "$gte"], ["token_count", op]]), "safe_is_none": False},
            ))
            n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 2. range 条件的客户端二次过滤判定
# ══════════════════════════════════════════════════════════════════

def _eval_single(op: str, value, threshold) -> bool:
    if op == "$gt":
        return value > threshold
    if op == "$gte":
        return value >= threshold
    if op == "$lt":
        return value < threshold
    if op == "$lte":
        return value <= threshold
    raise ValueError(op)


def gen_where_filter_eval() -> list[dict]:
    cases: list[dict] = []
    n = 0
    field = "pub_year"
    threshold = 2010

    # 单条件：value 在 threshold 上/下/相等三个位置 × 四个操作符
    for op in _RANGE_OPS:
        for delta in (-1, 0, 1):
            value = threshold + delta
            expected = _eval_single(op, value, threshold)
            cases.append(_case(
                f"wfe_single_{n}", "where_filter_eval",
                {"metadata": {field: value}, "range_conditions": [[field, op, threshold]]},
                {"passes": expected},
            ))
            n += 1

    # 缺字段 → 一定不通过
    for op in _RANGE_OPS:
        cases.append(_case(
            f"wfe_missing_{n}", "where_filter_eval",
            {"metadata": {"journal": "X"}, "range_conditions": [[field, op, threshold]]},
            {"passes": False},
        ))
        n += 1

    # 单条件更细网格：多阈值 × 多操作符 × value 处于上下相等位置
    for threshold2 in (2000, 2005, 2012, 2018):
        for op in _RANGE_OPS:
            for delta in (-2, -1, 0, 1, 2):
                value = threshold2 + delta
                cases.append(_case(
                    f"wfe_grid_{n}", "where_filter_eval",
                    {"metadata": {field: value}, "range_conditions": [[field, op, threshold2]]},
                    {"passes": _eval_single(op, value, threshold2)},
                ))
                n += 1

    # 双条件区间（AND：两者都满足才通过）
    for lo in (2000, 2008, 2010, 2015):
        for hi in (2005, 2012, 2018, 2020):
            for value in (1999, 2003, 2009, 2010, 2013, 2019, 2021):
                conds = [["pub_year", "$gte", lo], ["pub_year", "$lte", hi]]
                expected = (value >= lo) and (value <= hi)
                cases.append(_case(
                    f"wfe_range_{n}", "where_filter_eval",
                    {"metadata": {"pub_year": value}, "range_conditions": conds},
                    {"passes": expected},
                ))
                n += 1

    # 多字段 AND（pub_year + token_count）
    for py in (2008, 2012, 2016):
        for tc in (300, 500, 800):
            conds = [["pub_year", "$gte", 2010], ["token_count", "$lte", 500]]
            expected = (py >= 2010) and (tc <= 500)
            cases.append(_case(
                f"wfe_multi_{n}", "where_filter_eval",
                {"metadata": {"pub_year": py, "token_count": tc}, "range_conditions": conds},
                {"passes": expected},
            ))
            n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 3. 缓存键（GenerationCache.make_key）：相同输入=>相同键，不同输入=>不同键
# ══════════════════════════════════════════════════════════════════

def gen_cache_key() -> list[dict]:
    cases: list[dict] = []
    n = 0
    base_queries = [
        "diabetes genes", "wnt signaling cancer", "survival analysis", "sleep memory", "二甲双胍机制",
        "hypertension treatment", "crispr gene editing", "kaplan meier", "gwas snp diabetes",
        "metformin ampk", "beta catenin", "tumor suppressor p53",
    ]
    temps = ["0.0", "0.1", "0.2"]
    models = ["qwen2.5:7b-instruct", "other-model"]

    # stable：同一份 parts（键顺序打乱）应得同一个键
    for q in base_queries:
        for t in temps:
            parts_a = {"model": models[0], "prompt": q, "temperature": t, "system": "sys"}
            parts_b = {"temperature": t, "system": "sys", "prompt": q, "model": models[0]}  # 打乱顺序
            cases.append(_case(f"ck_stable_{n}", "cache_key",
                               {"parts_a": parts_a, "parts_b": parts_b}, {"equal": True}))
            n += 1

    # distinct：任一字段不同 => 键不同
    for q in base_queries:
        base = {"model": models[0], "prompt": q, "temperature": "0.1", "system": "sys"}
        variants = [
            {**base, "prompt": q + " extra"},
            {**base, "temperature": "0.2"},
            {**base, "model": models[1]},
            {**base, "system": "other"},
        ]
        for v in variants:
            cases.append(_case(f"ck_distinct_{n}", "cache_key",
                               {"parts_a": base, "parts_b": v}, {"equal": False}))
            n += 1

    # 跨查询两两不同
    for i in range(len(base_queries)):
        for j in range(len(base_queries)):
            if i >= j:
                continue
            pa = {"model": models[0], "prompt": base_queries[i], "temperature": "0.1", "system": "s"}
            pb = {"model": models[0], "prompt": base_queries[j], "temperature": "0.1", "system": "s"}
            cases.append(_case(f"ck_cross_{n}", "cache_key",
                               {"parts_a": pa, "parts_b": pb}, {"equal": False}))
            n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 4. 检索缓存参数规范化（_canonical_params）：键顺序无关 + 值敏感
# ══════════════════════════════════════════════════════════════════

def gen_canonical_params() -> list[dict]:
    cases: list[dict] = []
    n = 0
    samples = [
        {"query": "diabetes", "top_k": 6, "fusion_strategy": "rrf"},
        {"query": "cancer wnt", "top_k": 8, "fusion_strategy": "weighted"},
        {"query": "sleep", "top_k": 10, "fusion_strategy": "simple", "where_filter": {"pub_year": {"$gte": 2010}}},
        {"query": "metformin", "top_k": 5, "fusion_strategy": "rrf", "where_filter": None},
        {"query": "hypertension therapy", "top_k": 12, "fusion_strategy": "weighted"},
        {"query": "gwas snp", "top_k": 4, "fusion_strategy": "rrf", "where_filter": {"journal": "Nature"}},
        {"query": "survival curve", "top_k": 20, "fusion_strategy": "simple"},
        {"query": "beta catenin", "top_k": 7, "fusion_strategy": "rrf",
         "where_filter": {"$and": [{"pub_year": {"$gte": 2003}}, {"pub_year": {"$lte": 2004}}]}},
        {"query": "crispr cas9", "top_k": 9, "fusion_strategy": "weighted"},
        {"query": "p53 tumor", "top_k": 3, "fusion_strategy": "simple", "where_filter": {"chunk_type": "body"}},
    ]
    # 顺序无关：同 dict 不同插入顺序 => 规范化字符串相同
    for s in samples:
        reordered = {k: s[k] for k in reversed(list(s.keys()))}
        cases.append(_case(f"cp_order_{n}", "canonical_params",
                           {"a": s, "b": reordered}, {"equal": True}))
        n += 1
    # 值敏感：改任一值 => 不同
    for s in samples:
        for key in s:
            b = dict(s)
            if isinstance(s[key], int):
                b[key] = s[key] + 1
            elif isinstance(s[key], str):
                b[key] = s[key] + "_x"
            else:
                b[key] = {"changed": True}
            cases.append(_case(f"cp_val_{n}", "canonical_params",
                               {"a": s, "b": b}, {"equal": False}))
            n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 5. 查询理解：缩写展开（每个词表条目一条，验证检测+透传链路）
# ══════════════════════════════════════════════════════════════════

def gen_query_abbrev() -> list[dict]:
    cases: list[dict] = []
    # 只测 ASCII（英文）缩写：查询处理器 Step 0 会先做中文术语翻译，中文缩写
    # 键在缩写展开步骤之前就已被翻译掉，因此不通过 process() 的缩写展开路径
    # （这是流水线阶段顺序决定的真实行为，不是 bug）。
    for i, abbr in enumerate(sorted(k for k in ABBREVIATIONS if k.isascii())):
        query = f"What is the treatment for {abbr} in adults?"
        cases.append(_case(f"qa_abbr_{i}", "query_abbrev",
                           {"query": query, "abbr": abbr},
                           {"expansions": ABBREVIATIONS[abbr]}))
    return cases


# ══════════════════════════════════════════════════════════════════
# 6. 查询理解：时间范围过滤条件提取
# ══════════════════════════════════════════════════════════════════

def gen_query_time_filter() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # between X and Y —— 完全可确定
    for lo, hi in [(2003, 2004), (2010, 2015), (2000, 2020), (2005, 2018),
                   (2001, 2009), (2011, 2019), (2002, 2007), (2013, 2017)]:
        cases.append(_case(
            f"qt_between_{n}", "query_time_filter",
            {"query": f"studies published between {lo} and {hi} on cancer"},
            {"kind": "and_range", "gte": lo, "lte": hi},
        ))
        n += 1
    # in YYYY —— eq
    for y in [2003, 2008, 2012, 2019, 2021, 2001, 2006, 2014, 2017, 2020]:
        cases.append(_case(
            f"qt_in_{n}", "query_time_filter",
            {"query": f"research in {y} about diabetes"},
            {"kind": "eq", "eq": y},
        ))
        n += 1
    # after YYYY —— 下界 = y+1（上界依赖当前年份，不断言）
    for y in [2005, 2010, 2015, 2018, 2001, 2008, 2013, 2019]:
        cases.append(_case(
            f"qt_after_{n}", "query_time_filter",
            {"query": f"trials after {y} on hypertension"},
            {"kind": "and_lower", "gte": y + 1},
        ))
        n += 1
    # since YYYY —— 下界 = y
    for y in [2006, 2011, 2016, 2020, 2002, 2009, 2014, 2018]:
        cases.append(_case(
            f"qt_since_{n}", "query_time_filter",
            {"query": f"papers since {y} on genomics"},
            {"kind": "and_lower", "gte": y},
        ))
        n += 1
    # before YYYY —— 上界 = y-1，下界固定 1900
    for y in [2005, 2010, 2015, 2020, 2003, 2008, 2013, 2019]:
        cases.append(_case(
            f"qt_before_{n}", "query_time_filter",
            {"query": f"literature before {y} on vaccines"},
            {"kind": "and_range", "gte": 1900, "lte": y - 1},
        ))
        n += 1
    # 无时间线索 —— 不应有 pub_year 过滤
    for q in ["what genes cause diabetes", "wnt pathway in cancer", "kaplan meier survival",
              "metformin mechanism of action", "sleep and memory consolidation"]:
        cases.append(_case(f"qt_none_{n}", "query_time_filter",
                           {"query": q}, {"kind": "none"}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 7. 引用抽取与校验（CitationValidator）
# ══════════════════════════════════════════════════════════════════

def gen_citation_extract() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # 全部合法
    for ids in [[1], [1, 2], [1, 2, 3], [2, 3], [1, 3]]:
        valid = sorted(set(ids) | {1, 2, 3})
        text = "Answer body " + " ".join(f"[来源 {i}]" for i in ids) + "."
        cases.append(_case(f"cit_ok_{n}", "citation_extract",
                           {"text": text, "valid_ids": valid},
                           {"cited_ids": sorted(set(ids)), "has_invalid": False, "missing": False, "pass": True}))
        n += 1
    # 含非法编号（超出 valid）
    for ids, valid in [([4], [1, 2, 3]), ([1, 5], [1, 2, 3]), ([9], [1]), ([2, 8], [1, 2])]:
        cited = sorted(set(ids))
        invalid = sorted(set(ids) - set(valid))
        text = "Body " + " ".join(f"[来源 {i}]" for i in ids)
        cases.append(_case(f"cit_bad_{n}", "citation_extract",
                           {"text": text, "valid_ids": sorted(valid)},
                           {"cited_ids": cited, "has_invalid": bool(invalid), "missing": False,
                            "pass": len(invalid) == 0}))
        n += 1
    # 缺引用（有 valid_ids 但正文无标记，且非拒答）
    for valid in [[1], [1, 2], [1, 2, 3]]:
        cases.append(_case(f"cit_missing_{n}", "citation_extract",
                           {"text": "This is an answer without any citation markers.", "valid_ids": valid},
                           {"cited_ids": [], "has_invalid": False, "missing": True, "pass": False}))
        n += 1
    # 重复引用去重
    for ids in [[1, 1], [2, 2, 2], [1, 2, 1, 2]]:
        text = "Body " + " ".join(f"[来源 {i}]" for i in ids)
        cases.append(_case(f"cit_dup_{n}", "citation_extract",
                           {"text": text, "valid_ids": [1, 2, 3]},
                           {"cited_ids": sorted(set(ids)), "has_invalid": False, "missing": False, "pass": True}))
        n += 1
    # 各种间距格式 [来源1] / [来源 1] / [来源  1]
    for spacing in ["[来源1]", "[来源 1]", "[来源  1]"]:
        cases.append(_case(f"cit_space_{n}", "citation_extract",
                           {"text": f"Body {spacing} end.", "valid_ids": [1, 2]},
                           {"cited_ids": [1], "has_invalid": False, "missing": False, "pass": True}))
        n += 1
    # valid_ids 为空时不算 missing（无来源可引）
    cases.append(_case(f"cit_novalid_{n}", "citation_extract",
                       {"text": "No sources available.", "valid_ids": []},
                       {"cited_ids": [], "has_invalid": False, "missing": False, "pass": True}))
    n += 1
    # 扩充：批量组合合法引用（[1..6] 选 2/3/4，全部合法）
    import itertools
    valid_all = [1, 2, 3, 4, 5, 6]
    for r in (2, 3, 4):
        for combo in itertools.combinations([1, 2, 3, 4, 5], r):
            text = "Ans " + " ".join(f"[来源 {i}]" for i in combo)
            cases.append(_case(f"cit_combo{r}_{n}", "citation_extract",
                               {"text": text, "valid_ids": valid_all},
                               {"cited_ids": list(combo), "has_invalid": False, "missing": False, "pass": True}))
            n += 1
    # 扩充：组合里混入一个非法编号（7 超出 valid_all）
    for combo in itertools.combinations([1, 2, 3], 2):
        ids = list(combo) + [7]
        text = "Ans " + " ".join(f"[来源 {i}]" for i in ids)
        cases.append(_case(f"cit_badmix_{n}", "citation_extract",
                           {"text": text, "valid_ids": valid_all},
                           {"cited_ids": sorted(ids), "has_invalid": True, "missing": False, "pass": False}))
        n += 1
    # 扩充：[1..6] 选 2/3，valid_ids=[1..6] 全合法
    for r in (2, 3):
        for combo in itertools.combinations([1, 2, 3, 4, 5, 6], r):
            text = "Answer " + " ".join(f"[来源 {i}]" for i in combo)
            cases.append(_case(f"cit_c6_{r}_{n}", "citation_extract",
                               {"text": text, "valid_ids": valid_all},
                               {"cited_ids": list(combo), "has_invalid": False, "missing": False, "pass": True}))
            n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 8. 上下文组装：去重与引用编号
# ══════════════════════════════════════════════════════════════════

def _doc(chunk_id: str, text: str, doc_id: str, score: float) -> dict:
    return {"chunk_id": chunk_id, "text": text, "metadata": {"doc_id": doc_id, "journal": "J", "pub_year": 2004},
            "final_score": score}


def gen_context_dedup() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # 完全相同文本（不同 chunk_id、同 doc）应被 Jaccard 去重到 1 条
    identical_texts = [
        "the wnt signaling pathway regulates cell proliferation in cancer",
        "kaplan meier survival analysis estimates the median tumor latency period",
        "metformin activates the ampk energy sensor and lowers hepatic glucose output",
    ]
    for t_idx, itext in enumerate(identical_texts):
        for k in (2, 3, 4, 5, 6):
            docs = [_doc(f"c{t_idx}_{i}", itext, "D1", 0.9 - i * 0.01) for i in range(k)]
            cases.append(_case(f"cd_identical_{n}", "context_dedup",
                               {"docs": docs}, {"n_selected": 1, "contiguous": True}))
            n += 1
    # 完全不同文本、不同 doc、数量 <= 预算 => 全部保留，引用号连续
    distinct_texts = [
        "kaplan meier survival analysis estimates median latency",
        "wnt beta catenin stabilization drives tumor growth",
        "metformin activates ampk energy sensor pathway",
        "sleep spindles support memory consolidation overnight",
        "single nucleotide polymorphism associated with diabetes risk",
        "crispr cas9 mediated knockout of tumor suppressor genes",
    ]
    for k in (2, 3, 4, 5):
        docs = [_doc(f"x{i}", distinct_texts[i], f"D{i}", 0.9 - i * 0.05) for i in range(k)]
        cases.append(_case(f"cd_distinct_{n}", "context_dedup",
                           {"docs": docs}, {"n_selected": k, "contiguous": True}))
        n += 1
    # max_per_source=2 多样性排序：某来源超过 2 条时，超出的 chunk 不丢弃，而是
    # 被降优先级排到其它来源之后。构造 doc A 有 (2+extra) 条、doc B 有 1 条，
    # 期望：全部保留，且最后一条是 A 里分数最低的那条溢出 chunk。
    for extra in (1, 2, 3):
        n_a = 2 + extra
        docs_a = [_doc(f"A{i}", distinct_texts[i], "DOCA", 0.9 - i * 0.05) for i in range(n_a)]
        doc_b = [_doc("B0", distinct_texts[5], "DOCB", 0.50)]
        cases.append(_case(f"cd_persource_{n}", "context_dedup",
                           {"docs": docs_a + doc_b},
                           {"n_selected": n_a + 1, "contiguous": True, "last_chunk_id": f"A{n_a - 1}"}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 9. 工具调度引擎：参数校验 + 重试分类
# ══════════════════════════════════════════════════════════════════

def gen_tool_dispatch() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # 合法参数 + 正常 handler => 成功，无重试
    for q in ["a", "diabetes", "wnt cancer"]:
        cases.append(_case(f"td_ok_{n}", "tool_dispatch",
                           {"args": {"query": q}, "behavior": "ok", "max_retries": 2},
                           {"success": True, "retryable": False, "retry_count": 0}))
        n += 1
    # 缺必填参数 => 校验失败，不可重试，不重试
    for bad in [{}, {"top_k": 5}, {"wrong": "x"}]:
        cases.append(_case(f"td_badparam_{n}", "tool_dispatch",
                           {"args": bad, "behavior": "ok", "max_retries": 3},
                           {"success": False, "retryable": False, "retry_count": 0}))
        n += 1
    # 可重试异常：耗尽 max_retries 后失败，retry_count == max_retries
    for mr in (0, 1, 2, 3):
        cases.append(_case(f"td_retry_{n}", "tool_dispatch",
                           {"args": {"query": "x"}, "behavior": "raise_retryable", "max_retries": mr},
                           {"success": False, "retryable": True, "retry_count": mr}))
        n += 1
    # 不可重试异常（普通 ValueError）：立即失败，不重试
    for mr in (1, 2, 3):
        cases.append(_case(f"td_value_{n}", "tool_dispatch",
                           {"args": {"query": "x"}, "behavior": "raise_value", "max_retries": mr},
                           {"success": False, "retryable": False, "retry_count": 0}))
        n += 1
    # 可重试第 k 次成功
    for mr, succeed_on in [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2)]:
        cases.append(_case(f"td_recover_{n}", "tool_dispatch",
                           {"args": {"query": "x"}, "behavior": "recover", "max_retries": mr, "succeed_on": succeed_on},
                           {"success": True, "retryable": True, "retry_count": succeed_on}))
        n += 1
    # 扩充：多种合法 query + top_k + fusion_strategy 组合（均在 schema 内）
    queries = ["query one", "query two", "query three", "genes", "therapy",
               "diabetes risk", "wnt pathway", "survival curve", "sleep study", "gene panel"]
    for q in queries:
        for tk in (1, 5, 10, 20):
            for fs in ("rrf", "weighted", "simple"):
                cases.append(_case(f"td_okp_{n}", "tool_dispatch",
                                   {"args": {"query": q, "top_k": tk, "fusion_strategy": fs},
                                    "behavior": "ok", "max_retries": 2},
                                   {"success": True, "retryable": False, "retry_count": 0}))
                n += 1
    # 扩充：top_k 越界（>20 或 <1）=> 校验失败，不重试
    for tk in (0, -1, 21, 100):
        cases.append(_case(f"td_tkbad_{n}", "tool_dispatch",
                           {"args": {"query": "x", "top_k": tk}, "behavior": "ok", "max_retries": 2},
                           {"success": False, "retryable": False, "retry_count": 0}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 10. 会话生命周期
# ══════════════════════════════════════════════════════════════════

def gen_session_lifecycle() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # append N 轮 => turn_count == min(N, max_turns=20)
    for turns in (0, 1, 3, 5, 10, 19, 20, 21, 25, 40):
        cases.append(_case(f"sl_turns_{n}", "session_lifecycle",
                           {"scenario": "append_turns", "turns": turns},
                           {"turn_count": min(turns, 20)}))
        n += 1
    # 带 agent_trace 的轮次 => get_agent_trace 展平条数
    for turns, steps in [(1, 4), (2, 4), (3, 2), (5, 4)]:
        cases.append(_case(f"sl_trace_{n}", "session_lifecycle",
                           {"scenario": "agent_trace", "turns": turns, "steps_per_turn": steps},
                           {"flattened": turns * steps}))
        n += 1
    # 删除后 exists=False
    for turns in (0, 1, 3):
        cases.append(_case(f"sl_delete_{n}", "session_lifecycle",
                           {"scenario": "delete", "turns": turns},
                           {"exists_after_delete": False}))
        n += 1
    # 未知会话查询返回空/None
    for scen in ["get_unknown_history", "get_unknown_trace", "get_unknown_info"]:
        cases.append(_case(f"sl_unknown_{n}", "session_lifecycle",
                           {"scenario": scen}, {"empty": True}))
        n += 1
    # 纯 RAG 轮次（无 trace）=> get_agent_trace 为空
    for turns in (1, 2, 3):
        cases.append(_case(f"sl_ragonly_{n}", "session_lifecycle",
                           {"scenario": "rag_only", "turns": turns}, {"flattened": 0}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 11. Agent 状态与终止条件
# ══════════════════════════════════════════════════════════════════

def gen_agent_state() -> list[dict]:
    cases: list[dict] = []
    n = 0
    # 新建状态默认不终止
    for tk in (1, 5, 6, 8, 20):
        cases.append(_case(f"as_fresh_{n}", "agent_state",
                           {"scenario": "fresh", "top_k": tk, "max_iterations": 5},
                           {"terminate": False, "status": None}))
        n += 1
    # 达最大迭代
    for mi, ic in [(1, 1), (3, 3), (5, 5), (5, 6), (3, 10)]:
        cases.append(_case(f"as_maxiter_{n}", "agent_state",
                           {"scenario": "max_iter", "max_iterations": mi, "iteration_count": ic},
                           {"terminate": True, "status": "max_iterations_reached"}))
        n += 1
    # 未达最大迭代
    for mi, ic in [(5, 0), (5, 1), (5, 4), (10, 3), (3, 2)]:
        cases.append(_case(f"as_underiter_{n}", "agent_state",
                           {"scenario": "under_iter", "max_iterations": mi, "iteration_count": ic},
                           {"terminate": False, "status": None}))
        n += 1
    # 超时（构造 created_at 在很久以前，避免真实 sleep）
    for timeout in (1.0, 5.0, 30.0, 90.0):
        cases.append(_case(f"as_timeout_{n}", "agent_state",
                           {"scenario": "timeout", "timeout_seconds": timeout, "age_seconds": timeout + 100},
                           {"terminate": True, "status": "timeout"}))
        n += 1
    # 新建状态字段默认值
    for tk, fs in [(6, "rrf"), (8, "weighted"), (10, "simple")]:
        cases.append(_case(f"as_defaults_{n}", "agent_state",
                           {"scenario": "defaults", "top_k": tk, "fusion_strategy": fs},
                           {"top_k": tk, "fusion_strategy": fs, "status_initial": "planning"}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 12. 查询理解：同义词扩展（每个 SYNONYMS 词条一条）
# ══════════════════════════════════════════════════════════════════

def gen_query_synonym() -> list[dict]:
    from retrieval.medical_vocab import SYNONYMS
    cases: list[dict] = []
    for i, (term, syns) in enumerate(sorted(SYNONYMS.items())):
        cases.append(_case(f"qs_{i}", "query_synonym",
                           {"query": f"clinical guidelines for {term} management", "term": term},
                           {"synonyms": syns}))
    return cases


# ══════════════════════════════════════════════════════════════════
# 13. 查询理解：中文术语翻译（每个 ZH_TO_EN 词条一条）
# ══════════════════════════════════════════════════════════════════

def gen_query_zh_translate() -> list[dict]:
    from retrieval.medical_vocab import ZH_TO_EN
    cases: list[dict] = []
    keys = list(ZH_TO_EN.keys())
    # 只测"原子"键：若某个中文键包含另一个更短的中文键（如 "2型糖尿病" 含
    # "糖尿病"），翻译时可能被更短的键抢先部分替换，导致完整英文译名不出现——
    # 那是子串替换的固有行为，不适合做精确断言。过滤掉这类重叠键。
    atomic = [k for k in keys if not any(other != k and other in k for other in keys)]
    for i, zh in enumerate(sorted(atomic)):
        en = ZH_TO_EN[zh]
        cases.append(_case(f"zh_{i}", "query_zh_translate",
                           {"query": f"{zh}的临床研究", "en": en},
                           {"english": en.lower()}))
    return cases


# ══════════════════════════════════════════════════════════════════
# 14. BM25 分词属性（小写 / 去停用词 / 起始为字母 / 最短长度 2 / 无纯数字）
# ══════════════════════════════════════════════════════════════════

def gen_bm25_tokenize() -> list[dict]:
    cases: list[dict] = []
    texts = [
        "The patients had Type-2 Diabetes and MI in 2004!",
        "Kaplan-Meier survival analysis of cancer cohorts",
        "WNT/beta-catenin signaling pathway in tumors",
        "Metformin activates AMPK; a 500 mg dose was used",
        "A single-nucleotide polymorphism (SNP) at locus 9p21",
        "Sleep spindles and memory consolidation during REM",
        "The study of 1234 subjects across 5 sites in 2019",
        "COVID-19 vaccine efficacy in older adults over 65",
        "Randomized controlled trial with p < 0.05 significance",
        "Gene expression profiling using RNA-seq at day 7",
        "",
        "the a an of to in on with and or but",  # 纯停用词 -> 结果应为空
        "12345 6789 000",  # 纯数字 -> 结果应为空
        "Hypertension, hyperlipidemia, and obesity comorbidities",
        "CRISPR-Cas9 mediated knockout of TP53 in HeLa cells",
    ]
    for i, t in enumerate(texts):
        cases.append(_case(f"bm25_{i}", "bm25_tokenize", {"text": t}, {}))
    return cases


# ══════════════════════════════════════════════════════════════════
# 15. 格式校验：必需章节存在性判定
# ══════════════════════════════════════════════════════════════════

def gen_format_sections() -> list[dict]:
    from generation.prompt_templates import SECTION_HEADERS
    cases: list[dict] = []
    n = 0
    for lang in ("zh", "en"):
        headers = list(SECTION_HEADERS.get(lang, SECTION_HEADERS["zh"]).values())
        # 全部章节都在（每个标题独占一行）=> pass
        full = "\n".join(f"## {h}\n内容 content here." for h in headers)
        cases.append(_case(f"fmt_full_{n}", "format_sections",
                           {"text": full, "language": lang}, {"pass": True, "n_missing": 0}))
        n += 1
        # 缺其中一个章节 => 不 pass，missing 计数 = 1
        for drop_idx in range(len(headers)):
            kept = [h for j, h in enumerate(headers) if j != drop_idx]
            text = "\n".join(f"## {h}\n内容 content." for h in kept)
            cases.append(_case(f"fmt_drop_{n}", "format_sections",
                               {"text": text, "language": lang}, {"pass": False, "n_missing": 1}))
            n += 1
        # 正文里以子串形式出现标题词但不独占标题行 => 仍判为缺失（不误判）
        inline = "This paragraph merely mentions " + " and ".join(headers) + " within a sentence."
        cases.append(_case(f"fmt_inline_{n}", "format_sections",
                           {"text": inline, "language": lang}, {"pass": False, "n_missing": len(headers)}))
        n += 1
    return cases


# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════

_GENERATORS = [
    gen_where_filter_split,
    gen_where_filter_eval,
    gen_cache_key,
    gen_canonical_params,
    gen_query_abbrev,
    gen_query_synonym,
    gen_query_zh_translate,
    gen_query_time_filter,
    gen_citation_extract,
    gen_context_dedup,
    gen_tool_dispatch,
    gen_session_lifecycle,
    gen_agent_state,
    gen_bm25_tokenize,
    gen_format_sections,
]


def build_corpus() -> list[dict]:
    corpus: list[dict] = []
    for gen in _GENERATORS:
        corpus.extend(gen())
    return corpus


def category_counts(corpus: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in corpus:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    return counts


def write_corpus(path: Path = CORPUS_PATH) -> int:
    corpus = build_corpus()
    with path.open("w", encoding="utf-8") as f:
        for case in corpus:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    return len(corpus)


if __name__ == "__main__":
    corpus = build_corpus()
    counts = category_counts(corpus)
    n = write_corpus()
    print(f"已写入 {n} 条用例 -> {CORPUS_PATH}")
    for cat, cnt in sorted(counts.items()):
        print(f"  {cat:24s} {cnt}")
    print(f"  {'合计':24s} {n}")
