"""
tests/test_regression_suite.py — 全量功能回归测试套件（数据驱动，1000+ 条）

用例数据来自 tests/regression_corpus.py 的 build_corpus()（同时落盘为
tests/regression_corpus.jsonl 供人工查看/版本化）。这里把语料里的**每一条**
用例动态挂成一个独立的 test 方法（方法名 test_<case_id>），因此 unittest 会
如实报告 "Ran 1003 tests"（1001 条用例 + 2 条语料自检），每条用例单独计数、
单独报错，互不掩盖。

全部针对纯逻辑函数，不加载 GPU 模型、不连 Ollama、不连 Redis，因此整套
1000+ 条能在几秒内跑完（对比：真实调 LLM 的端到端测试每条 5-10s）。真正
调模型的少量冒烟用例在 tests/test_regression_llm_smoke.py，不在本套件内。

运行：
    PYTHONUTF8=1 python -m unittest tests.test_regression_suite            # 汇总
    PYTHONUTF8=1 python -m unittest tests.test_regression_suite -v         # 逐条
    PYTHONUTF8=1 python -m unittest tests.test_regression_suite -k where   # 按名字筛
"""

from __future__ import annotations

import logging
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tool_dispatch 的负例会故意触发校验失败/不可重试异常，其 ERROR/WARNING 日志是
# 预期行为而非测试问题——调高门槛避免刷屏，保持测试输出干净。
logging.getLogger("agent.tool_dispatcher").setLevel(logging.CRITICAL)

from tests.regression_corpus import build_corpus

# 被测的真实纯逻辑函数/类
from pmc_vector_index import _split_where_filter, _passes_range_conditions
from generation.cache import GenerationCache
from generation.citation_validator import CitationValidator
from generation.context_assembler import ContextAssembler
from generation.format_checker import FormatChecker
from agent.memory import _canonical_params
from agent.state import new_agent_state, should_terminate
from agent.tool_registry import ToolRegistry, ToolSpec
from agent.tool_dispatcher import ToolDispatcherEngine, RetryableError
from agent.retrieval_tool import RetrievalToolParams
from api.session import SessionManager
from retrieval.query_processor import MedicalQueryProcessor
from retrieval.bm25_index import tokenize, STOPWORDS

# 无状态/可复用的被测组件，进程内构造一次即可（均不加载 GPU 模型）
_QP = MedicalQueryProcessor()
_ASSEMBLER = ContextAssembler()
_CITATION = CitationValidator()
_FMT = FormatChecker()


# ══════════════════════════════════════════════════════════════════
# 单条用例 checker（每个都对一条 case 做真实断言）
# ══════════════════════════════════════════════════════════════════

def check_where_filter_split(tc, case):
    safe, ranges = _split_where_filter(case["input"]["where_filter"])
    pairs = sorted([[f, op] for (f, op, _t) in ranges])
    tc.assertEqual(pairs, case["expect"]["range_pairs"])
    tc.assertEqual(safe is None, case["expect"]["safe_is_none"])


def check_where_filter_eval(tc, case):
    conds = [tuple(rc) for rc in case["input"]["range_conditions"]]
    tc.assertEqual(_passes_range_conditions(case["input"]["metadata"], conds), case["expect"]["passes"])


def check_cache_key(tc, case):
    ka = GenerationCache.make_key(**case["input"]["parts_a"])
    kb = GenerationCache.make_key(**case["input"]["parts_b"])
    tc.assertEqual(ka == kb, case["expect"]["equal"])


def check_canonical_params(tc, case):
    ca = _canonical_params(case["input"]["a"])
    cb = _canonical_params(case["input"]["b"])
    tc.assertEqual(ca == cb, case["expect"]["equal"])


def check_query_abbrev(tc, case):
    result = _QP.process(case["input"]["query"])
    abbr = case["input"]["abbr"]
    tc.assertIn(abbr, result.abbreviations)
    tc.assertEqual(result.abbreviations[abbr], case["expect"]["expansions"])


def check_query_synonym(tc, case):
    result = _QP.process(case["input"]["query"])
    term = case["input"]["term"]
    tc.assertIn(term, result.synonyms)
    tc.assertEqual(result.synonyms[term], case["expect"]["synonyms"])


def check_query_zh_translate(tc, case):
    result = _QP.process(case["input"]["query"])
    blob = (result.cleaned + " " + result.keyword_query + " " + result.vector_query).lower()
    tc.assertIn(case["expect"]["english"], blob)


def check_query_time_filter(tc, case):
    filters = _QP.process(case["input"]["query"]).filters
    exp = case["expect"]
    if exp["kind"] == "none":
        tc.assertNotIn("pub_year", filters)
        tc.assertNotIn("$and", filters)
    elif exp["kind"] == "eq":
        tc.assertEqual(filters.get("pub_year"), {"$eq": exp["eq"]})
    elif exp["kind"] == "and_range":
        tc.assertIn("$and", filters)
        tc.assertEqual(filters["$and"][0]["pub_year"]["$gte"], exp["gte"])
        tc.assertEqual(filters["$and"][1]["pub_year"]["$lte"], exp["lte"])
    elif exp["kind"] == "and_lower":
        tc.assertIn("$and", filters)
        tc.assertEqual(filters["$and"][0]["pub_year"]["$gte"], exp["gte"])
    else:
        tc.fail(f"未知 time filter kind {exp['kind']}")


def check_citation_extract(tc, case):
    v = _CITATION.validate(case["input"]["text"], set(case["input"]["valid_ids"]))
    exp = case["expect"]
    tc.assertEqual(v["cited_ids"], exp["cited_ids"])
    tc.assertEqual(v["has_invalid"], exp["has_invalid"])
    tc.assertEqual(v["missing_citations"], exp["missing"])
    tc.assertEqual(v["pass"], exp["pass"])


def check_context_dedup(tc, case):
    out = _ASSEMBLER.assemble(case["input"]["docs"])
    selected = out["selected_chunks"]
    tc.assertEqual(len(selected), case["expect"]["n_selected"])
    if case["expect"].get("contiguous"):
        ids = [c.metadata["_citation_id"] for c in selected]
        tc.assertEqual(ids, list(range(1, len(selected) + 1)))
    if "last_chunk_id" in case["expect"]:
        tc.assertEqual(selected[-1].chunk_id, case["expect"]["last_chunk_id"])


def check_tool_dispatch(tc, case):
    inp, exp = case["input"], case["expect"]
    behavior = inp["behavior"]
    state = {"n": 0}

    def handler(query, top_k=8, fusion_strategy="rrf", where_filter=None):
        state["n"] += 1
        if behavior == "ok":
            return {"results": []}
        if behavior == "raise_retryable":
            raise RetryableError("transient")
        if behavior == "raise_value":
            raise ValueError("bug")
        if behavior == "recover":
            if state["n"] <= inp["succeed_on"]:
                raise RetryableError("transient")
            return {"results": []}
        raise AssertionError(behavior)

    registry = ToolRegistry()
    registry.register(ToolSpec(name="retrieval", handler=handler,
                               param_schema=RetrievalToolParams, max_retries=inp["max_retries"]))
    dispatcher = ToolDispatcherEngine(registry, backoff_base_seconds=0.0)
    call = dispatcher.dispatch("retrieval", inp["args"])
    tc.assertEqual(call.success, exp["success"])
    tc.assertEqual(call.retryable, exp["retryable"])
    tc.assertEqual(call.retry_count, exp["retry_count"])


def check_session_lifecycle(tc, case):
    inp, exp = case["input"], case["expect"]
    sm = SessionManager()
    scen = inp["scenario"]
    if scen == "append_turns":
        sid, _ = sm.create_session()
        for i in range(inp["turns"]):
            sm.append_turn(sid, f"q{i}", f"a{i}")
        tc.assertEqual(sm.get_session_info(sid)["turn_count"], exp["turn_count"])
    elif scen == "agent_trace":
        sid, _ = sm.create_session()
        trace = [{"step": f"s{j}"} for j in range(inp["steps_per_turn"])]
        for i in range(inp["turns"]):
            sm.append_turn(sid, f"q{i}", f"a{i}", agent_trace=trace)
        tc.assertEqual(len(sm.get_agent_trace(sid)), exp["flattened"])
    elif scen == "delete":
        sid, _ = sm.create_session()
        for i in range(inp["turns"]):
            sm.append_turn(sid, f"q{i}", f"a{i}")
        sm.delete_session(sid)
        tc.assertEqual(sm.exists(sid), exp["exists_after_delete"])
    elif scen == "get_unknown_history":
        tc.assertEqual(sm.get_history("nope"), [])
    elif scen == "get_unknown_trace":
        tc.assertEqual(sm.get_agent_trace("nope"), [])
    elif scen == "get_unknown_info":
        tc.assertIsNone(sm.get_session_info("nope"))
    elif scen == "rag_only":
        sid, _ = sm.create_session()
        for i in range(inp["turns"]):
            sm.append_turn(sid, f"q{i}", f"a{i}")
        tc.assertEqual(len(sm.get_agent_trace(sid)), exp["flattened"])
    else:
        tc.fail(f"未知 session 场景 {scen}")


def check_agent_state(tc, case):
    inp, exp = case["input"], case["expect"]
    scen = inp["scenario"]
    if scen == "fresh":
        st = new_agent_state(query="q", top_k=inp["top_k"], max_iterations=inp["max_iterations"])
        terminate, status = should_terminate(st)
        tc.assertFalse(terminate)
        tc.assertIsNone(status)
    elif scen in ("max_iter", "under_iter"):
        st = new_agent_state(query="q", max_iterations=inp["max_iterations"])
        st["iteration_count"] = inp["iteration_count"]
        terminate, status = should_terminate(st)
        tc.assertEqual(terminate, exp["terminate"])
        if exp["status"] is None:
            tc.assertIsNone(status)
        else:
            tc.assertEqual(status.value, exp["status"])
    elif scen == "timeout":
        st = new_agent_state(query="q", timeout_seconds=inp["timeout_seconds"])
        st["created_at"] = time.time() - inp["age_seconds"]
        terminate, status = should_terminate(st)
        tc.assertTrue(terminate)
        tc.assertEqual(status.value, exp["status"])
    elif scen == "defaults":
        st = new_agent_state(query="q", top_k=inp["top_k"], fusion_strategy=inp["fusion_strategy"])
        tc.assertEqual(st["top_k"], exp["top_k"])
        tc.assertEqual(st["fusion_strategy"], exp["fusion_strategy"])
        tc.assertEqual(st["execution_status"].value, exp["status_initial"])
    else:
        tc.fail(f"未知 agent_state 场景 {scen}")


def check_bm25_tokenize(tc, case):
    for t in tokenize(case["input"]["text"]):
        tc.assertEqual(t, t.lower(), f"未小写化: {t!r}")
        tc.assertNotIn(t, STOPWORDS, f"停用词未过滤: {t!r}")
        tc.assertGreaterEqual(len(t), 2, f"token 过短: {t!r}")
        tc.assertTrue(t[0].isalpha(), f"token 未以字母开头: {t!r}")


def check_format_sections(tc, case):
    r = _FMT.check_required_sections(case["input"]["text"], case["input"]["language"])
    tc.assertEqual(r["pass"], case["expect"]["pass"])
    tc.assertEqual(len(r["missing"]), case["expect"]["n_missing"])


CHECKERS = {
    "where_filter_split": check_where_filter_split,
    "where_filter_eval": check_where_filter_eval,
    "cache_key": check_cache_key,
    "canonical_params": check_canonical_params,
    "query_abbrev": check_query_abbrev,
    "query_synonym": check_query_synonym,
    "query_zh_translate": check_query_zh_translate,
    "query_time_filter": check_query_time_filter,
    "citation_extract": check_citation_extract,
    "context_dedup": check_context_dedup,
    "tool_dispatch": check_tool_dispatch,
    "session_lifecycle": check_session_lifecycle,
    "agent_state": check_agent_state,
    "bm25_tokenize": check_bm25_tokenize,
    "format_sections": check_format_sections,
}


# ══════════════════════════════════════════════════════════════════
# 动态挂载：语料里每一条用例 -> 一个独立 test 方法
# ══════════════════════════════════════════════════════════════════

_CORPUS = build_corpus()


class TestRegressionCorpus(unittest.TestCase):
    def test_corpus_size_at_least_1000(self):
        self.assertGreaterEqual(len(_CORPUS), 1000, f"语料仅 {len(_CORPUS)} 条，未达 1000")

    def test_corpus_ids_unique(self):
        ids = [c["id"] for c in _CORPUS]
        self.assertEqual(len(ids), len(set(ids)), "存在重复的用例 id")


def _make_test(case):
    checker = CHECKERS.get(case["category"])

    def _test(self):
        if checker is None:
            self.fail(f"无对应 checker 的类别：{case['category']}")
        checker(self, case)

    _test.__doc__ = f"[{case['category']}] {case['id']}"
    return _test


_seen: set[str] = set()
for _case in _CORPUS:
    _name = f"test_{_case['id']}"
    if _name in _seen:  # 语料 id 唯一性另有守卫测试，这里防御性避免覆盖
        _name = f"{_name}_dup"
    _seen.add(_name)
    setattr(TestRegressionCorpus, _name, _make_test(_case))


if __name__ == "__main__":
    unittest.main()
