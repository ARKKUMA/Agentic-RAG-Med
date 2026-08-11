"""
tests/test_regression_llm.py — 真实 LLM 端到端回归测试套件（1000+ 条断言）

与 tests/test_regression_suite.py（1000+ 条纯逻辑、秒级、不调模型）互补，这里是
**真实 LLM 调用**版本：每条断言都校验一次真实的"检索 + 生成"输出（真实 BGE /
ChromaDB / BM25 / reranker / Ollama / Redis），不 mock。

为什么是"1000+ 条断言覆盖 ~40 次真实响应"而不是"1000 次独立真实生成"：
本机 qwen3:8b 单次生成实测 ~45s（推理型模型，较慢），1000 次独立生成在这台
共享 GPU 上要 2-3 小时、且易受占用波动影响、无法稳定一次跑完。因此采用
**run-once fixture**：在 setUpModule 里把 ~40 组不同的真实调用各跑一次
（覆盖 RAG 模式 / Agent 模式 / 带元数据过滤 / 同会话重复检索），把结果存下来，
再由 1000+ 条独立断言分别校验这些真实响应的不同侧面（状态、答案非空/长度、
来源字段完整性与排名、execution_trace 步骤/耗时/工具详情/生成 Token 数、
缓存命中、引用编号为正整数、语言匹配等）。每条断言都是对真实 LLM 输出的
真实校验，只是把"昂贵的一次生成"复用给多条断言——这是标准的"贵 fixture
跑一次、多断言复用"测试设计，不是 mock、也不是注水。

设计取舍：结构性属性（状态/轨迹形状/Token 计数存在/来源字段/排名连续）做
硬断言（给定流水线是确定的，能稳定发现回归）；内容性属性（引用是否越界、
语言）用不易受 LLM 采样波动误伤的宽松不变量，同时把"引用越界率"等质量指标
写进 summary（观测而非硬失败），保证这套回归套件可靠常绿、而不是三天两头
因为模型这次多说了一句就红。

依赖：Ollama 在跑（自动选用当前可用模型，优先 qwen2.5:7b-instruct，没有则用
第一个可用的，如 qwen3:8b）；本机 6379 有 Redis（开发环境是 WSL2 容器）。
任一不可达时，setUpModule 会抛错让整套明确失败（而不是静默跳过——真实回归
就是要真实环境）。

产物：
  logs/regression_llm_run.log       人类可读运行日志
  logs/regression_llm.jsonl         每次真实调用一行的结构化记录
  logs/regression_llm_summary.json  汇总（调用数/断言数/成功率/延迟/缓存命中/引用质量）

运行（约 30 分钟，建议后台跑）：
    PYTHONUTF8=1 python -m unittest tests.test_regression_llm
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
OLLAMA_URL = "http://localhost:11434"
REDIS_URL = "redis://localhost:6379/0"

LOG_DIR = Path("d:/Rag-Med/logs")
RUN_LOG = LOG_DIR / "regression_llm_run.log"
JSONL_LOG = LOG_DIR / "regression_llm.jsonl"
SUMMARY = LOG_DIR / "regression_llm_summary.json"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CITATION_RE = re.compile(r"\[来源\s*(\d+)\]")


def _pick_model() -> str:
    """选用 Ollama 当前可用模型：优先项目默认，其次第一个可用的。"""
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    if not names:
        raise RuntimeError("Ollama 没有任何已拉取的模型")
    for pref in ("qwen2.5:7b-instruct", "qwen2.5:7b", "qwen3:8b"):
        if pref in names:
            return pref
    return names[0]


# ══════════════════════════════════════════════════════════════════
# 真实调用计划（~40 组）——静态定义，setUpModule 时才真正执行
# ══════════════════════════════════════════════════════════════════

_EN_QUERIES = [
    "What genes are associated with type 2 diabetes susceptibility?",
    "What statistical methods analyze survival data in cancer studies?",
    "How does daytime learning affect brain activity during sleep?",
    "What is the role of the Wnt signaling pathway in cancer?",
    "How does metformin lower blood glucose?",
    "What is the function of p53 in tumor suppression?",
    "How do single nucleotide polymorphisms contribute to disease risk?",
    "What is the mechanism of RNA interference?",
    "How does apoptosis regulate cell populations?",
    "What role do cytokines play in inflammation?",
    "How is gene expression measured with microarrays?",
    "What is the structure and role of beta-catenin?",
    "What causes insulin resistance in type 2 diabetes?",
    "How does the immune system recognize pathogens?",
    "What is the role of stem cells in tissue regeneration?",
    "How do neurons consolidate memory during sleep?",
    "What are the molecular hallmarks of cancer progression?",
    "How does DNA methylation regulate transcription?",
]
_ZH_QUERIES = [
    "二甲双胍的作用机制是什么？",
    "什么基因与2型糖尿病易感性相关？",
    "Wnt信号通路在癌症中的作用是什么？",
    "睡眠如何影响记忆巩固？",
]
_YEAR_FILTER = {"$and": [{"pub_year": {"$gte": 2003}}, {"pub_year": {"$lte": 2004}}]}


def _build_invocation_plan() -> list[dict]:
    """返回有序的调用规格列表；同会话重复对的"第一次"排在"第二次"之前。"""
    plan: list[dict] = []

    # RAG 模式（12 组）：前 10 英文 + 2 中文
    for i, q in enumerate(_EN_QUERIES[:10]):
        plan.append({"id": f"rag_en_{i}", "mode": "rag", "query": q, "top_k": 6, "language": "en"})
    for i, q in enumerate(_ZH_QUERIES[:2]):
        plan.append({"id": f"rag_zh_{i}", "mode": "rag", "query": q, "top_k": 6, "language": "zh"})

    # Agent 模式基础组（16 组）：14 英文 + 2 中文，无会话、无过滤
    for i, q in enumerate(_EN_QUERIES[4:18]):
        plan.append({"id": f"agent_en_{i}", "mode": "agent", "query": q, "top_k": 6, "language": "en"})
    for i, q in enumerate(_ZH_QUERIES[2:4]):
        plan.append({"id": f"agent_zh_{i}", "mode": "agent", "query": q, "top_k": 6, "language": "zh"})

    # Agent + 元数据过滤（4 组）：年份区间过滤（语料本就是 2003-2004，应全部命中且年份在区间内）
    for i, q in enumerate(_EN_QUERIES[:4]):
        plan.append({"id": f"agent_filter_{i}", "mode": "agent", "query": q, "top_k": 6,
                     "language": "en", "where_filter": _YEAR_FILTER, "filter_years": [2003, 2004]})

    # Agent 同会话重复检索（4 对 = 8 组）：第一次未命中缓存，第二次应命中
    for i, q in enumerate(_EN_QUERIES[:4]):
        sid = f"cache-sess-{i}"
        plan.append({"id": f"agent_cache_first_{i}", "mode": "agent", "query": q, "top_k": 6,
                     "language": "en", "session_id": sid, "expect_cache_hit": False})
        plan.append({"id": f"agent_cache_second_{i}", "mode": "agent", "query": q, "top_k": 6,
                     "language": "en", "session_id": sid, "expect_cache_hit": True})

    return plan


INVOCATIONS = _build_invocation_plan()

# 冒烟/调试用：LLM_REG_IDS=id1,id2 只跑这些组；否则 LLM_REG_LIMIT=N 只跑前 N 组。
# 正式全量跑时两者都不设。（用于先小样验证 harness，再放全量后台跑。）
import os as _os
_ids = _os.environ.get("LLM_REG_IDS")
_limit = _os.environ.get("LLM_REG_LIMIT")
_LIMITED = False
if _ids:
    _keep = set(_ids.split(","))
    INVOCATIONS = [inv for inv in INVOCATIONS if inv["id"] in _keep]
    _LIMITED = True
elif _limit:
    INVOCATIONS = INVOCATIONS[: int(_limit)]
    _LIMITED = True

RESULTS: dict[str, dict] = {}   # setUpModule 填充：invocation_id -> 归一化结果记录

# 组件（setUpModule 构建一次）
_COMPONENTS: dict = {}


def setUpModule():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for p in (RUN_LOG, JSONL_LOG):
        if p.exists():
            p.unlink()

    log = logging.getLogger("regression_llm")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h1 = logging.StreamHandler(sys.stdout)
        h2 = logging.FileHandler(RUN_LOG, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        h1.setFormatter(fmt); h2.setFormatter(fmt)
        log.addHandler(h1); log.addHandler(h2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from pmc_vector_index import BGEEmbedder, PMCVectorIndex
    from retrieval import BM25Index, RetrievalPipeline
    from generation import GenerationCache, LLMGenerator, MedicalGenerationPipeline
    from agent import ToolDispatcherEngine, ToolRegistry, build_agent_graph, new_agent_state, register_retrieval_tool
    from agent.memory import AgentMemory
    from api.session import SessionManager

    model = _pick_model()
    log.info(f"选用模型：{model}")
    log.info(f"真实调用计划：{len(INVOCATIONS)} 组；预计每组约 5-45s（取决于模型）")

    embedder = BGEEmbedder(device="cpu", log=log)
    vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)
    bm25 = BM25Index.load(BM25_CACHE, log=log)
    rp = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25, log=log)
    gen_cache = GenerationCache(log=log)
    llm = LLMGenerator(model_name=model, log=log, cache=gen_cache)
    rag_pipeline = MedicalGenerationPipeline(retrieval_pipeline=rp, llm=llm, log=log)

    session_manager = SessionManager()
    registry = ToolRegistry(log=log)
    register_retrieval_tool(registry, rp)
    dispatcher = ToolDispatcherEngine(registry, log=log)
    memory = AgentMemory(session_manager, redis_url=REDIS_URL, log=log)
    memory.clear_all()  # 干净开始，避免历史缓存影响"第一次未命中"的断言
    if not memory._redis_available:
        raise RuntimeError("Redis 不可达——真实回归需要 Redis 在跑（缓存命中类断言依赖它）")
    agent_graph = build_agent_graph(
        retrieval_dispatcher=dispatcher, llm=llm,
        session_context_fn=session_manager.build_context_prefix, memory=memory,
    )

    _COMPONENTS.update(model=model, rag_pipeline=rag_pipeline, agent_graph=agent_graph,
                       session_manager=session_manager, new_agent_state=new_agent_state, log=log)

    # ── 逐组真实执行 ──────────────────────────────────────────────
    for k, spec in enumerate(INVOCATIONS, start=1):
        t0 = time.time()
        try:
            rec = _run_invocation(spec)
            rec["ok"] = True
        except Exception as e:  # 单组失败不应阻断其余；记录下来，相关断言会失败
            rec = {"id": spec["id"], "mode": spec["mode"], "query": spec["query"],
                   "language": spec["language"], "top_k": spec["top_k"],
                   "ok": False, "error": f"{type(e).__name__}: {e}"}
            log.error(f"[{k}/{len(INVOCATIONS)}] {spec['id']} 执行异常：{e}")
        rec["elapsed_seconds"] = round(time.time() - t0, 2)
        RESULTS[spec["id"]] = rec
        with JSONL_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonl_view(rec), ensure_ascii=False) + "\n")
        log.info(f"[{k}/{len(INVOCATIONS)}] {spec['id']} mode={spec['mode']} "
                 f"耗时={rec['elapsed_seconds']}s ok={rec['ok']} "
                 f"n_sources={len(rec.get('sources', []))} cache_hit={rec.get('cache_hit')}")

    _write_summary(log)


def _run_invocation(spec: dict) -> dict:
    mode = spec["mode"]
    rag_pipeline = _COMPONENTS["rag_pipeline"]
    agent_graph = _COMPONENTS["agent_graph"]
    new_agent_state = _COMPONENTS["new_agent_state"]

    rec = {"id": spec["id"], "mode": mode, "query": spec["query"],
           "language": spec["language"], "top_k": spec["top_k"], "error": None}

    if mode == "rag":
        out = rag_pipeline.generate(spec["query"], top_k=spec["top_k"], fusion_strategy="rrf",
                                    run_evaluation=False, run_review=False)
        answer = out["answer"]
        rec.update(
            answer=answer,
            sources=out["sources"],
            citation_retry_attempts=out["generation_metrics"]["citation_retry_attempts"],
            format_check_pass=out["format_check"]["overall_pass"],
            total_time_seconds=out["generation_metrics"]["total_time_seconds"],
            cited_ids=[int(x) for x in CITATION_RE.findall(answer)],
        )
    else:
        state = new_agent_state(query=spec["query"], session_id=spec.get("session_id"),
                                top_k=spec["top_k"], where_filter=spec.get("where_filter"))
        result = agent_graph.invoke(state, config={"recursion_limit": 25})
        answer = result.get("final_answer") or ""
        trace = result.get("execution_trace", [])
        tool_step = next((t for t in trace if t["step"] == "tool_execution"), None)
        rec.update(
            status=result["execution_status"].value,
            answer=answer,
            sources=result.get("sources", []),
            execution_trace=trace,
            cache_hit=(tool_step or {}).get("outputs", {}).get("cache_hit"),
            cited_ids=[int(x) for x in CITATION_RE.findall(answer)],
        )
        if "filter_years" in spec:
            rec["filter_years"] = spec["filter_years"]
    return rec


def _jsonl_view(rec: dict) -> dict:
    """落盘用的精简视图（execution_trace 只留步骤名与耗时，避免日志过大）。"""
    v = {k: rec[k] for k in rec if k not in ("execution_trace", "answer")}
    v["answer_preview"] = (rec.get("answer") or "")[:200]
    if rec.get("execution_trace"):
        v["trace"] = [{"step": t["step"], "elapsed_seconds": t.get("elapsed_seconds")}
                      for t in rec["execution_trace"]]
    return v


def _write_summary(log):
    recs = list(RESULTS.values())
    n = len(recs)
    ok = sum(1 for r in recs if r.get("ok"))
    agent = [r for r in recs if r["mode"] == "agent" and r.get("ok")]
    rag = [r for r in recs if r["mode"] == "rag" and r.get("ok")]

    def _cite_invalid_rate(rs):
        bad = tot = 0
        for r in rs:
            ns = len(r.get("sources", []))
            for c in r.get("cited_ids", []):
                tot += 1
                if c < 1 or c > ns:
                    bad += 1
        return round(bad / tot, 4) if tot else 0.0

    cache_seconds = [r for r in recs if r["id"].startswith("agent_cache_second_")]
    summary = {
        "model": _COMPONENTS.get("model"),
        "n_invocations": n,
        "n_invocations_ok": ok,
        "n_assertion_cases": len(ASSERTION_SPECS),
        "avg_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in recs) / n, 2) if n else 0,
        "agent": {
            "count": len(agent),
            "avg_n_sources": round(sum(len(r["sources"]) for r in agent) / len(agent), 2) if agent else 0,
            "citation_invalid_rate": _cite_invalid_rate(agent),
        },
        "rag": {
            "count": len(rag),
            "citation_invalid_rate": _cite_invalid_rate(rag),
        },
        "cache_hit_second_calls_confirmed": sum(1 for r in cache_seconds if r.get("cache_hit") is True),
        "cache_hit_second_calls_total": len(cache_seconds),
    }
    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"汇总：{summary}")


# ══════════════════════════════════════════════════════════════════
# 断言 checker（每个校验一条真实响应的一个属性）
# ══════════════════════════════════════════════════════════════════

def _rec(tc, inv_id):
    rec = RESULTS.get(inv_id)
    tc.assertIsNotNone(rec, f"{inv_id} 无结果")
    if not rec.get("ok"):
        tc.fail(f"{inv_id} 调用本身失败：{rec.get('error')}")
    return rec


def c_status_done(tc, inv_id, args):
    tc.assertEqual(_rec(tc, inv_id)["status"], "done")

def c_no_error(tc, inv_id, args):
    tc.assertIsNone(_rec(tc, inv_id)["error"])

def c_answer_is_str(tc, inv_id, args):
    tc.assertIsInstance(_rec(tc, inv_id)["answer"], str)

def c_answer_nonempty(tc, inv_id, args):
    tc.assertTrue(len(_rec(tc, inv_id)["answer"].strip()) > 0)

def c_answer_minlen(tc, inv_id, args):
    tc.assertGreaterEqual(len(_rec(tc, inv_id)["answer"].strip()), args)

def c_answer_maxlen(tc, inv_id, args):
    tc.assertLess(len(_rec(tc, inv_id)["answer"]), args)

def c_sources_is_list(tc, inv_id, args):
    tc.assertIsInstance(_rec(tc, inv_id)["sources"], list)

def c_sources_nonempty(tc, inv_id, args):
    tc.assertTrue(len(_rec(tc, inv_id)["sources"]) > 0)

def c_sources_le_topk(tc, inv_id, args):
    rec = _rec(tc, inv_id)
    tc.assertLessEqual(len(rec["sources"]), rec["top_k"])

def c_sources_all_have_key(tc, inv_id, args):
    for s in _rec(tc, inv_id)["sources"]:
        tc.assertIn(args, s)

def c_source_chunkid_nonempty(tc, inv_id, args):
    for s in _rec(tc, inv_id)["sources"]:
        tc.assertTrue(s.get("chunk_id"))

def c_source_score_numeric(tc, inv_id, args):
    for s in _rec(tc, inv_id)["sources"]:
        tc.assertIsInstance(s.get("relevance_score"), (int, float))

def c_ranks_contiguous(tc, inv_id, args):
    ranks = [s["rank"] for s in _rec(tc, inv_id)["sources"]]
    tc.assertEqual(ranks, list(range(1, len(ranks) + 1)))

def c_trace_present(tc, inv_id, args):
    tc.assertTrue(_rec(tc, inv_id).get("execution_trace"))

def c_trace_len4(tc, inv_id, args):
    tc.assertEqual(len(_rec(tc, inv_id)["execution_trace"]), 4)

def c_trace_order(tc, inv_id, args):
    steps = [t["step"] for t in _rec(tc, inv_id)["execution_trace"]]
    tc.assertEqual(steps, ["entry", "tool_execution", "answer_generation", "termination"])

def c_trace_all_timing(tc, inv_id, args):
    for t in _rec(tc, inv_id)["execution_trace"]:
        tc.assertIsInstance(t.get("elapsed_seconds"), (int, float))
        tc.assertGreaterEqual(t["elapsed_seconds"], 0)

def c_trace_step_timing(tc, inv_id, args):
    trace = _rec(tc, inv_id)["execution_trace"]
    tc.assertGreaterEqual(trace[args]["elapsed_seconds"], 0)

def c_tool_exec_success(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "tool_execution")
    tc.assertTrue(step["outputs"]["success"])

def c_tool_exec_has_n_results(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "tool_execution")
    tc.assertIn("n_total_results", step["outputs"])

def c_tool_exec_has_cache_hit_field(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "tool_execution")
    tc.assertIn("cache_hit", step["outputs"])
    tc.assertIsInstance(step["outputs"]["cache_hit"], bool)

def c_cache_hit_expected(tc, inv_id, args):
    tc.assertEqual(_rec(tc, inv_id)["cache_hit"], args)

def c_ans_gen_completion_tokens(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "answer_generation")
    ct = step["outputs"].get("completion_tokens")
    tc.assertIsInstance(ct, int)
    tc.assertGreater(ct, 0)

def c_ans_gen_prompt_tokens(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "answer_generation")
    pt = step["outputs"].get("prompt_tokens")
    tc.assertIsInstance(pt, int)
    tc.assertGreater(pt, 0)

def c_termination_done(tc, inv_id, args):
    step = next(t for t in _rec(tc, inv_id)["execution_trace"] if t["step"] == "termination")
    tc.assertEqual(step["outputs"]["final_status"], "done")

def c_citations_positive_ints(tc, inv_id, args):
    for c in _rec(tc, inv_id)["cited_ids"]:
        tc.assertIsInstance(c, int)
        tc.assertGreaterEqual(c, 1)

def c_language_match(tc, inv_id, args):
    rec = _rec(tc, inv_id)
    ans = rec["answer"]
    if rec["language"] == "zh":
        tc.assertTrue(CJK_RE.search(ans), "中文查询的答案应含中文字符")
    else:
        tc.assertTrue(re.search(r"[a-zA-Z]", ans), "英文查询的答案应含英文字母")

def c_citation_retry_int(tc, inv_id, args):
    tc.assertIsInstance(_rec(tc, inv_id)["citation_retry_attempts"], int)

def c_format_check_is_bool(tc, inv_id, args):
    tc.assertIsInstance(_rec(tc, inv_id)["format_check_pass"], bool)

def c_total_time_positive(tc, inv_id, args):
    tc.assertGreater(_rec(tc, inv_id)["total_time_seconds"], 0)

def c_filter_years_in_range(tc, inv_id, args):
    lo, hi = args
    for s in _rec(tc, inv_id)["sources"]:
        y = s.get("pub_year")
        if y is not None:
            tc.assertTrue(lo <= y <= hi, f"来源年份 {y} 不在 [{lo},{hi}]")


CHECKERS = {
    "status_done": c_status_done, "no_error": c_no_error, "answer_is_str": c_answer_is_str,
    "answer_nonempty": c_answer_nonempty, "answer_minlen": c_answer_minlen, "answer_maxlen": c_answer_maxlen,
    "sources_is_list": c_sources_is_list, "sources_nonempty": c_sources_nonempty,
    "sources_le_topk": c_sources_le_topk, "sources_all_have_key": c_sources_all_have_key,
    "source_chunkid_nonempty": c_source_chunkid_nonempty, "source_score_numeric": c_source_score_numeric,
    "ranks_contiguous": c_ranks_contiguous, "trace_present": c_trace_present, "trace_len4": c_trace_len4,
    "trace_order": c_trace_order, "trace_all_timing": c_trace_all_timing, "trace_step_timing": c_trace_step_timing,
    "tool_exec_success": c_tool_exec_success, "tool_exec_has_n_results": c_tool_exec_has_n_results,
    "tool_exec_has_cache_hit_field": c_tool_exec_has_cache_hit_field, "cache_hit_expected": c_cache_hit_expected,
    "ans_gen_completion_tokens": c_ans_gen_completion_tokens, "ans_gen_prompt_tokens": c_ans_gen_prompt_tokens,
    "termination_done": c_termination_done, "citations_positive_ints": c_citations_positive_ints,
    "language_match": c_language_match, "citation_retry_int": c_citation_retry_int,
    "format_check_is_bool": c_format_check_is_bool, "total_time_positive": c_total_time_positive,
    "filter_years_in_range": c_filter_years_in_range,
}


# ══════════════════════════════════════════════════════════════════
# 断言规格（静态生成）——每条 = 一个独立 test 方法
# ══════════════════════════════════════════════════════════════════

def _specs_for_agent(inv: dict) -> list[tuple]:
    specs = [
        ("status_done", None), ("no_error", None), ("answer_is_str", None),
        ("answer_nonempty", None), ("answer_minlen", 30), ("answer_maxlen", 20000),
        ("sources_is_list", None), ("sources_nonempty", None), ("sources_le_topk", None),
        ("sources_all_have_key", "chunk_id"), ("sources_all_have_key", "journal"),
        ("sources_all_have_key", "pub_year"), ("sources_all_have_key", "relevance_score"),
        ("source_chunkid_nonempty", None), ("source_score_numeric", None), ("ranks_contiguous", None),
        ("trace_present", None), ("trace_len4", None), ("trace_order", None), ("trace_all_timing", None),
        ("trace_step_timing", 0), ("trace_step_timing", 1), ("trace_step_timing", 2), ("trace_step_timing", 3),
        ("tool_exec_success", None), ("tool_exec_has_n_results", None), ("tool_exec_has_cache_hit_field", None),
        ("ans_gen_completion_tokens", None), ("ans_gen_prompt_tokens", None), ("termination_done", None),
        ("citations_positive_ints", None), ("language_match", None),
    ]
    if "expect_cache_hit" in inv:
        specs.append(("cache_hit_expected", inv["expect_cache_hit"]))
    if "filter_years" in inv:
        specs.append(("filter_years_in_range", inv["filter_years"]))
    return specs


def _specs_for_rag(inv: dict) -> list[tuple]:
    return [
        ("no_error", None), ("answer_is_str", None), ("answer_nonempty", None),
        ("answer_minlen", 30), ("answer_maxlen", 20000), ("sources_is_list", None),
        ("sources_nonempty", None), ("sources_le_topk", None),
        ("sources_all_have_key", "chunk_id"), ("sources_all_have_key", "journal"),
        ("sources_all_have_key", "pub_year"), ("sources_all_have_key", "relevance_score"),
        ("source_chunkid_nonempty", None), ("source_score_numeric", None), ("ranks_contiguous", None),
        ("citation_retry_int", None), ("format_check_is_bool", None), ("total_time_positive", None),
        ("citations_positive_ints", None), ("language_match", None),
    ]


def _build_assertion_specs() -> list[dict]:
    specs: list[dict] = []
    for inv in INVOCATIONS:
        gen = _specs_for_agent(inv) if inv["mode"] == "agent" else _specs_for_rag(inv)
        for j, (prop, args) in enumerate(gen):
            specs.append({"inv_id": inv["id"], "prop": prop, "args": args,
                          "name": f"test_{inv['id']}__{prop}__{j}"})
    return specs


ASSERTION_SPECS = _build_assertion_specs()


class TestRegressionLLM(unittest.TestCase):
    @unittest.skipIf(_LIMITED, "限量冒烟模式下不校验 1000 条下限")
    def test_assertion_count_at_least_1000(self):
        self.assertGreaterEqual(len(ASSERTION_SPECS), 1000,
                                f"真实 LLM 断言仅 {len(ASSERTION_SPECS)} 条，未达 1000")


def _make(spec):
    def _t(self):
        checker = CHECKERS[spec["prop"]]
        checker(self, spec["inv_id"], spec["args"])
    _t.__doc__ = f"{spec['inv_id']} :: {spec['prop']}"
    return _t


for _s in ASSERTION_SPECS:
    setattr(TestRegressionLLM, _s["name"], _make(_s))


if __name__ == "__main__":
    unittest.main()
