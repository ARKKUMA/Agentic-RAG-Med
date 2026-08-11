"""
test_agent_pipeline_week2.py — Agent 第 2 周真实端到端验证：
  上下文缓存记忆 + 基础 Agent 原型联调 + 兼容性/性能回归

用真实的 BGE 嵌入模型 / ChromaDB / BM25 / reranker / Ollama / 本机 Redis 跑：

  第 1 部分：多组医学查询功能测试（检索相关性/答案质量/引用/来源追溯/执行轨迹）
  第 2 部分：缓存与去重验证（同会话重复检索被拦截、跨轮次文献去重计数、手动清除）
  第 3 部分：元数据过滤验证（显式 where_filter + 查询文本自动提取两条路径）
  第 4 部分：异常兜底验证（工具调用失败/参数非法/重试耗尽场景不崩溃）
  第 5 部分：性能对比（RAG vs Agent 延迟、缓存命中 vs 未命中延迟、内存增量、并发吞吐）

生成日志：
  logs/agent_pipeline_week2_run.log        — 人类可读运行日志
  logs/agent_pipeline_week2.jsonl          — 结构化日志（功能测试部分，每条查询一行）
  logs/agent_pipeline_week2_summary.json   — 汇总指标（功能 + 缓存 + 过滤 + 异常 + 性能）

单元测试（mock，不依赖真实模型/Redis）见 tests/test_agent.py、tests/test_agent_memory.py；
API 层集成测试见 tests/test_api.py::TestAgentModeEndToEnd。这里是"确实真实跑起来了"的
最后一道验证，同时是本周任务书要求的"全量功能回归测试 + 性能对比数据 + Agent 原型功能
测试报告"的数据来源。
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import psutil

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline
from generation import GenerationCache, LLMGenerator, MedicalGenerationPipeline
from api.session import SessionManager
from agent import ToolDispatcherEngine, ToolRegistry, build_agent_graph, new_agent_state, register_retrieval_tool
from agent.memory import AgentMemory
from agent.tool_dispatcher import RetryableError

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "agent_pipeline_week2_run.log"
JSONL_LOG_PATH = LOG_DIR / "agent_pipeline_week2.jsonl"
SUMMARY_PATH = LOG_DIR / "agent_pipeline_week2_summary.json"
SEP = "=" * 70


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("agent_pipeline_week2")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


log = setup_logging(TEXT_LOG_PATH)
if JSONL_LOG_PATH.exists():
    JSONL_LOG_PATH.unlink()  # 每次重新运行都是一份干净的结构化日志

process = psutil.Process(os.getpid())


def rss_mb() -> float:
    return round(process.memory_info().rss / (1024 * 1024), 1)


log.info(f"文本日志：{TEXT_LOG_PATH}")
log.info(f"JSONL 日志：{JSONL_LOG_PATH}")
log.info(f"汇总指标：{SUMMARY_PATH}")

rss_before_load = rss_mb()
log.info(f"加载前进程 RSS：{rss_before_load} MB")

log.info("加载 BGE 嵌入模型与 ChromaDB 索引…")
embedder = BGEEmbedder(device="cpu", log=log)
vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)

log.info("加载 BM25 索引…")
if os.path.exists(BM25_CACHE):
    bm25_index = BM25Index.load(BM25_CACHE, log=log)
else:
    bm25_index = BM25Index(log=log).build_from_chroma(vector_index.collection)

retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)
gen_cache = GenerationCache(log=log)
llm = LLMGenerator(model_name="qwen2.5:7b-instruct", log=log, cache=gen_cache)
rag_pipeline = MedicalGenerationPipeline(retrieval_pipeline=retrieval_pipeline, llm=llm, log=log)

rss_after_rag_load = rss_mb()
log.info(f"RAG 流水线加载完成后 RSS：{rss_after_rag_load} MB（增量 {round(rss_after_rag_load - rss_before_load, 1)} MB）")

# ── 组装 Agent 层（第 2 周：Registry + Dispatcher + AgentMemory + Graph）──
log.info(SEP)
log.info("组装 Agent 层（复用上面已加载的检索/生成组件，不重复加载模型）…")
session_manager = SessionManager()
registry = ToolRegistry(log=log)
register_retrieval_tool(registry, retrieval_pipeline)
dispatcher = ToolDispatcherEngine(registry, log=log)
agent_memory = AgentMemory(session_manager, log=log)
agent_memory.clear_all()  # 干净状态开始本次验证，不受历史运行残留缓存影响
agent_graph = build_agent_graph(
    retrieval_dispatcher=dispatcher, llm=llm,
    session_context_fn=session_manager.build_context_prefix, memory=agent_memory,
)

rss_after_agent_load = rss_mb()
agent_layer_delta_mb = round(rss_after_agent_load - rss_after_rag_load, 1)
log.info(f"Agent 层组装完成后 RSS：{rss_after_agent_load} MB（Agent 层增量 {agent_layer_delta_mb} MB）")

summary: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


# ══════════════════════════════════════════════════════════════════
# 第 1 部分：多组医学查询功能测试
# ══════════════════════════════════════════════════════════════════

log.info(SEP)
log.info("第 1 部分：多组医学查询功能测试")
log.info(SEP)

FUNCTIONAL_QUERIES = [
    "What genes have been associated with susceptibility to type 2 diabetes?",
    "What statistical methods are used to analyze survival data in cancer studies?",
    "How does learning during the day affect brain activity during sleep?",
    "What is the role of the Wnt signaling pathway in cancer?",
    "二甲双胍的作用机制是什么？",
]

func_session_id, _ = session_manager.create_session()
functional_records = []

for query in FUNCTIONAL_QUERIES:
    log.info("-" * 70)
    log.info(f"查询: {query}")
    t0 = time.time()
    state = new_agent_state(query=query, session_id=func_session_id, top_k=6)
    result = agent_graph.invoke(state, config={"recursion_limit": 25})
    elapsed = time.time() - t0

    tool_call = result["tool_call_history"][0]
    answer = result["final_answer"] or ""
    n_sources = len(result["sources"])
    has_citation_marker = ("[来源" in answer) or ("[source" in answer.lower())

    log.info(f"耗时: {elapsed:.2f}s  状态: {result['execution_status'].value}  "
             f"检索结果数: {len(result['retrieval_results'])}  来源数: {n_sources}  含引用标记: {has_citation_marker}")
    log.info(f"答案前 200 字符: {answer[:200]}")

    session_manager.append_turn(func_session_id, query, answer, agent_trace=result["execution_trace"])

    record = {
        "part": "functional", "query": query, "elapsed_seconds": round(elapsed, 2),
        "status": result["execution_status"].value, "n_retrieval_results": len(result["retrieval_results"]),
        "n_sources": n_sources, "has_citation_marker": has_citation_marker,
        "tool_call_success": tool_call.success, "answer_length": len(answer),
    }
    functional_records.append(record)
    with JSONL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

summary["functional"] = {
    "n_queries": len(functional_records),
    "success_rate": round(sum(1 for r in functional_records if r["status"] == "done") / len(functional_records), 4),
    "citation_marker_rate": round(sum(1 for r in functional_records if r["has_citation_marker"]) / len(functional_records), 4),
    "avg_n_sources": round(statistics.mean(r["n_sources"] for r in functional_records), 2),
    "avg_elapsed_seconds": round(statistics.mean(r["elapsed_seconds"] for r in functional_records), 2),
}
log.info(f"第 1 部分汇总: {summary['functional']}")


# ══════════════════════════════════════════════════════════════════
# 第 2 部分：缓存与去重验证
# ══════════════════════════════════════════════════════════════════

log.info(SEP)
log.info("第 2 部分：缓存与去重验证")
log.info(SEP)

cache_session_id, _ = session_manager.create_session()
cache_query = "What genes have been associated with susceptibility to type 2 diabetes?"

t0 = time.time()
state1 = new_agent_state(query=cache_query, session_id=cache_session_id, top_k=6)
result1 = agent_graph.invoke(state1, config={"recursion_limit": 25})
elapsed_cold = time.time() - t0
tool_trace1 = next(t for t in result1["execution_trace"] if t["step"] == "tool_execution")
log.info(f"第 1 次调用（冷）: 总耗时 {elapsed_cold:.2f}s  检索节点耗时 {tool_trace1['elapsed_seconds']}s  "
         f"cache_hit={tool_trace1['outputs']['cache_hit']}")

t0 = time.time()
state2 = new_agent_state(query=cache_query, session_id=cache_session_id, top_k=6)
result2 = agent_graph.invoke(state2, config={"recursion_limit": 25})
elapsed_warm = time.time() - t0
tool_trace2 = next(t for t in result2["execution_trace"] if t["step"] == "tool_execution")
log.info(f"第 2 次调用（同会话同查询，应命中缓存）: 总耗时 {elapsed_warm:.2f}s  "
         f"检索节点耗时 {tool_trace2['elapsed_seconds']}s  cache_hit={tool_trace2['outputs']['cache_hit']}")

# 换一个不同查询，验证跨轮次去重登记（部分文献可能与第一个查询重叠）
dedup_query = "What is the genetic basis of diabetes mellitus?"
state3 = new_agent_state(query=dedup_query, session_id=cache_session_id, top_k=6)
result3 = agent_graph.invoke(state3, config={"recursion_limit": 25})
tool_trace3 = next(t for t in result3["execution_trace"] if t["step"] == "tool_execution")
log.info(f"第 3 次调用（同会话不同查询）: cross_session_repeat_count="
         f"{tool_trace3['outputs']['cross_session_repeat_count']}（>0 说明去重集合真的在跨轮次识别重复文献）")

seen_chunks = agent_memory.get_seen_chunk_ids(cache_session_id)
log.info(f"该会话累计去重集合大小: {len(seen_chunks)}")

# 手动清除缓存，验证失效策略里的"手动清除"分支
n_cleared = agent_memory.clear_session(cache_session_id)
log.info(f"手动清除该会话缓存: 删除 {n_cleared} 个 key")
cleared_check = agent_memory.get_cached_tool_result(
    "retrieval", dispatcher.auto_fill_params("retrieval", state1), session_id=cache_session_id,
)
log.info(f"清除后再查询该 key: {'仍命中（异常）' if cleared_check is not None else '未命中（符合预期）'}")

memory_stats = agent_memory.stats()
summary["cache_and_dedup"] = {
    "cold_call_seconds": round(elapsed_cold, 2),
    "warm_call_seconds": round(elapsed_warm, 2),
    "speedup_pct": round((elapsed_cold - elapsed_warm) / elapsed_cold * 100, 1) if elapsed_cold else None,
    "second_call_cache_hit": tool_trace2["outputs"]["cache_hit"],
    "cross_session_repeat_count_on_third_call": tool_trace3["outputs"]["cross_session_repeat_count"],
    "seen_chunk_set_size_after_three_calls": len(seen_chunks),
    "manual_clear_deleted_keys": n_cleared,
    "manual_clear_verified_miss_after": cleared_check is None,
    "memory_hit_rate_overall": memory_stats["hit_rate"],
}
log.info(f"第 2 部分汇总: {summary['cache_and_dedup']}")


# ══════════════════════════════════════════════════════════════════
# 第 3 部分：元数据过滤验证
# ══════════════════════════════════════════════════════════════════

log.info(SEP)
log.info("第 3 部分：元数据过滤验证")
log.info(SEP)

# 3a：显式 where_filter（直接在 AgentState 里指定，不依赖查询文本）
explicit_filter = {"$and": [{"pub_year": {"$gte": 2003}}, {"pub_year": {"$lte": 2004}}]}
state_explicit = new_agent_state(
    query="cancer treatment mechanisms", session_id=None, top_k=6, where_filter=explicit_filter,
)
result_explicit = agent_graph.invoke(state_explicit, config={"recursion_limit": 25})
years_explicit = {r["metadata"].get("pub_year") for r in result_explicit["retrieval_results"] if r.get("metadata")}
log.info(f"显式 where_filter={explicit_filter} -> 命中结果数={len(result_explicit['retrieval_results'])}  "
         f"实际年份分布={years_explicit}（应全部在 [2003, 2004] 内——语料库本身就是这个区间，见 README）")

# 3b：查询文本自动提取过滤条件（不显式传 where_filter，依赖 query_processor 从文本抽取）
state_auto = new_agent_state(
    query="What research on diabetes was published between 2003 and 2004?", session_id=None, top_k=6,
)
result_auto = agent_graph.invoke(state_auto, config={"recursion_limit": 25})
auto_tool_call = result_auto["tool_call_history"][0]
log.info(f"自动提取过滤条件路径 -> 检索结果数={len(result_auto['retrieval_results'])}  "
         f"工具调用成功={auto_tool_call.success}")

summary["metadata_filter"] = {
    "explicit_filter_n_results": len(result_explicit["retrieval_results"]),
    "explicit_filter_years_in_range": years_explicit.issubset({2003, 2004}) if years_explicit else None,
    "auto_extracted_filter_n_results": len(result_auto["retrieval_results"]),
    "auto_extracted_filter_tool_success": auto_tool_call.success,
}
log.info(f"第 3 部分汇总: {summary['metadata_filter']}")


# ══════════════════════════════════════════════════════════════════
# 第 4 部分：异常兜底验证
# ══════════════════════════════════════════════════════════════════

log.info(SEP)
log.info("第 4 部分：异常兜底验证")
log.info(SEP)

resilience_results = {}

# 4a：非法元数据过滤条件（真实 ChromaDB 会拒绝这个 where 语法）
bad_filter_state = new_agent_state(
    query="diabetes treatment", top_k=5, where_filter={"$invalid_operator": {"x": True}},
)
bad_filter_result = agent_graph.invoke(bad_filter_state, config={"recursion_limit": 25})
bad_filter_call = bad_filter_result["tool_call_history"][0]
log.info(f"4a 非法过滤条件: tool_call.success={bad_filter_call.success}  "
         f"graph 状态={bad_filter_result['execution_status'].value}（不应崩溃，应仍产出兜底回答）")
resilience_results["invalid_where_filter"] = {
    "tool_call_success": bad_filter_call.success,
    "graph_completed_without_crash": True,   # 能走到这里说明 invoke() 没有抛出未捕获异常
    "final_status": bad_filter_result["execution_status"].value,
}

# 4b：检索工具持续超时/连接失败（模拟瞬时性异常，验证指数退避重试 + 最终优雅失败）
class _AlwaysDownPipeline:
    def retrieve(self, query, top_k=8, fusion_strategy="rrf", where_filter=None):
        raise ConnectionError("模拟的向量库连接失败（用于验证重试与兜底逻辑）")

down_registry = ToolRegistry(log=log)
register_retrieval_tool(down_registry, _AlwaysDownPipeline(), max_retries=2)
down_dispatcher = ToolDispatcherEngine(down_registry, log=log, backoff_base_seconds=0.05)
down_graph = build_agent_graph(
    retrieval_dispatcher=down_dispatcher, llm=llm,
    session_context_fn=session_manager.build_context_prefix, memory=agent_memory,
)
t0 = time.time()
down_result = down_graph.invoke(new_agent_state(query="test resilience", top_k=5), config={"recursion_limit": 25})
down_elapsed = time.time() - t0
down_call = down_result["tool_call_history"][0]
log.info(f"4b 检索工具持续失败: 重试次数={down_call.retry_count}  最终 success={down_call.success}  "
         f"耗时={down_elapsed:.2f}s（含退避等待）  graph 状态={down_result['execution_status'].value}")
resilience_results["retrieval_always_down"] = {
    "retry_count": down_call.retry_count,
    "final_tool_call_success": down_call.success,
    "graph_completed_without_crash": True,
    "final_status": down_result["execution_status"].value,
    "fallback_answer_produced": bool(down_result["final_answer"]),
}

# 4c：Redis 缓存后端不可达时的降级（指向一个没有监听的端口）
broken_memory = AgentMemory(session_manager, redis_url="redis://127.0.0.1:1/0", log=log)
broken_graph = build_agent_graph(
    retrieval_dispatcher=dispatcher, llm=llm,
    session_context_fn=session_manager.build_context_prefix, memory=broken_memory,
)
t0 = time.time()
broken_cache_result = broken_graph.invoke(new_agent_state(query="metformin mechanism", top_k=5), config={"recursion_limit": 25})
broken_cache_elapsed = time.time() - t0
log.info(f"4c Redis 不可达降级: graph 状态={broken_cache_result['execution_status'].value}  "
         f"耗时={broken_cache_elapsed:.2f}s（应能正常检索+生成，只是没有缓存加速）")
resilience_results["redis_unavailable_degradation"] = {
    "redis_available_flag": broken_memory._redis_available,
    "graph_completed_without_crash": True,
    "final_status": broken_cache_result["execution_status"].value,
    "answer_produced": bool(broken_cache_result["final_answer"]),
}

summary["resilience"] = resilience_results
log.info(f"第 4 部分汇总: {resilience_results}")


# ══════════════════════════════════════════════════════════════════
# 第 5 部分：性能对比
# ══════════════════════════════════════════════════════════════════

log.info(SEP)
log.info("第 5 部分：性能对比（RAG vs Agent 延迟 / 内存增量 / 并发吞吐）")
log.info(SEP)

PERF_QUERIES = [
    "What genes have been associated with susceptibility to type 2 diabetes?",
    "What statistical methods are used to analyze survival data in cancer studies?",
    "How does learning during the day affect brain activity during sleep?",
]

# 5a：单请求延迟对比（RAG 直通 vs Agent，各自独立 session，避免命中彼此缓存）
rag_times, agent_times = [], []
for q in PERF_QUERIES:
    t0 = time.time()
    rag_pipeline.generate(q, top_k=6, run_evaluation=False, run_review=False)
    rag_times.append(time.time() - t0)

    t0 = time.time()
    agent_graph.invoke(new_agent_state(query=q + " ", top_k=6), config={"recursion_limit": 25})  # 加空格避免命中上面的缓存
    agent_times.append(time.time() - t0)

perf_latency = {
    "rag_seconds": {"avg": round(statistics.mean(rag_times), 2), "min": round(min(rag_times), 2), "max": round(max(rag_times), 2)},
    "agent_seconds": {"avg": round(statistics.mean(agent_times), 2), "min": round(min(agent_times), 2), "max": round(max(agent_times), 2)},
    "avg_overhead_pct_vs_rag": round((statistics.mean(agent_times) - statistics.mean(rag_times)) / statistics.mean(rag_times) * 100, 1),
}
log.info(f"5a 单请求延迟对比: {perf_latency}")

# 5b：内存增量（进程级 RSS，第 2 周新增组件相对第 1 周 RAG 流水线的增量）
perf_memory = {
    "rss_before_any_load_mb": rss_before_load,
    "rss_after_rag_pipeline_mb": rss_after_rag_load,
    "rss_after_agent_layer_mb": rss_after_agent_load,
    "agent_layer_delta_mb": agent_layer_delta_mb,
    "rss_after_full_run_mb": rss_mb(),
}
log.info(f"5b 内存增量: {perf_memory}")

# 5c：并发吞吐（线程池并发调用同一份已加载好的流水线对象，分别测 RAG 与 Agent 两条链路）
def _rag_call(q):
    t0 = time.time()
    rag_pipeline.generate(q, top_k=5, run_evaluation=False, run_review=False)
    return time.time() - t0

def _agent_call(q):
    t0 = time.time()
    agent_graph.invoke(new_agent_state(query=q, top_k=5), config={"recursion_limit": 25})
    return time.time() - t0

CONCURRENCY = 3
throughput_queries = [f"What is the mechanism of drug {i}?" for i in range(CONCURRENCY)]

t0 = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    rag_concurrent_times = list(pool.map(_rag_call, throughput_queries))
rag_wall = time.time() - t0

t0 = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    agent_concurrent_times = list(pool.map(_agent_call, throughput_queries))
agent_wall = time.time() - t0

perf_throughput = {
    "concurrency": CONCURRENCY,
    "rag_wall_seconds": round(rag_wall, 2),
    "rag_qps": round(CONCURRENCY / rag_wall, 3),
    "rag_per_request_avg_seconds": round(statistics.mean(rag_concurrent_times), 2),
    "agent_wall_seconds": round(agent_wall, 2),
    "agent_qps": round(CONCURRENCY / agent_wall, 3),
    "agent_per_request_avg_seconds": round(statistics.mean(agent_concurrent_times), 2),
}
log.info(f"5c 并发吞吐（{CONCURRENCY} 并发）: {perf_throughput}")

summary["performance"] = {"latency": perf_latency, "memory": perf_memory, "throughput": perf_throughput}


# ══════════════════════════════════════════════════════════════════
# 写汇总文件
# ══════════════════════════════════════════════════════════════════

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

log.info(SEP)
log.info(f"全部完成，汇总指标已写入 {SUMMARY_PATH}")
log.info(SEP)
