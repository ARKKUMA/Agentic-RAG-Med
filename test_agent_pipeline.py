"""
test_agent_pipeline.py — Agent 底座真实端到端验证
用真实的 BGE 嵌入模型 / ChromaDB / BM25 / reranker / Ollama 跑一遍完整的
entry -> tool_execution -> answer_generation -> termination 状态机，
验证：
  1. 真实检索工具通过 ToolDispatcherEngine 正常调用，返回真实检索结果
  2. 真实 LLM 基于检索上下文生成引用了 [来源 N] 的初始答案
  3. 执行轨迹（execution_trace）完整记录四个节点，可持久化进会话（agent_trace）
  4. 与现有单轮 RAG 流水线（MedicalGenerationPipeline）耗时量级相近，
     确认 Agent 编排本身没有引入明显的额外开销（unittest 中的
     tests/test_api.py 回归套件已确认原有 RAG 功能本身零改动、零退化，
     这里只比较"新链路"与"旧链路"的耗时量级）

生成日志：
  logs/agent_pipeline_run.log    — 人类可读运行日志
  logs/agent_pipeline.jsonl      — 结构化日志，每条查询一行
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline
from generation import LLMGenerator, MedicalGenerationPipeline
from api.session import SessionManager
from agent import ToolDispatcherEngine, ToolRegistry, build_agent_graph, new_agent_state, register_retrieval_tool

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "agent_pipeline_run.log"
JSONL_LOG_PATH = LOG_DIR / "agent_pipeline.jsonl"


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("agent_pipeline_demo")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


log = setup_logging(TEXT_LOG_PATH)
log.info(f"文本日志：{TEXT_LOG_PATH}")
log.info(f"JSONL 日志：{JSONL_LOG_PATH}")

log.info("加载 BGE 嵌入模型与 ChromaDB 索引…")
embedder = BGEEmbedder(device="cpu", log=log)
vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)

log.info("加载 BM25 索引…")
if os.path.exists(BM25_CACHE):
    bm25_index = BM25Index.load(BM25_CACHE, log=log)
else:
    bm25_index = BM25Index(log=log).build_from_chroma(vector_index.collection)
    bm25_index.save(BM25_CACHE)
log.info(f"BM25 索引文档数：{len(bm25_index):,}")

retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)
llm = LLMGenerator(model_name="qwen2.5:7b-instruct", log=log)

# ── 组装 Agent 图（真实检索工具 + 真实 LLM）───────────────────────
log.info("注册检索工具、组装 Agent 状态机…")
registry = ToolRegistry(log=log)
register_retrieval_tool(registry, retrieval_pipeline)
dispatcher = ToolDispatcherEngine(registry, log=log)

session_manager = SessionManager()
agent_graph = build_agent_graph(
    retrieval_dispatcher=dispatcher,
    llm=llm,
    session_context_fn=session_manager.build_context_prefix,
)

# ── 对照组：现有单轮 RAG 流水线（验证零退化用）──────────────────────
rag_pipeline = MedicalGenerationPipeline(retrieval_pipeline=retrieval_pipeline, llm=llm, log=log)

TEST_QUERIES = [
    "What genes have been associated with susceptibility to type 2 diabetes?",
    "What statistical methods are used to analyze survival data in cancer studies?",
    "二甲双胍的作用机制是什么？",
]

SEP = "=" * 70
session_id, _ = session_manager.create_session()

for query in TEST_QUERIES:
    log.info(SEP)
    log.info(f"查询: {query}")

    # --- Agent 模式 ---
    t0 = time.time()
    initial_state = new_agent_state(query=query, session_id=session_id, top_k=6)
    result = agent_graph.invoke(initial_state, config={"recursion_limit": 25})
    agent_elapsed = time.time() - t0

    log.info(f"[Agent 模式] 耗时: {agent_elapsed:.2f}s  状态: {result['execution_status'].value}")
    log.info(f"[Agent 模式] 检索工具调用: success={result['tool_call_history'][0].success}  "
             f"n_results={len(result['retrieval_results'])}")
    log.info(f"[Agent 模式] 执行轨迹步骤: {[t['step'] for t in result['execution_trace']]}")
    log.info(f"[Agent 模式] 最终答案（前 200 字符）: {result['final_answer'][:200]}")

    # 持久化执行轨迹进会话（对应"会话记忆扩展"里的 agent_trace 落地）
    session_manager.append_turn(session_id, query, result["final_answer"], agent_trace=result["execution_trace"])

    # --- 单轮 RAG 模式（对照，验证零退化）───────────────────────────
    t0 = time.time()
    rag_result = rag_pipeline.generate(query, top_k=6, run_evaluation=False, run_review=False)
    rag_elapsed = time.time() - t0
    log.info(f"[RAG 对照] 耗时: {rag_elapsed:.2f}s（run_evaluation=False, run_review=False，"
              f"对应 Agent 模式当前只做检索+单次生成，公平比较同等步骤量级）")

    record = {
        "query": query,
        "agent_elapsed_seconds": round(agent_elapsed, 2),
        "agent_status": result["execution_status"].value,
        "agent_n_retrieval_results": len(result["retrieval_results"]),
        "agent_tool_call_success": result["tool_call_history"][0].success,
        "agent_trace_steps": [t["step"] for t in result["execution_trace"]],
        "rag_elapsed_seconds": round(rag_elapsed, 2),
    }
    with JSONL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

log.info(SEP)
log.info("验证会话 agent_trace 持久化 + 查询接口…")
info = session_manager.get_session_info(session_id)
log.info(f"会话共 {info['turn_count']} 轮，全部含 agent_trace: "
          f"{all(t.agent_trace is not None for t in info['turns'])}")
all_trace_steps = [e['step'] for e in session_manager.get_agent_trace(session_id)]
log.info(f"get_agent_trace() 展平后共 {len(all_trace_steps)} 条记录")
tool_exec_only = session_manager.get_agent_trace(session_id, step="tool_execution")
log.info(f"按 step='tool_execution' 筛选后共 {len(tool_exec_only)} 条记录")

log.info(SEP)
log.info("完成：Agent 底座真实端到端验证结束。")
