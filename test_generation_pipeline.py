"""
test_generation_pipeline.py — 医学生成流水线端到端测试
检索(向量+BM25融合+重排序) -> 上下文组装 -> 证据评估 -> 答案草稿 -> 批判性审查 -> 最终答案

使用 test_dir_mode（1854 chunks，2003-2004 年 PLoS Biology）作为检索语料 —
测试查询专门围绕该语料实际覆盖的主题（2型糖尿病易感基因、生存分析统计方法、
睡眠与记忆），确保生成结果有真实依据可评估，而非受限于语料覆盖不足。

生成日志：
  logs/generation_pipeline_run.log    — 人类可读运行日志
  logs/generation_pipeline.jsonl      — 结构化 JSONL 日志（每条查询一行，含阶段耗时/成功情况）
"""
import logging
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline
from generation import LLMGenerator, MedicalGenerationPipeline

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "generation_pipeline_run.log"
JSONL_LOG_PATH = LOG_DIR / "generation_pipeline.jsonl"


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("generation_pipeline_demo")
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
embedder = BGEEmbedder(device="cpu")
vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)

log.info("加载 BM25 索引…")
if os.path.exists(BM25_CACHE):
    bm25_index = BM25Index.load(BM25_CACHE, log=log)
else:
    bm25_index = BM25Index(log=log).build_from_chroma(vector_index.collection)
    bm25_index.save(BM25_CACHE)
log.info(f"BM25 索引文档数：{len(bm25_index):,}")

log.info("加载重排序模型…")
retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)

log.info("连接 Ollama 本地 LLM 服务…")
llm = LLMGenerator(model_name="qwen2.5:7b-instruct", log=log)

generation_pipeline = MedicalGenerationPipeline(
    retrieval_pipeline=retrieval_pipeline,
    llm=llm,
    log=log,
    log_path=JSONL_LOG_PATH,
)

# ── 测试查询：均围绕 test_dir_mode 语料实际覆盖的主题 ──────────────
TEST_QUERIES = [
    "What genes have been associated with susceptibility to type 2 diabetes?",
    "What statistical methods are used to analyze survival data in cancer studies?",
    "How does learning during the day affect brain activity during sleep?",
]

SEP = "=" * 70
for query in TEST_QUERIES:
    log.info(SEP)
    log.info(f"查询: {query}")
    result = generation_pipeline.generate(query, top_k=6)

    metrics = result["generation_metrics"]
    log.info(f"总耗时: {metrics['total_time_seconds']}s")
    log.info(f"阶段耗时: {metrics['stage_times']}")
    log.info(f"阶段成功: {metrics['stage_success']}")
    log.info(f"Token 统计: {metrics['token_counts']}")
    log.info(f"上下文统计: {result['context_metadata']}")

    log.info(f"--- 最终答案 (长度={len(result['answer'])}字符) ---")
    log.info(result["answer"])

    log.info("--- 引用来源 ---")
    for s in result["sources"]:
        log.info(f"  [{s['rank']}] {s['journal']} ({s['pub_year']})  {s['pmc_id']}  rel={s['relevance_score']:.4f}")

    if result["intermediate_results"]["evidence_evaluation"]:
        log.info(f"--- 证据评估 ---\n{result['intermediate_results']['evidence_evaluation']}")
    if result["intermediate_results"]["review_feedback"]:
        log.info(f"--- 审查反馈 ---\n{result['intermediate_results']['review_feedback']}")

log.info(SEP)
log.info(f"完成，共运行 {len(TEST_QUERIES)} 条测试查询。")
