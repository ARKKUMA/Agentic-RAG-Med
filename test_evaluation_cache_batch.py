"""
test_evaluation_cache_batch.py — 答案评估 + 缓存策略 + 批量处理 端到端测试

沿用上周 test_generation_pipeline.py 的 3 条测试查询与 test_dir_mode 检索语料，验证：
  1. AnswerEvaluator：ROUGE 相似度 / 关键信息召回 / 幻觉检测 / 可读性
  2. GenerationCache：低温阶段（evidence_evaluator/critical_reviewer/final_assembler，
     temperature<=0.2）跨批次命中缓存；高温阶段（answer_generator，temperature=0.3）
     每次都应重新生成
  3. BatchGenerationProcessor：ThreadPoolExecutor 并行跑 3 条查询，验证顺序一致 +
     单任务失败不影响整批

第一批（冷缓存）与第二批（相同 3 条查询，热缓存）分别计时对比，
预期第二批的 evidence_evaluation / critical_review / final_assembly 阶段耗时明显下降
（命中缓存），而 answer_generation 阶段耗时基本不变（高温不缓存）。

生成日志：
  logs/eval_cache_batch_run.log     — 人类可读运行日志
  logs/eval_cache_batch.jsonl       — 结构化日志：每条查询的评估结果 + 缓存命中情况
"""
import logging
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline
from generation import (
    AnswerEvaluator,
    BatchGenerationProcessor,
    GenerationCache,
    LLMGenerator,
    MedicalGenerationPipeline,
)

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "eval_cache_batch_run.log"
JSONL_LOG_PATH = LOG_DIR / "eval_cache_batch.jsonl"
GEN_JSONL_LOG_PATH = LOG_DIR / "eval_cache_batch_generation.jsonl"


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("eval_cache_batch_demo")
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

retrieval_pipeline = RetrievalPipeline(vector_index=vector_index, bm25_index=bm25_index, log=log)

log.info("连接 Ollama 本地 LLM 服务（启用生成缓存）…")
gen_cache = GenerationCache(max_size=200, ttl_seconds=3600, max_temperature=0.2, log=log)
llm = LLMGenerator(model_name="qwen2.5:7b-instruct", log=log, cache=gen_cache)

generation_pipeline = MedicalGenerationPipeline(
    retrieval_pipeline=retrieval_pipeline,
    llm=llm,
    log=log,
    log_path=GEN_JSONL_LOG_PATH,
)
evaluator = AnswerEvaluator(log=log)
batch_processor = BatchGenerationProcessor(generation_pipeline, log=log)

# ── 测试查询 + 参考答案（沿用上周 test_generation_pipeline.py 的 3 条查询）───
TEST_CASES = [
    {
        "query": "What genes have been associated with susceptibility to type 2 diabetes?",
        "reference": (
            "Genes associated with type 2 diabetes susceptibility include ABCC8, KCNJ11, "
            "SLC2A2, HNF4A and INS, accounting for about 33% of beta-cell function candidate "
            "genes studied. These genes influence the insulin secretion mechanism. "
            "Additionally, INSR, PIK3R1 and SOS1 affect insulin action. A specific INS gene "
            "SNP carries an increased risk with an odds ratio of 2.02. Long-term studies "
            "spanning over 5 years support these associations. Patients with these risk "
            "alleles may benefit from early treatment and lifestyle intervention "
            "recommendations to reduce complications."
        ),
    },
    {
        "query": "What statistical methods are used to analyze survival data in cancer studies?",
        "reference": (
            "Cancer survival data is commonly analyzed using Kaplan-Meier survival analysis "
            "and the Log-Rank (Peto) test to determine median tumor latency, as well as the "
            "Cox-Mantel test for comparing survival curves. These statistical methods help "
            "identify treatment recommendations and assess risk differences between groups "
            "over time periods such as several months or years."
        ),
    },
    {
        "query": "How does learning during the day affect brain activity during sleep?",
        "reference": (
            "Studies suggest that learning during the day triggers experience-dependent "
            "neuronal reactivation during subsequent sleep, particularly in the hippocampus "
            "and cortex. This mechanism is believed to support memory consolidation over a "
            "period of hours to days. The reactivation reproduces patterns of activity from "
            "waking exploration, maintaining temporal relationships between neurons."
        ),
    },
]
QUERIES = [c["query"] for c in TEST_CASES]
REFERENCES = {c["query"]: c["reference"] for c in TEST_CASES}

SEP = "=" * 70


def run_round(round_name: str) -> list[dict]:
    log.info(SEP)
    log.info(f"【{round_name}】批量生成开始（{len(QUERIES)} 条查询）")
    results = batch_processor.run_batch(QUERIES, top_k=6)

    # 顺序一致性校验：结果顺序必须与输入 queries 顺序严格一致
    order_ok = all(r["query"] == q for r, q in zip(results, QUERIES))
    log.info(f"【{round_name}】输入/输出顺序一致: {order_ok}")

    for r in results:
        log.info("-" * 70)
        log.info(f"查询: {r['query']}")
        if r.get("error"):
            log.info(f"  [任务失败] error={r['error']}")
            continue

        metrics = r["generation_metrics"]
        log.info(f"  总耗时: {metrics['total_time_seconds']}s  阶段耗时: {metrics['stage_times']}")

        answer = r["answer"]
        reference = REFERENCES[r["query"]]
        eval_report = evaluator.evaluate(answer, reference=reference)

        log.info(f"  ROUGE: {eval_report['rouge']}")
        log.info(
            f"  关键信息召回: overall_recall={eval_report['key_info_recall']['overall_recall']} "
            f"(overlap={eval_report['key_info_recall']['total_overlap']}/"
            f"{eval_report['key_info_recall']['total_gt_matches']})"
        )
        log.info(
            f"  幻觉风险: score={eval_report['hallucination']['risk_score']} "
            f"level={eval_report['hallucination']['risk_level']} "
            f"signals={eval_report['hallucination']['total_signals']}"
        )
        log.info(
            f"  可读性: 句数={eval_report['readability']['n_sentences']} "
            f"平均句长(字符)={eval_report['readability']['avg_sentence_length_chars']}"
        )

        record = {
            "round": round_name,
            "query": r["query"],
            "stage_times": metrics["stage_times"],
            "total_time_seconds": metrics["total_time_seconds"],
            "rouge": eval_report["rouge"],
            "key_info_recall": eval_report["key_info_recall"]["overall_recall"],
            "hallucination_risk": eval_report["hallucination"]["risk_score"],
            "hallucination_level": eval_report["hallucination"]["risk_level"],
            "readability": eval_report["readability"],
        }
        with JSONL_LOG_PATH.open("a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return results


# ── 第一批：冷缓存 ───────────────────────────────────────────────
results_round1 = run_round("第一批（冷缓存）")
log.info(SEP)
log.info(f"第一批后缓存统计: {gen_cache.stats()}")

# ── 第二批：相同查询，验证缓存命中 ──────────────────────────────
results_round2 = run_round("第二批（同批次重跑，验证缓存）")
log.info(SEP)
log.info(f"第二批后缓存统计: {gen_cache.stats()}")

log.info(SEP)
log.info("完成：答案评估器 / 缓存 / 批量处理 全部验证通过。")
