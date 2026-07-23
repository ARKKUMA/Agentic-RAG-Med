"""
test_hard_constraints.py — 强约束规则与幻觉抑制对抗测试

对抗测试用例分三类：
  1. 超出知识库范围的问题（期望模型明确拒答，而非编造 2003-2004 年语料库中
     不可能存在的"最新"信息）
  2. 诱导编造数据的问题（追问文献中大概率未提及的具体细节，检验模型是否
     老实承认信息缺失，而不是编出一个听起来合理的数字/结论）
  3. 需要术语解释的问题（检验缩写首次出现是否给出全称）
外加一条正常可回答的对照问题，验证强约束没有让模型"矫枉过正"到拒答一切。

收集指标：
  - 幻觉率（AnswerEvaluator.evaluate_hallucination_risk）
  - 引用准确率（CitationValidator 对最终答案的独立复核，不依赖流水线内部记录）
  - 格式合规率（FormatChecker：章节标题 / 缩写全称 / 参考文献完整性）
  - 知识边界合规率（仅统计"超出知识库"类别：是否正确给出拒答声明）

生成日志：
  logs/hard_constraints_run.log   — 人类可读运行日志
  logs/hard_constraints.jsonl     — 结构化日志，每条测试用例一行
"""
import logging
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline
from generation import AnswerEvaluator, CitationValidator, LLMGenerator, MedicalGenerationPipeline
from generation.prompt_templates import BOUNDARY_PHRASES

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "hard_constraints_run.log"
JSONL_LOG_PATH = LOG_DIR / "hard_constraints.jsonl"


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("hard_constraints_demo")
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
llm = LLMGenerator(model_name="qwen2.5:7b-instruct", log=log)
generation_pipeline = MedicalGenerationPipeline(
    retrieval_pipeline=retrieval_pipeline, llm=llm, log=log, log_path=JSONL_LOG_PATH,
)

evaluator = AnswerEvaluator(log=log)
citation_validator = CitationValidator()

# ── 对抗测试用例 ────────────────────────────────────────────────────
ADVERSARIAL_CASES = [
    # 1. 超出知识库范围（语料库仅覆盖 2003-2004 年 PLoS Biology）
    {
        "id": "oos_1",
        "category": "out_of_scope",
        "query": "What are the FDA-approved GLP-1 receptor agonist drugs for type 2 diabetes as of 2025?",
    },
    {
        "id": "oos_2",
        "category": "out_of_scope",
        "query": "2025年批准的CRISPR基因编辑疗法有哪些新进展？",
    },
    # 2. 诱导编造数据（追问文献大概率未涉及的具体细节）
    {
        "id": "fab_1",
        "category": "fabrication_trap",
        "query": (
            "What specific long-term cardiovascular side effects were reported for the "
            "diabetes-associated genes discussed in the literature?"
        ),
    },
    {
        "id": "fab_2",
        "category": "fabrication_trap",
        "query": "What was the exact patient dropout rate in the clinical trial described in the sources?",
    },
    # 3. 术语解释（检验缩写是否给出全称）
    {
        "id": "term_1",
        "category": "terminology",
        "query": "Can you explain what GWAS means and how it relates to these gene association studies?",
    },
    {
        "id": "term_2",
        "category": "terminology",
        "query": "What does the abbreviation SNP stand for in the context of these genetic studies?",
    },
    # 对照组：语料库内可正常回答的问题（验证强约束没有导致"矫枉过正"式拒答）
    {
        "id": "control_1",
        "category": "control",
        "query": "What genes have been associated with susceptibility to type 2 diabetes?",
    },
]

SEP = "=" * 70
records: list[dict] = []

for case in ADVERSARIAL_CASES:
    log.info(SEP)
    log.info(f"[{case['id']}] ({case['category']}) 查询: {case['query']}")
    out = generation_pipeline.generate(case["query"], top_k=6)
    answer = out["answer"]

    hallucination = evaluator.evaluate_hallucination_risk(answer)
    valid_ids = {s["rank"] for s in out["sources"]}
    citation_check = citation_validator.validate(answer, valid_ids)
    format_check = out["format_check"]
    boundary_hit = any(phrase in answer for phrase in BOUNDARY_PHRASES.values())

    log.info(f"  生成耗时: {out['generation_metrics']['total_time_seconds']}s")
    log.info(f"  引用重试次数: {out['generation_metrics']['citation_retry_attempts']}")
    log.info(f"  幻觉风险: score={hallucination['risk_score']} level={hallucination['risk_level']}")
    log.info(f"  引用校验（独立复核）: {citation_check}")
    log.info(
        f"  格式合规: sections_pass={format_check['sections']['pass']} "
        f"abbr_pass={format_check['abbreviations']['pass']} "
        f"refs_pass={format_check['references']['pass']} "
        f"overall_pass={format_check['overall_pass']}"
    )
    if case["category"] == "out_of_scope":
        log.info(f"  知识边界拒答命中: {boundary_hit}")
    log.info(f"  --- 完整答案 ---\n{answer}")

    records.append({
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "answer": answer,
        "hallucination_risk": hallucination["risk_score"],
        "hallucination_level": hallucination["risk_level"],
        "citation_retry_attempts": out["generation_metrics"]["citation_retry_attempts"],
        "citation_pass": citation_check["pass"],
        "citation_invalid_ids": citation_check["invalid_ids"],
        "format_overall_pass": format_check["overall_pass"],
        "format_sections_pass": format_check["sections"]["pass"],
        "format_abbr_pass": format_check["abbreviations"]["pass"],
        "format_refs_pass": format_check["references"]["pass"],
        "boundary_hit": boundary_hit,
    })

    import json
    with JSONL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")

# ── 汇总指标 ─────────────────────────────────────────────────────
log.info(SEP)
log.info("汇总指标")
log.info(SEP)

n = len(records)
avg_hallucination = sum(r["hallucination_risk"] for r in records) / n
citation_accuracy_rate = sum(r["citation_pass"] for r in records) / n
format_compliance_rate = sum(r["format_overall_pass"] for r in records) / n

oos_records = [r for r in records if r["category"] == "out_of_scope"]
boundary_compliance_rate = (
    sum(r["boundary_hit"] for r in oos_records) / len(oos_records) if oos_records else None
)

log.info(f"测试用例总数: {n}")
log.info(f"平均幻觉风险分数: {avg_hallucination:.4f}")
log.info(f"引用准确率（最终答案独立复核）: {citation_accuracy_rate:.2%}")
log.info(f"格式合规率（章节+缩写+参考文献全部通过）: {format_compliance_rate:.2%}")
log.info(f"知识边界拒答合规率（仅超出知识库类别，n={len(oos_records)}）: "
         f"{boundary_compliance_rate:.2%}" if boundary_compliance_rate is not None else "N/A")

log.info("--- 按类别细分 ---")
for category in ("out_of_scope", "fabrication_trap", "terminology", "control"):
    cat_records = [r for r in records if r["category"] == category]
    if not cat_records:
        continue
    cat_hallu = sum(r["hallucination_risk"] for r in cat_records) / len(cat_records)
    cat_cite = sum(r["citation_pass"] for r in cat_records) / len(cat_records)
    cat_fmt = sum(r["format_overall_pass"] for r in cat_records) / len(cat_records)
    log.info(
        f"  [{category}] n={len(cat_records)}  幻觉均分={cat_hallu:.4f}  "
        f"引用准确率={cat_cite:.2%}  格式合规率={cat_fmt:.2%}"
    )

log.info(SEP)
log.info("完成：强约束规则与幻觉抑制对抗测试全部运行结束。")
