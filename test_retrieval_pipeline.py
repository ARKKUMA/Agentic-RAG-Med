"""
test_retrieval_pipeline.py — 完整检索流水线样例运行
向量检索 + BM25 关键词检索 + 融合（rrf/weighted/simple） + 交叉编码器重排序

生成两类日志：
  logs/retrieval_pipeline_run.log   — 人类可读运行日志（控制台 + 文件双写）
  logs/retrieval_pipeline.jsonl     — 结构化 JSONL 日志（每条查询一行，含完整分数明细）
"""
import logging
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "test_dir_mode"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "retrieval_pipeline_run.log"
JSONL_LOG_PATH = LOG_DIR / "retrieval_pipeline.jsonl"


# ── 日志配置：控制台 + 文件双写（沿用 pmc_vector_index.py 的日志风格）──
def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("retrieval_pipeline_demo")
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

log.info("构建/加载 BM25 索引…")
if os.path.exists(BM25_CACHE):
    bm25_index = BM25Index.load(BM25_CACHE, log=log)
else:
    bm25_index = BM25Index(log=log).build_from_chroma(vector_index.collection)
    bm25_index.save(BM25_CACHE)
log.info(f"BM25 索引文档数：{len(bm25_index):,}")

log.info("加载重排序模型（首次运行会从 HuggingFace 下载 BAAI/bge-reranker-base）…")
pipeline = RetrievalPipeline(
    vector_index=vector_index,
    bm25_index=bm25_index,
    log=log,
    log_path=JSONL_LOG_PATH,
)

# ── 样例查询：覆盖缩写/中文/多实体/时间过滤/纯缩写/方法学/无实体边界/融合策略全覆盖 ──
SAMPLE_QUERIES = [
    ("What is the effect of metformin on T2DM patients in the last 5 years?", "rrf"),
    ("二甲双胍对心血管疾病有何影响？", "weighted"),
    ("mechanism of aspirin in preventing MI and stroke", "rrf"),
    ("COVID-19 vaccine efficacy trial results since 2021", "weighted"),
    ("HbA1c LDL HDL in T2DM patients after statin therapy", "simple"),
    ("statistical methods used for survival analysis", "rrf"),
    ("how does exercise affect mental health?", "weighted"),
    ("CRISPR gene editing cancer therapy", "simple"),
]

SEP = "=" * 70
for query, strategy in SAMPLE_QUERIES:
    log.info(SEP)
    log.info(f"查询: {query}   融合策略: {strategy}")
    out = pipeline.retrieve(query, top_k=5, fusion_strategy=strategy)
    log.info(
        f"识别实体: {out['query_info'].entities}  "
        f"过滤条件: {out['query_info'].filters}  "
        f"融合候选数: {out['fused_candidates']}"
    )
    for r in out["results"]:
        log.info(
            f"  [{r['final_rank']}] final={r['final_score']:.4f} "
            f"rel={r['relevance_score']:.4f} rec={r['recency_score']:.4f} "
            f"auth={r['authority_score']:.4f}  fused={r['fused_score']:.4f}  "
            f"journal={r['metadata'].get('journal', '?')!r} year={r['metadata'].get('pub_year', '?')} "
            f"sources={r['sources']}"
        )
        log.info(f"       {r['text'][:120]}…")

log.info(SEP)
log.info(f"完成，共运行 {len(SAMPLE_QUERIES)} 条样例查询。")
