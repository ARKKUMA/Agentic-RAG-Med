"""
test_retrieval_pipeline_full.py — 全量语料检索流水线样例运行
向量检索改用 pmc_full 集合（580 万 chunks，覆盖真实年份/期刊分布），
BM25 使用跨全库随机采样的 20 万 chunk 子集（build_bm25_sample.py 预构建）。

对比 test_retrieval_pipeline.py（仅用 1854 条 2003-2004 年 PLoS Biology 样本，
查询主题在库内本就没有真实匹配，relevance 分数普遍偏低是数据覆盖问题，不是模型问题）。

生成日志：
  logs/retrieval_pipeline_full_run.log
  logs/retrieval_pipeline_full.jsonl
"""
import logging
import sys
from pathlib import Path

import psutil

sys.stdout.reconfigure(encoding="utf-8")

from pmc_vector_index import BGEEmbedder, PMCVectorIndex
from retrieval import BM25Index, RetrievalPipeline

DB_DIR = r"d:\Rag-Med\pipeline_output\chroma_db"
COLLECTION = "pmc_full"
BM25_CACHE = r"d:\Rag-Med\pipeline_output\bm25_index_pmc_sample200k.pkl"
LOG_DIR = Path("d:/Rag-Med/logs")
TEXT_LOG_PATH = LOG_DIR / "retrieval_pipeline_full_run.log"
JSONL_LOG_PATH = LOG_DIR / "retrieval_pipeline_full.jsonl"


def setup_logging(log_path: Path) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log = logging.getLogger("retrieval_pipeline_full_demo")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


log = setup_logging(TEXT_LOG_PATH)
log.info(f"文本日志：{TEXT_LOG_PATH}")
log.info(f"JSONL 日志：{JSONL_LOG_PATH}")

log.info("加载 BGE 嵌入模型…")
embedder = BGEEmbedder(device="cpu")
log.info(f"RSS={rss_gb():.2f}GB")

log.info(f"打开 ChromaDB 集合 {COLLECTION}（580 万向量，首次查询会加载 HNSW 索引到内存）…")
vector_index = PMCVectorIndex(db_dir=DB_DIR, collection_name=COLLECTION, embedder=embedder, log=log)

log.info(f"加载 BM25 采样索引：{BM25_CACHE}")
bm25_index = BM25Index.load(BM25_CACHE, log=log)
log.info(f"BM25 索引文档数：{len(bm25_index):,}  RSS={rss_gb():.2f}GB")

log.info("加载重排序模型 BAAI/bge-reranker-base…")
pipeline = RetrievalPipeline(
    vector_index=vector_index,
    bm25_index=bm25_index,
    log=log,
    log_path=JSONL_LOG_PATH,
)
log.info(f"流水线就绪，RSS={rss_gb():.2f}GB")

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
log.info(f"完成，共运行 {len(SAMPLE_QUERIES)} 条样例查询。RSS={rss_gb():.2f}GB")
