"""
build_bm25_sample.py — 从 batches_full 随机采样构建大规模 BM25 索引
batch_*.parquet 单文件行数极不均匀（24 ~ 74000+ 行），因此按累计行数采样
（而非固定文件数），确保跨全语料库（~580 万 chunks）随机抽样，兼顾主题/年份多样性
与内存可控性（BM25Okapi 需要全量分词语料常驻内存）。
"""
import glob
import random
import sys
import time

import pandas as pd
import psutil

sys.path.insert(0, r"d:\Rag-Med")
from retrieval import BM25Index

BATCH_DIR = r"d:\Rag-Med\pipeline_output\batches_full"
TARGET_ROWS = 200_000
OUTPUT_PATH = r"d:\Rag-Med\pipeline_output\bm25_index_pmc_sample200k.pkl"
SEED = 42


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def main():
    files = sorted(glob.glob(f"{BATCH_DIR}/batch_*.parquet"))
    rng = random.Random(SEED)
    rng.shuffle(files)

    print(f"共 {len(files)} 个批次文件，随机采样直到累计 {TARGET_ROWS:,} 行…")
    dfs = []
    total = 0
    t0 = time.time()
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        total += len(df)
        if len(dfs) % 20 == 0 or total >= TARGET_ROWS:
            print(f"  已读取 {len(dfs)} 个文件，累计 {total:,} 行，RSS={rss_gb():.2f}GB")
        if total >= TARGET_ROWS:
            break

    combined = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"采样完成：{len(combined):,} 行，来自 {len(combined['doc_id'].unique()) if 'doc_id' in combined.columns else '?'} 篇文档，"
          f"耗时 {time.time()-t0:.1f}s，RSS={rss_gb():.2f}GB")

    print("年份分布（采样后）：")
    print(combined["pub_year"].value_counts().sort_index().tail(15))

    bm25 = BM25Index().build_from_dataframe(combined)
    print(f"BM25 索引构建完成，文档数={len(bm25):,}，RSS={rss_gb():.2f}GB")

    bm25.save(OUTPUT_PATH)
    print(f"已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
