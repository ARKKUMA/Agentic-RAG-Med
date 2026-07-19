"""
PMC oa_comm 向量索引构建 Pipeline
===================================
使用 BAAI/bge-base-en-v1.5 对切分好的 chunk 数据进行 embedding，
构建 ChromaDB 持久化向量数据库，支持语义检索与元数据过滤。

工作流：
  1. 加载 chunks Parquet 文件
  2. BGEEmbedder 批量生成文档向量（文档无前缀，查询加指令前缀）
  3. PMCVectorIndex 构建 ChromaDB 持久化索引（余弦相似度）
  4. 质量验证（基础统计 / 自相似性 / 边界情况）

用法：
  # 构建索引
  python pmc_vector_index.py --input pipeline_output/chunks_ft_smoke.parquet

  # 指定输出目录和集合名
  python pmc_vector_index.py --input chunks_full.parquet --db-dir pipeline_output/chroma_db --collection pmc_full

  # 限制处理数量（测试）
  python pmc_vector_index.py --input chunks_ft_smoke.parquet --limit 200

  # CPU 模式
  python pmc_vector_index.py --input chunks_ft_smoke.parquet --device cpu
"""

# ══════════════════════════════════════════════════════════════════
# 第 1 节：环境准备与常量
# ══════════════════════════════════════════════════════════════════

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# libgomp 要求 OMP_NUM_THREADS 必须是正整数，空值或非整数会警告
if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)

import numpy as np
import pandas as pd
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ── 默认参数 ──────────────────────────────────────────────────────
DEFAULT_MODEL      = "BAAI/bge-base-en-v1.5"
EMBED_BATCH_SIZE   = 256     # 每次送入模型的句子数
CHROMA_BATCH_SIZE  = 2000   # 每次写入 ChromaDB 的文档数
DEFAULT_COLLECTION = "pmc_chunks"
DEFAULT_DB_DIR     = r"d:\Rag-Med\pipeline_output\chroma_db"

# BGE 检索指令（仅用于 Query，文档不加前缀）
# bge-base-en-v1.5 在加指令时 retrieval 性能稍优
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ChromaDB 元数据字段：str 类型的 None 替换为 ""，int 类型替换为 0
_STR_META_FIELDS = [
    "doc_id", "chunk_type", "section_title", "section_path",
    "imrad_type", "split_strategy", "pmc_id", "pmid", "doi",
    "journal", "article_type", "abstract_source", "source_url", "source_title",
]
_INT_META_FIELDS = ["pub_year", "token_count", "chunk_index", "total_chunks"]


# ── 日志 ──────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = logging.getLogger("pmc_vector")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


# ══════════════════════════════════════════════════════════════════
# 第 2 节：BGEEmbedder — 嵌入模型封装
# ══════════════════════════════════════════════════════════════════

class BGEEmbedder:
    """
    BAAI/bge-base-en-v1.5 嵌入模型封装。

    关键设计：
      - 文档向量：无前缀，直接编码（bge-v1.5 文档侧无需指令）
      - 查询向量：添加检索指令前缀，提升召回质量
      - normalize_embeddings=True：L2 归一化，配合余弦相似度使用
      - show_progress_bar=False：避免在日志中产生 tqdm 噪声
    """

    def __init__(
        self,
        model_name:  str = DEFAULT_MODEL,
        batch_size:  int = EMBED_BATCH_SIZE,
        device:      str = "auto",
        log: logging.Logger | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.log        = log or logging.getLogger("pmc_vector")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.log.info(f"加载嵌入模型：{model_name}  设备：{self.device}")
        self.model = SentenceTransformer(model_name, device=self.device,
                                         model_kwargs={"torch_dtype": torch.float16})
        get_dim = getattr(self.model, "get_embedding_dimension",
                          self.model.get_sentence_embedding_dimension)
        self.dimension = get_dim()
        self.log.info(f"模型加载完成，向量维度：{self.dimension}")

    # ── 批量生成文档向量 ──────────────────────────────────────────
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        批量生成文档向量（无指令前缀）。

        返回 np.ndarray，shape=(len(texts), dimension)，dtype=float32。
        sentence-transformers 内部已处理分批，无需手动分 batch。
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        result = self.model.encode(
            texts,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
            convert_to_numpy     = True,
        )
        # 释放碎片缓存，减少长时间运行时的 OOM 风险
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    # ── 生成查询向量 ──────────────────────────────────────────────
    def embed_query(self, query_text: str) -> list[float]:
        """
        生成单条查询向量，自动添加 BGE 检索指令前缀。

        BGE 查询指令格式：
          "Represent this sentence for searching relevant passages: {query}"
        """
        if not query_text or not query_text.strip():
            raise ValueError("查询文本不能为空")

        prefixed = BGE_QUERY_INSTRUCTION + query_text.strip()
        emb = self.model.encode(
            prefixed,
            normalize_embeddings = True,
            show_progress_bar    = False,
            convert_to_numpy     = True,
        )
        return emb.tolist()


# ══════════════════════════════════════════════════════════════════
# 第 3 节：PMCVectorIndex — ChromaDB 索引构建与查询
# ══════════════════════════════════════════════════════════════════

class PMCVectorIndex:
    """
    ChromaDB 持久化向量索引，封装索引构建、查询和统计。

    ChromaDB 配置：
      - 距离函数：cosine（余弦相似度，配合 L2 归一化使用）
      - 持久化：本地文件（HNSWlib）
      - 元数据：str / int 字段，None 填充为 "" / 0（ChromaDB 不接受 None）
    """

    def __init__(
        self,
        db_dir:          str | Path,
        collection_name: str               = DEFAULT_COLLECTION,
        embedder:        BGEEmbedder | None = None,
        log:             logging.Logger | None = None,
    ):
        self.db_dir          = Path(db_dir)
        self.collection_name = collection_name
        self.embedder        = embedder
        self.log             = log or logging.getLogger("pmc_vector")

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client     = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_or_create_collection(
            name     = collection_name,
            metadata = {"hnsw:space": "cosine"},
        )
        self.log.info(
            f"ChromaDB 集合：{collection_name}  "
            f"当前已有向量：{self.collection.count()}"
        )

    # ── 元数据清洗（向量化批量版，替代逐行 iterrows）─────────────
    @staticmethod
    def _batch_sanitize_meta(df: pd.DataFrame) -> list[dict]:
        """
        将 DataFrame 整批转为 ChromaDB 所需 metadata list。
        比逐行 iterrows() 快约 10–50x。
        """
        tmp = {}
        for f in _STR_META_FIELDS:
            tmp[f] = df[f].fillna("").astype(str).tolist() if f in df.columns \
                     else [""] * len(df)
        for f in _INT_META_FIELDS:
            if f in df.columns:
                tmp[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).astype(int).tolist()
            else:
                tmp[f] = [0] * len(df)

        all_fields = _STR_META_FIELDS + _INT_META_FIELDS
        return [
            {f: tmp[f][i] for f in all_fields}
            for i in range(len(df))
        ]

    # 保留单行版本供外部偶尔调用
    @staticmethod
    def _sanitize_meta(row: pd.Series) -> dict:
        meta = {}
        for field in _STR_META_FIELDS:
            v = row.get(field)
            meta[field] = str(v) if (v is not None and pd.notna(v)) else ""
        for field in _INT_META_FIELDS:
            v = row.get(field)
            try:
                meta[field] = int(v) if (v is not None and pd.notna(v)) else 0
            except (TypeError, ValueError):
                meta[field] = 0
        return meta

    # ── 构建索引 ──────────────────────────────────────────────────
    def build(
        self,
        chunks_df:      pd.DataFrame,
        chroma_batch:   int = CHROMA_BATCH_SIZE,
        overwrite:      bool = False,
    ) -> dict:
        """
        从 chunks DataFrame 批量生成 embedding 并写入 ChromaDB。

        参数：
          chunks_df   : pmc_document_splitter.py 输出的 DataFrame
          chroma_batch: 每批写入 ChromaDB 的文档数
          overwrite   : True 时删除已有同名集合后重建

        返回：索引统计 dict（同 save_stats 结构）
        """
        if self.embedder is None:
            raise RuntimeError("未提供 BGEEmbedder，无法构建索引")

        if overwrite and self.collection.count() > 0:
            self.log.info(f"overwrite=True，删除集合 {self.collection_name} 并重建")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name     = self.collection_name,
                metadata = {"hnsw:space": "cosine"},
            )

        total = len(chunks_df)
        self.log.info(f"开始构建索引：{total:,} chunks")
        t0 = time.time()

        # ── 批量 Embed ──────────────────────────────────────────
        texts = chunks_df["text"].tolist()
        self.log.info("Step 1/2  批量生成文档向量…")
        embeddings = self.embedder.embed_documents(texts)

        # ── 批量写入 ChromaDB ────────────────────────────────────
        self.log.info("Step 2/2  写入 ChromaDB…")
        for start in range(0, total, chroma_batch):
            end   = min(start + chroma_batch, total)
            sub   = chunks_df.iloc[start:end]
            self.collection.add(
                ids        = sub["chunk_id"].tolist(),
                embeddings = embeddings[start:end].tolist(),
                documents  = sub["text"].tolist(),
                metadatas  = self._batch_sanitize_meta(sub),
            )
            self.log.info(f"  写入 {end:>6,}/{total:,}")

        elapsed = time.time() - t0
        n_indexed = self.collection.count()
        self.log.info(
            f"\n索引构建完成  向量总数={n_indexed:,}  耗时={elapsed:.1f}s  "
            f"({total / elapsed:.0f} chunks/s)"
        )

        # ── 验证索引大小 ─────────────────────────────────────────
        if n_indexed != total:
            self.log.warning(f"向量数 ({n_indexed}) ≠ 输入 chunks ({total})，可能有重复 chunk_id")

        # ── 构建统计信息 ─────────────────────────────────────────
        tc = chunks_df["token_count"].dropna() if "token_count" in chunks_df.columns else pd.Series(dtype=float)
        stats = {
            "collection_name":    self.collection_name,
            "total_chunks":       n_indexed,
            "embedding_model":    self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "index_built_at":     pd.Timestamp.now().isoformat(),
            "elapsed_seconds":    round(elapsed, 1),
            "db_dir":             str(self.db_dir),
            "chunk_size_stats": {
                "mean": round(float(tc.mean()), 1),
                "max":  int(tc.max()),
                "min":  int(tc.min()),
            } if len(tc) > 0 else {},
            "metadata_fields": _STR_META_FIELDS + _INT_META_FIELDS,
        }
        return stats

    # ── 断点续传：embedding checkpoint ───────────────────────────
    def _embed_checkpoint_path(self, batch_dir: Path) -> Path:
        return batch_dir / f"embed_checkpoint_{self.collection_name}.json"

    def _load_embed_checkpoint(self, batch_dir: Path) -> set[str]:
        p = self._embed_checkpoint_path(batch_dir)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return set(json.load(f).get("done_files", []))
        return set()

    def _save_embed_checkpoint(self, batch_dir: Path, done_files: set[str]) -> None:
        p = self._embed_checkpoint_path(batch_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"done_files": sorted(done_files)}, f, indent=2)

    # ── 从批次目录逐批 embed 写入 ChromaDB ───────────────────────
    def build_from_dir(
        self,
        batch_dir:    str | Path,
        chroma_batch: int  = CHROMA_BATCH_SIZE,
        overwrite:    bool = False,
        resume:       bool = True,
    ) -> dict:
        """
        遍历 batch_dir 下所有 batch_*.parquet，逐批生成 embedding 写入 ChromaDB。
        每批完成后保存断点，支持 resume=True 跳过已完成的批次。

        内存占用：单批 DataFrame 大小（约 2000 行），不会累积。
        """
        if self.embedder is None:
            raise RuntimeError("未提供 BGEEmbedder，无法构建索引")

        batch_dir   = Path(batch_dir)
        batch_files = sorted(batch_dir.glob("batch_*.parquet"))
        if not batch_files:
            raise FileNotFoundError(f"未找到批次文件：{batch_dir}")

        # overwrite：清空集合
        if overwrite and self.collection.count() > 0:
            self.log.info(f"overwrite=True，删除集合 {self.collection_name} 并重建")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name     = self.collection_name,
                metadata = {"hnsw:space": "cosine"},
            )

        # 加载断点
        done_files = self._load_embed_checkpoint(batch_dir) if resume else set()
        pending    = [bf for bf in batch_files if str(bf) not in done_files]
        skipped    = len(batch_files) - len(pending)

        self.log.info(
            f"批次 embedding 开始：共 {len(batch_files):,} 个批次文件，"
            f"已完成 {skipped:,} 个，待处理 {len(pending):,} 个"
        )

        t0        = time.time()
        total_new = 0
        # 不再累积 token_counts 列表（77M 条会大量占内存），改为滚动统计
        tok_sum = 0
        tok_cnt = 0

        # ── ChromaDB 写入（后台线程执行，compaction 错误长退避重试）──
        def _write_chroma(df: pd.DataFrame, embs: np.ndarray) -> int:
            n = 0
            for start in range(0, len(df), chroma_batch):
                end = min(start + chroma_batch, len(df))
                sub = df.iloc[start:end]
                wait = 5
                for attempt in range(5):
                    try:
                        self.collection.add(
                            ids        = sub["chunk_id"].tolist(),
                            embeddings = embs[start:end].tolist(),
                            documents  = sub["text"].tolist(),
                            metadatas  = self._batch_sanitize_meta(sub),
                        )
                        break
                    except Exception as e:
                        if attempt == 4:
                            raise
                        self.log.warning(
                            f"ChromaDB add 失败 (attempt {attempt+1}/5): {e}  "
                            f"等待 {wait}s 后重试..."
                        )
                        time.sleep(wait)
                        wait *= 2   # 指数退避：5→10→20→40s
                n += end - start
            return n

        # ── 预读线程：在 GPU encode 期间提前读取下一批 parquet ────────
        # 目标时序：
        #   GPU: [══encode i══][══encode i+1══][══encode i+2══]
        #   IO:  [read i][read i+1] [read i+2]
        #   DB:        [write i-1]  [write i]  [write i+1]
        import queue as _queue

        PREFETCH_AHEAD = 2   # 提前预读的批次数

        def _parquet_reader(files: list, q: "_queue.Queue"):
            for f in files:
                try:
                    df = pd.read_parquet(f)
                except Exception as e:
                    q.put((f, None, e))
                    continue
                q.put((f, df, None))
            q.put(None)  # sentinel

        prefetch_q  = _queue.Queue(maxsize=PREFETCH_AHEAD)
        reader_thr  = threading.Thread(
            target=_parquet_reader, args=(pending, prefetch_q), daemon=True
        )
        reader_thr.start()

        # ── 主循环：三路重叠（IO预读 / GPU encode / ChromaDB后台写）──
        # 时序：
        #   IO:  [read i][read i+1]  [read i+2]
        #   GPU:        [══encode i══][══encode i+1══][══encode i+2══]
        #   DB:                [write i-1] [write i]  [write i+1]
        executor  = ThreadPoolExecutor(max_workers=1)
        write_fut = None
        prev_bf   = None
        prev_n    = 0

        i = 0
        while True:
            item = prefetch_q.get()
            if item is None:
                break
            bf, df, err = item

            if err is not None:
                self.log.warning(f"读取失败，跳过 {bf.name}: {err}")
                i += 1
                continue

            if df.empty:
                if write_fut is not None:
                    write_fut.result()
                    done_files.add(str(prev_bf))
                    self._save_embed_checkpoint(batch_dir, done_files)
                    write_fut = None
                done_files.add(str(bf))
                self._save_embed_checkpoint(batch_dir, done_files)
                prev_bf = None
                i += 1
                continue

            # GPU encode（IO预读线程同时在准备 i+1, i+2）
            embs = self.embedder.embed_documents(df["text"].tolist())

            # 等待上一批 ChromaDB 写入完成，记录断点
            if write_fut is not None:
                write_fut.result()
                done_files.add(str(prev_bf))
                self._save_embed_checkpoint(batch_dir, done_files)
                total_new += prev_n

            # 滚动 token 统计
            if "token_count" in df.columns:
                tc = df["token_count"].dropna()
                tok_sum += int(tc.sum())
                tok_cnt += len(tc)

            prev_bf   = bf
            prev_n    = len(df)
            # 提交写入到后台线程，主线程立刻开始 encode 下一批
            write_fut = executor.submit(_write_chroma, df, embs)

            # 进度日志（每 10 批或最后一批）
            done_batches = skipped + i + 1
            if (i + 1) % 10 == 0 or (i + 1) == len(pending):
                elapsed = time.time() - t0
                speed   = (total_new + prev_n) / elapsed if elapsed > 0 else 0
                eta_s   = (len(pending) - i - 1) * prev_n / speed if speed > 0 else 0
                self.log.info(
                    f"  [{done_batches:,}/{len(batch_files):,}批]  "
                    f"新增≈{total_new + prev_n:,}  "
                    f"速度={speed:.0f}chunks/s  ETA={eta_s/3600:.1f}h"
                )
            i += 1

        reader_thr.join()

        # 等待最后一批写入
        if write_fut is not None:
            write_fut.result()
            done_files.add(str(prev_bf))
            self._save_embed_checkpoint(batch_dir, done_files)
            total_new += prev_n

        executor.shutdown(wait=False)

        elapsed_total = time.time() - t0
        n_indexed     = self.collection.count()

        # 清除断点文件（全部完成）
        ckpt = self._embed_checkpoint_path(batch_dir)
        if ckpt.exists():
            ckpt.unlink()
            self.log.info(f"embedding 断点文件已清除：{ckpt}")

        stats = {
            "collection_name":     self.collection_name,
            "total_chunks":        n_indexed,
            "new_chunks_this_run": total_new,
            "embedding_model":     self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "index_built_at":      pd.Timestamp.now().isoformat(),
            "elapsed_seconds":     round(elapsed_total, 1),
            "db_dir":              str(self.db_dir),
            "batch_dir":           str(batch_dir),
            "batch_files_total":   len(batch_files),
            "chunk_size_stats": {
                "mean": round(tok_sum / tok_cnt, 1),
            } if tok_cnt > 0 else {},
            "metadata_fields": _STR_META_FIELDS + _INT_META_FIELDS,
        }

        self.log.info(
            f"\n{'=' * 60}\n"
            f"  Embedding 完成\n"
            f"  本次新增向量   : {total_new:,}\n"
            f"  集合总向量数   : {n_indexed:,}\n"
            f"  耗时           : {elapsed_total:.1f}s\n"
            f"{'=' * 60}"
        )
        return stats

    # ── 查询 ──────────────────────────────────────────────────────
    def query(
        self,
        query_text:   str,
        n_results:    int          = 10,
        where_filter: dict | None  = None,
    ) -> dict:
        """
        语义检索，支持元数据过滤。

        参数：
            query_text   : 查询文本（自动添加 BGE 检索指令）
            n_results    : 返回结果数量
            where_filter : ChromaDB 元数据过滤条件，如
                           {"imrad_type": "methods"}
                           {"chunk_type": "body", "pub_year": {"$gte": 2020}}

        返回：
            {
              "query": str,
              "n_results": int,
              "results": [
                  {
                    "rank": int,
                    "chunk_id": str,
                    "distance": float,     # 余弦距离（越低越相似）
                    "similarity": float,   # 1 - distance
                    "text_preview": str,   # 前 200 字符
                    "metadata": dict,
                  }, ...
              ]
            }
        """
        if self.embedder is None:
            raise RuntimeError("未提供 BGEEmbedder，无法生成查询向量")

        query_text = (query_text or "").strip()
        if not query_text:
            self.log.warning("查询文本为空，返回空结果")
            return {"query": "", "n_results": 0, "results": []}

        # 生成查询向量（含 BGE 指令前缀）
        q_emb = self.embedder.embed_query(query_text)

        # ChromaDB 查询（余弦距离，越低越相似）
        kwargs: dict = {
            "query_embeddings": [q_emb],
            "n_results":        min(n_results, self.collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        raw = self.collection.query(**kwargs)

        # 整理结果
        ids       = raw["ids"][0]
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            results.append({
                "rank":         rank,
                "chunk_id":     cid,
                "distance":     round(float(dist), 6),
                "similarity":   round(1.0 - float(dist), 6),
                "text_preview": doc[:200] if doc else "",
                "metadata":     meta,
            })

        return {
            "query":    query_text,
            "n_results": len(results),
            "results":  results,
        }

    # ── 保存统计信息 ──────────────────────────────────────────────
    def save_stats(self, stats: dict, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)
        self.log.info(f"索引统计已保存：{output_path}")


# ══════════════════════════════════════════════════════════════════
# 第 4 节：质量验证
# ══════════════════════════════════════════════════════════════════

def validate_index(
    index:       PMCVectorIndex,
    chunks_df:   pd.DataFrame,
    validate_n:  int                = 50,
    log:         logging.Logger | None = None,
) -> dict:
    """
    三项质量验证：
      1. 基础统计验证   — 向量总数 + 抽样元数据内容检查
      2. 自相似性验证   — 用已入库文本查询，验证 Top-1 返回自身（召回率）
      3. 边界情况验证   — 空查询 / 超长查询的鲁棒性

    返回验证报告 dict。
    """
    _p   = log.info if log else print
    report: dict = {}

    _p(f"\n{'=' * 60}")
    _p("向量索引质量验证")
    _p("=" * 60)

    # ── 1. 基础统计验证 ──────────────────────────────────────────
    _p("\n[1] 基础统计验证")
    n_indexed = index.collection.count()
    n_input   = len(chunks_df)
    match     = n_indexed == n_input
    _p(f"  索引向量数 : {n_indexed:,}")
    _p(f"  输入 chunks: {n_input:,}")
    _p(f"  数量匹配   : {'✅' if match else '⚠️'} {'一致' if match else '不一致'}")

    # 抽取前 3 条，验证元数据字段是否正确写入
    sample_get = index.collection.get(
        limit   = 3,
        include = ["documents", "metadatas"],
    )
    meta_ok = True
    for i, (sid, doc, meta) in enumerate(zip(
        sample_get["ids"], sample_get["documents"], sample_get["metadatas"]
    )):
        _p(f"\n  样本 [{i}]  id={sid}")
        _p(f"    chunk_type  : {meta.get('chunk_type', '?')}")
        _p(f"    imrad_type  : {meta.get('imrad_type', '?')}")
        _p(f"    section_path: {meta.get('section_path', '?')}")
        _p(f"    journal     : {meta.get('journal', '?')}")
        _p(f"    pub_year    : {meta.get('pub_year', '?')}")
        _p(f"    token_count : {meta.get('token_count', '?')}")
        _p(f"    text[:80]   : {doc[:80] if doc else 'N/A'}…")
        if not meta.get("doc_id"):
            meta_ok = False

    report["basic_stats"] = {
        "n_indexed":      n_indexed,
        "n_input":        n_input,
        "count_match":    match,
        "metadata_ok":    meta_ok,
    }

    # ── 2. 自相似性验证 ──────────────────────────────────────────
    _p(f"\n[2] 自相似性验证（n={validate_n}）")
    _p("  取已入库文本作为查询，Top-1 应返回自身（距离接近 0）")

    sample_n   = min(validate_n, len(chunks_df))
    sample_df  = chunks_df.sample(sample_n, random_state=42)
    self_hits  = 0
    distances  = []

    for _, row in sample_df.iterrows():
        result = index.query(row["text"], n_results=1)
        if not result["results"]:
            continue
        top1       = result["results"][0]
        expected   = row["chunk_id"]
        got        = top1["chunk_id"]
        dist       = top1["distance"]
        distances.append(dist)
        if got == expected:
            self_hits += 1

    hit_rate = self_hits / sample_n * 100 if sample_n > 0 else 0
    mean_dist = float(np.mean(distances)) if distances else -1
    max_dist  = float(np.max(distances))  if distances else -1

    flag = "✅" if hit_rate >= 95 else ("⚠️" if hit_rate >= 80 else "❌")
    _p(f"  {flag} Top-1 命中率 : {self_hits}/{sample_n} = {hit_rate:.1f}%")
    _p(f"     自相似距离均值  : {mean_dist:.6f}（目标 ≈ 0）")
    _p(f"     自相似距离最大值: {max_dist:.6f}")

    report["self_similarity"] = {
        "sample_n":   sample_n,
        "self_hits":  self_hits,
        "hit_rate":   round(hit_rate, 2),
        "mean_dist":  round(mean_dist, 6),
        "max_dist":   round(max_dist, 6),
        "pass":       hit_rate >= 95,
    }

    # ── 3. 边界情况验证 ──────────────────────────────────────────
    _p("\n[3] 边界情况验证")
    edge_results: list[dict] = []

    # 3a. 空查询
    _p("  3a. 空查询")
    try:
        r = index.query("", n_results=3)
        status = "✅" if r["n_results"] == 0 else "⚠️"
        _p(f"  {status} 空查询返回 {r['n_results']} 条（预期 0 条）")
        edge_results.append({"case": "empty_query", "n_results": r["n_results"], "pass": r["n_results"] == 0})
    except Exception as e:
        _p(f"  ✅ 空查询抛出异常（正常）：{type(e).__name__}: {e}")
        edge_results.append({"case": "empty_query", "exception": str(e), "pass": True})

    # 3b. 极长查询（超过模型 512 token 上限）
    _p("  3b. 超长查询（2000 字符）")
    long_query = ("cardiovascular disease treatment outcomes " * 50)[:2000]
    try:
        r = index.query(long_query, n_results=3)
        _p(f"  ✅ 超长查询正常处理，返回 {r['n_results']} 条结果")
        if r["results"]:
            _p(f"     Top-1 similarity={r['results'][0]['similarity']:.4f}")
        edge_results.append({"case": "long_query", "n_results": r["n_results"], "pass": True})
    except Exception as e:
        _p(f"  ⚠️ 超长查询异常：{type(e).__name__}: {e}")
        edge_results.append({"case": "long_query", "exception": str(e), "pass": False})

    # 3c. 元数据过滤查询
    _p("  3c. 元数据过滤（imrad_type=methods）")
    try:
        r = index.query(
            "statistical analysis methods used in study",
            n_results=5,
            where_filter={"imrad_type": "methods"},
        )
        all_methods = all(res["metadata"].get("imrad_type") == "methods" for res in r["results"])
        flag = "✅" if all_methods else "⚠️"
        _p(f"  {flag} 过滤查询返回 {r['n_results']} 条，全为 methods 类型：{all_methods}")
        if r["results"]:
            for res in r["results"][:3]:
                _p(f"     [{res['rank']}] sim={res['similarity']:.4f}  "
                   f"sec={res['metadata'].get('section_title','?')!r}  "
                   f"text={res['text_preview'][:60]}…")
        edge_results.append({"case": "metadata_filter", "n_results": r["n_results"], "all_correct": all_methods, "pass": all_methods})
    except Exception as e:
        _p(f"  ⚠️ 元数据过滤异常：{type(e).__name__}: {e}")
        edge_results.append({"case": "metadata_filter", "exception": str(e), "pass": False})

    report["edge_cases"] = edge_results

    # ── 汇总 ─────────────────────────────────────────────────────
    all_pass = (
        report["basic_stats"]["count_match"]
        and report["self_similarity"]["pass"]
        and all(e.get("pass", False) for e in edge_results)
    )
    report["overall_pass"] = all_pass

    _p(f"\n{'=' * 60}")
    _p(f"验证总结：{'✅ 全部通过' if all_pass else '⚠️ 存在问题，请查看上方详情'}")
    _p("=" * 60)

    return report


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="PMC oa_comm 向量索引构建 Pipeline")
    # 输入：单文件 或 批次目录（二选一）
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input",      help="单个 chunks Parquet 文件路径")
    grp.add_argument("--input-dir",  help="批次目录（含 batch_*.parquet），逐批 embed，节省内存")
    ap.add_argument("--db-dir",      default=DEFAULT_DB_DIR,     help="ChromaDB 持久化目录")
    ap.add_argument("--collection",  default=DEFAULT_COLLECTION, help="集合名称")
    ap.add_argument("--model",       default=DEFAULT_MODEL,      help="嵌入模型名称")
    ap.add_argument("--embed-batch", type=int, default=EMBED_BATCH_SIZE,  help="嵌入批次大小")
    ap.add_argument("--chroma-batch",type=int, default=CHROMA_BATCH_SIZE, help="ChromaDB 写入批次大小")
    ap.add_argument("--device",      default="auto",  choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit",       type=int, default=None, help="处理 chunk 数上限（单文件模式，测试用）")
    ap.add_argument("--validate-n",  type=int, default=50,  help="自相似性验证抽样数（0=跳过）")
    ap.add_argument("--overwrite",   action="store_true", help="清空并重建集合")
    ap.add_argument("--resume",      action="store_true", help="（--input-dir 模式）跳过已完成的批次")
    args = ap.parse_args()

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    log_path = db_dir / f"vector_index_{args.collection}.log"
    log      = setup_logging(log_path)
    log.info(f"日志文件：{log_path}")
    log.info(f"ChromaDB：{db_dir}  集合：{args.collection}")

    # ── 初始化模型 ────────────────────────────────────────────────
    embedder = BGEEmbedder(
        model_name = args.model,
        batch_size = args.embed_batch,
        device     = args.device,
        log        = log,
    )

    index = PMCVectorIndex(
        db_dir          = db_dir,
        collection_name = args.collection,
        embedder        = embedder,
        log             = log,
    )

    # ── 模式分支 ──────────────────────────────────────────────────
    if args.input_dir:
        # 批次目录模式：逐批 embed，内存友好，支持断点续传
        log.info(f"批次目录模式：{args.input_dir}  resume={args.resume}")
        stats     = index.build_from_dir(
            batch_dir    = args.input_dir,
            chroma_batch = args.chroma_batch,
            overwrite    = args.overwrite,
            resume       = args.resume,
        )
        chunks_df = pd.read_parquet(                                  # 取第一个批次做验证
            sorted(Path(args.input_dir).glob("batch_*.parquet"))[0]
        ).head(args.validate_n * 2) if args.validate_n > 0 else pd.DataFrame()

    else:
        # 单文件模式（小数据集 / 测试）
        log.info(f"单文件模式：{args.input}")
        chunks_df = pd.read_parquet(args.input)
        log.info(f"已加载 {len(chunks_df):,} chunks")
        if args.limit:
            chunks_df = chunks_df.head(args.limit).copy()
            log.info(f"限制处理量：{len(chunks_df):,} chunks")
        stats = index.build(
            chunks_df    = chunks_df,
            chroma_batch = args.chroma_batch,
            overwrite    = args.overwrite,
        )

    # ── 保存统计 ──────────────────────────────────────────────────
    stats_path = db_dir / f"index_stats_{args.collection}.json"
    index.save_stats(stats, stats_path)
    log.info(f"  向量总数 : {stats['total_chunks']:,}")
    log.info(f"  嵌入模型 : {stats['embedding_model']}  维度={stats['embedding_dimension']}")

    # ── 质量验证 ──────────────────────────────────────────────────
    if args.validate_n > 0 and not chunks_df.empty:
        val_report = validate_index(
            index      = index,
            chunks_df  = chunks_df,
            validate_n = min(args.validate_n, len(chunks_df)),
            log        = log,
        )
        val_path = db_dir / f"validation_report_{args.collection}.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_report, f, ensure_ascii=True, indent=2)
        log.info(f"验证报告已保存：{val_path}")

    # ── 示例查询 ──────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("示例查询演示")
    log.info("=" * 60)

    demo_queries = [
        ("What are the effects of COVID-19 on cardiovascular health?", None),
        ("statistical methods used for survival analysis",              {"imrad_type": "methods"}),
        ("CRISPR gene editing cancer therapy",                          {"chunk_type": "body"}),
    ]

    for q_text, q_filter in demo_queries:
        log.info(f"\n查询: {q_text}")
        if q_filter:
            log.info(f"过滤: {q_filter}")
        results = index.query(q_text, n_results=3, where_filter=q_filter)
        for r in results["results"]:
            log.info(
                f"  [{r['rank']}] sim={r['similarity']:.4f}  "
                f"type={r['metadata'].get('chunk_type','?')}  "
                f"imrad={r['metadata'].get('imrad_type','?')}  "
                f"journal={r['metadata'].get('journal','?')!r}"
            )
            log.info(f"       {r['text_preview'][:100]}…")

    log.info("\n完成。")


if __name__ == "__main__":
    main()
