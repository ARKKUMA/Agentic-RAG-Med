"""
PMC oa_comm 文档解析与分割 Pipeline
=====================================
策略来源：数据集分析.md（PMC oa_comm 数据集分析与预处理报告）

输入：
  - oa_comm_filelist_merged.csv   （清单：AccessionID / PMID / 文件路径）
  - pmc_xml_extracted/            （已解压 XML 文件）

输出：
  - chunks_<split>.parquet        （文本块数据集）
  - pipeline_stats_<split>.json   （处理统计）
  - quality_report_<split>.json   （质量验证）

用法：
  python pmc_chunking_pipeline.py               # 处理全量
  python pmc_chunking_pipeline.py --limit 5000  # 快速测试
  python pmc_chunking_pipeline.py --split train --limit 10000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import tiktoken

# ── 策略常量（来自数据集分析报告）─────────────────────────────
EMBED_LIMIT   = 512   # 嵌入模型 token 上限
CHUNK_SIZE    = 400   # 滑动窗口 chunk 大小（留余量给 title prefix）
CHUNK_OVERLAP = 80    # 重叠 token 数（≈20%）

# 保留的文章类型
KEEP_TYPES = {
    "research-article", "review-article", "systematic-review",
    "case-report", "brief-report", "methods-article", "data-paper",
    "rapid-communication", "protocol", "other",
}
# 丢弃的文章类型（无摘要、无检索价值）
DROP_TYPES = {
    "correction", "retraction", "editorial", "letter", "book-review",
    "news", "reply", "article-commentary", "discussion", "abstract",
    "product-review", "meeting-report",
}

# 期刊名标准化映射（来自分析报告）
JOURNAL_ALIASES = {
    "PLOS One":  "PLoS ONE",
    "PLOS ONE":  "PLoS ONE",
    "Plos One":  "PLoS ONE",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# XML 解析
# ═══════════════════════════════════════════════════════════════

def _all_text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_xml(path: Path) -> dict | None:
    """解析单篇 JATS XML，返回结构化字段字典；失败返回 None。"""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    # ── IDs ──
    pmc_id = pmid = doi = None
    for el in root.findall(".//article-id"):
        t = el.get("pub-id-type", "")
        v = el.text.strip() if el.text else None
        if t == "pmc":    pmc_id = v
        elif t == "pmid": pmid   = v
        elif t == "doi":  doi    = v

    # ── 基础字段 ──
    article_type = root.get("article-type", "other")
    title_el     = root.find(".//article-title")
    title        = _all_text(title_el)
    journal_el   = root.find(".//journal-title")
    journal_raw  = journal_el.text.strip() if journal_el is not None and journal_el.text else None
    journal      = JOURNAL_ALIASES.get(journal_raw, journal_raw)

    # ── pub_year ──
    pub_year = None
    for ptype in ("epub", "ppub", "collection"):
        pd_el = root.find(f".//pub-date[@pub-type='{ptype}']")
        if pd_el is not None:
            yr = pd_el.findtext("year")
            if yr and yr.isdigit():
                pub_year = int(yr)
                break
    if pub_year is None:
        yr = root.findtext(".//pub-date/year")
        if yr and yr.isdigit():
            pub_year = int(yr)

    # ── Abstract ──
    abs_el   = root.find(".//abstract")
    abstract = _all_text(abs_el) if abs_el is not None else None

    # ── Body fallback ──
    abstract_source = "original"
    if not abstract or len(abstract) < 50:
        body_el = root.find(".//body")
        if body_el is not None:
            body_text = _all_text(body_el)
            if body_text:
                # 取前 500 字符作代理摘要
                abstract = body_text[:500]
                abstract_source = "body_fallback"
            else:
                return None  # 无摘要无正文，丢弃
        else:
            return None

    # ── 溯源 URL（三级兜底）──
    if pmid:
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif doi:
        source_url = f"https://doi.org/{doi}"
    elif pmc_id:
        source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
    else:
        source_url = None

    return {
        "pmc_id":          pmc_id,
        "pmid":            pmid,
        "doi":             doi,
        "title":           title,
        "journal":         journal,
        "pub_year":        pub_year,
        "article_type":    article_type,
        "abstract":        abstract,
        "abstract_source": abstract_source,
        "source_url":      source_url,
    }


# ═══════════════════════════════════════════════════════════════
# Pipeline 主类
# ═══════════════════════════════════════════════════════════════

class PMCChunkingPipeline:

    def __init__(
        self,
        manifest_csv: str  = r"d:\Rag-Med\oa_comm_filelist_merged.csv",
        extracted_dir: str = r"d:\Rag-Med\pmc_xml_extracted",
        output_dir: str    = r"d:\Rag-Med\pipeline_output",
        file_index_cache: str = r"d:\Rag-Med\pipeline_output\file_index.json",
        embed_limit: int   = EMBED_LIMIT,
        chunk_size: int    = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.manifest_csv     = Path(manifest_csv)
        self.extracted_dir    = Path(extracted_dir)
        self.output_dir       = Path(output_dir)
        self.file_index_cache = Path(file_index_cache)
        self.embed_limit      = embed_limit
        self.chunk_size       = chunk_size
        self.chunk_overlap    = chunk_overlap

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        log.info("Tokenizer loaded: cl100k_base")

    # ── Token 计数 ──────────────────────────────────────────────
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ── 滑动窗口分割（策略 B）──────────────────────────────────
    def _sliding_window_split(self, text: str) -> list[str]:
        """在 token 级别做滑动窗口分割，保证每块 ≤ chunk_size。"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        step   = self.chunk_size - self.chunk_overlap
        start  = 0
        while start < len(tokens):
            end    = min(start + self.chunk_size, len(tokens))
            chunk  = self.tokenizer.decode(tokens[start:end])
            chunks.append(chunk)
            if end >= len(tokens):
                break
            start += step
        return chunks

    # ── 生成 chunk_id ───────────────────────────────────────────
    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        return f"{doc_id}_chunk_{index:04d}"

    # ── 整体不分割（策略 A）──────────────────────────────────────
    def _no_split(self, document: dict) -> list[dict]:
        full_text = f"{document['title']}\n\n{document['abstract']}"
        return [{
            "chunk_id":      document["doc_id"],   # 文献ID直接作为块ID
            "text":          full_text,
            "doc_id":        document["doc_id"],
            "chunk_index":   0,
            "total_chunks":  1,
            "source_title":  document["title"],
            "token_count":   self._count_tokens(full_text),
        }]

    # ── 滑动窗口（策略 B）──────────────────────────────────────
    def _split_smart(self, document: dict) -> list[dict]:
        title    = document["title"]
        abstract = document["abstract"]
        texts    = self._sliding_window_split(abstract)

        chunks = []
        for i, text in enumerate(texts):
            chunk_id  = self._make_chunk_id(document["doc_id"], i)
            full_text = f"{title}\n\n{text}"   # 每块保留标题
            chunks.append({
                "chunk_id":     chunk_id,
                "text":         full_text,
                "doc_id":       document["doc_id"],
                "chunk_index":  i,
                "total_chunks": len(texts),
                "source_title": title,
                "token_count":  self._count_tokens(full_text),
            })
        return chunks

    # ── 派发分割策略 ─────────────────────────────────────────────
    def chunk_document(self, document: dict) -> list[dict]:
        """
        策略 A（整体不分割）：摘要 + 标题 ≤ embed_limit tokens
        策略 B（滑动窗口）  ：超出 embed_limit tokens
        """
        full_text   = f"{document['title']}\n\n{document['abstract']}"
        token_count = self._count_tokens(full_text)

        if token_count <= self.embed_limit:
            chunks = self._no_split(document)
            strategy = "A_no_split"
        else:
            chunks = self._split_smart(document)
            strategy = "B_sliding_window"

        # 附加元数据到每个 chunk
        meta = {
            "pmc_id":          document["pmc_id"],
            "pmid":            document["pmid"],
            "doi":             document["doi"],
            "journal":         document["journal"],
            "pub_year":        document["pub_year"],
            "article_type":    document["article_type"],
            "abstract_source": document["abstract_source"],
            "source_url":      document["source_url"],
            "split_strategy":  strategy,
        }
        for c in chunks:
            c.update(meta)
        return chunks

    # ── 构建文件索引 ─────────────────────────────────────────────
    def build_file_index(self, force_rebuild: bool = False) -> dict:
        """扫描解压目录，构建 {pmc_id: filepath} 映射并缓存。"""
        if not force_rebuild and self.file_index_cache.exists():
            log.info(f"加载文件索引缓存：{self.file_index_cache}")
            with open(self.file_index_cache, encoding="utf-8") as f:
                return json.load(f)

        log.info(f"扫描 {self.extracted_dir}，构建文件索引（首次运行约需 1–2 分钟）…")
        index = {}
        for xml_path in self.extracted_dir.rglob("*.xml"):
            pmc_id = xml_path.stem  # 文件名去掉 .xml 即 PMC ID
            index[pmc_id] = str(xml_path)

        self.file_index_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_index_cache, "w", encoding="utf-8") as f:
            json.dump(index, f)
        log.info(f"文件索引构建完成：{len(index)} 个 XML 文件 → {self.file_index_cache}")
        return index

    # ── 加载清单 ─────────────────────────────────────────────────
    def load_manifest(self, limit: int | None = None) -> pd.DataFrame:
        log.info(f"加载清单：{self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv, dtype=str)
        df.columns = [c.strip() for c in df.columns]

        # 重命名
        df = df.rename(columns={
            "AccessionID":                      "pmc_id",
            "PMID":                             "pmid_csv",
            "Article File":                     "article_file",
            "LastUpdated (YYYY-MM-DD HH:MM:SS)":"last_updated",
        })

        original_n = len(df)

        # 过滤已撤稿文章
        df = df[df["Retracted"].str.lower() != "yes"].copy()
        log.info(f"过滤撤稿：{original_n} → {len(df)} 篇（移除 {original_n - len(df)} 篇）")

        if limit:
            df = df.head(limit)
            log.info(f"限制处理量：{len(df)} 篇")

        return df.reset_index(drop=True)

    # ── 主运行逻辑 ───────────────────────────────────────────────
    def run(
        self,
        data_split: str   = "full",
        limit: int | None = None,
        batch_size: int   = 2000,
        force_index: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """
        执行完整 pipeline，返回 (chunks_df, stats)。
        """
        t0 = time.time()

        # 1. 构建文件索引
        file_index = self.build_file_index(force_rebuild=force_index)

        # 2. 加载清单
        df_raw = self.load_manifest(limit=limit)

        # 3. 批量处理
        all_chunks  = []
        parsed_ok   = 0
        skipped_type = 0
        skipped_nofile = 0
        skipped_parse  = 0
        strategy_counts = {"A_no_split": 0, "B_sliding_window": 0}

        total = len(df_raw)
        log.info(f"开始处理 {total} 篇文献，batch_size={batch_size}")

        for batch_start in range(0, total, batch_size):
            batch = df_raw.iloc[batch_start:batch_start + batch_size]
            batch_chunks = []

            for _, row in batch.iterrows():
                pmc_id = row["pmc_id"]

                # 查找 XML 文件
                xml_path_str = file_index.get(pmc_id)
                if xml_path_str is None:
                    skipped_nofile += 1
                    continue

                # 解析 XML
                parsed = parse_xml(Path(xml_path_str))
                if parsed is None:
                    skipped_parse += 1
                    continue

                # 文章类型过滤
                art_type = (parsed.get("article_type") or "other").lower()
                if art_type in DROP_TYPES:
                    skipped_type += 1
                    continue

                # 补充 CSV 中的 pmid（XML 的 pmid 可能为空，CSV 的更完整）
                if not parsed.get("pmid") and pd.notna(row.get("pmid_csv")):
                    parsed["pmid"] = row["pmid_csv"]
                    if parsed["pmid"]:
                        parsed["source_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{parsed['pmid']}/"

                # doc_id 使用 pmc_id（0% 缺失率）
                parsed["doc_id"] = pmc_id

                # 执行分割
                chunks = self.chunk_document(parsed)
                for c in chunks:
                    strategy_counts[c["split_strategy"]] = \
                        strategy_counts.get(c["split_strategy"], 0) + 1

                batch_chunks.extend(chunks)
                parsed_ok += 1

            all_chunks.extend(batch_chunks)

            # 进度日志
            done = min(batch_start + batch_size, total)
            elapsed = time.time() - t0
            speed   = done / elapsed if elapsed > 0 else 0
            eta     = (total - done) / speed if speed > 0 else 0
            log.info(
                f"进度 {done:>7}/{total} ({done/total*100:5.1f}%) | "
                f"chunks={len(all_chunks):>7} | "
                f"速度={speed:.0f}篇/s | ETA={eta:.0f}s"
            )

        # 4. 转 DataFrame
        chunks_df = pd.DataFrame(all_chunks)
        if not chunks_df.empty:
            # 类型优化
            for col in ["chunk_index", "total_chunks", "token_count", "pub_year"]:
                if col in chunks_df.columns:
                    chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce").astype("Int64")

        elapsed_total = time.time() - t0

        # 5. 统计
        stats = {
            "processed_date":      pd.Timestamp.now().isoformat(),
            "data_split":          data_split,
            "original_documents":  len(df_raw),
            "parsed_ok":           parsed_ok,
            "skipped_no_file":     skipped_nofile,
            "skipped_article_type": skipped_type,
            "skipped_parse_error": skipped_parse,
            "total_chunks":        len(chunks_df),
            "chunks_per_doc":      round(len(chunks_df) / parsed_ok, 3) if parsed_ok > 0 else 0,
            "strategy_A_no_split":       strategy_counts.get("A_no_split", 0),
            "strategy_B_sliding_window": strategy_counts.get("B_sliding_window", 0),
            "embed_limit":    self.embed_limit,
            "chunk_size":     self.chunk_size,
            "chunk_overlap":  self.chunk_overlap,
            "elapsed_seconds": round(elapsed_total, 1),
            "output_dir":     str(self.output_dir),
        }

        log.info(
            f"\n{'='*60}\n"
            f"  处理完成\n"
            f"  原始文献     : {len(df_raw):,}\n"
            f"  成功解析     : {parsed_ok:,}\n"
            f"  无 XML 文件  : {skipped_nofile:,}\n"
            f"  类型过滤     : {skipped_type:,}\n"
            f"  解析失败     : {skipped_parse:,}\n"
            f"  生成 chunks  : {len(chunks_df):,}\n"
            f"  策略 A       : {strategy_counts.get('A_no_split', 0):,}\n"
            f"  策略 B       : {strategy_counts.get('B_sliding_window', 0):,}\n"
            f"  耗时         : {elapsed_total:.1f}s\n"
            f"{'='*60}"
        )
        return chunks_df, stats

    # ── 保存结果 ─────────────────────────────────────────────────
    def save(
        self,
        chunks_df: pd.DataFrame,
        stats: dict,
        data_split: str = "full",
    ) -> dict:
        """保存数据集（优先 Parquet，无 pyarrow 则降级 JSONL）+ 统计 JSON。"""
        stats_path = self.output_dir / f"pipeline_stats_{data_split}.json"
        paths: dict = {}

        # ── Parquet（推荐，需 pyarrow）──
        try:
            parquet_path = self.output_dir / f"chunks_{data_split}.parquet"
            chunks_df.to_parquet(parquet_path, index=False, compression="snappy")
            size_mb = parquet_path.stat().st_size / 1e6
            log.info(f"Parquet saved: {parquet_path}  ({size_mb:.1f} MB)")
            stats["output_file"] = str(parquet_path)
            stats["output_format"] = "parquet"
            paths["parquet"] = str(parquet_path)
        except ImportError:
            log.warning(
                "pyarrow not found — falling back to JSONL.\n"
                "  Install with:  pip install pyarrow"
            )
            jsonl_path = self.output_dir / f"chunks_{data_split}.jsonl"
            # Int64 → Python int for JSON serialization
            df_out = chunks_df.copy()
            for col in df_out.select_dtypes("Int64").columns:
                df_out[col] = df_out[col].astype(object).where(df_out[col].notna(), None)
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in df_out.to_dict("records"):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            size_mb = jsonl_path.stat().st_size / 1e6
            log.info(f"JSONL saved: {jsonl_path}  ({size_mb:.1f} MB)")
            stats["output_file"] = str(jsonl_path)
            stats["output_format"] = "jsonl"
            paths["jsonl"] = str(jsonl_path)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        log.info(f"Stats saved: {stats_path}")
        paths["stats"] = str(stats_path)

        return paths

    # ── 预览结果 ─────────────────────────────────────────────────
    def preview(self, chunks_df: pd.DataFrame, n: int = 5):
        """打印 n 个样本的格式化预览。"""
        print(f"\n{'='*60}")
        print(f"数据集预览（前 {n} 行）")
        print(f"总行数：{len(chunks_df):,}  列：{list(chunks_df.columns)}")
        print("="*60)
        for i, row in chunks_df.head(n).iterrows():
            print(f"\n[{i}] chunk_id    : {row['chunk_id']}")
            print(f"    doc_id      : {row['doc_id']}")
            print(f"    chunk_index : {row['chunk_index']} / {row['total_chunks']}")
            print(f"    token_count : {row['token_count']}")
            print(f"    strategy    : {row.get('split_strategy', '?')}")
            print(f"    journal     : {row.get('journal', '?')}")
            print(f"    pub_year    : {row.get('pub_year', '?')}")
            print(f"    source_url  : {row.get('source_url', '?')}")
            print(f"    text[:200]  : {str(row['text'])[:200]}…")
        print("="*60)

    # ── 质量验证 ─────────────────────────────────────────────────
    def validate(
        self,
        chunks_df: pd.DataFrame,
        data_split: str = "full",
        sample_n: int   = 200,
    ) -> dict:
        """
        质量抽样检查，返回验证报告 dict 并保存到 JSON。
        检查项：
          - token 是否超模型上限
          - 文本是否包含标题
          - 文本是否被截断（末尾完整性）
          - 多块文档的 overlap 是否存在
        """
        import random, re

        report = {"sample_size": min(sample_n, len(chunks_df))}

        # ── 1. 整体统计 ──
        token_counts = chunks_df["token_count"].dropna().astype(int)
        report["token_stats"] = {
            "mean":   round(float(token_counts.mean()), 1),
            "median": round(float(token_counts.median()), 1),
            "p90":    round(float(token_counts.quantile(0.90)), 1),
            "p95":    round(float(token_counts.quantile(0.95)), 1),
            "p99":    round(float(token_counts.quantile(0.99)), 1),
            "max":    int(token_counts.max()),
            "min":    int(token_counts.min()),
        }

        over_limit = int((token_counts > self.embed_limit).sum())
        report["over_limit"] = {
            "count":   over_limit,
            "percent": round(over_limit / len(chunks_df) * 100, 2),
        }

        # ── 2. 抽样质量检查 ──
        sample = chunks_df.sample(min(sample_n, len(chunks_df)), random_state=42)

        has_title = 0
        truncated = 0    # 文本末尾不以句号/换行结束
        empty_text = 0

        for _, row in sample.iterrows():
            text = str(row.get("text", ""))
            title = str(row.get("source_title", ""))

            if not text.strip():
                empty_text += 1
                continue

            # 标题检查：文本开头应含 source_title 关键词
            title_words = title.split()[:4]
            if title_words and any(w in text for w in title_words if len(w) > 3):
                has_title += 1

            # 截断检查：末尾应以标点、字母或数字结尾（不应在单词中间截断）
            stripped = text.rstrip()
            if stripped and not re.search(r'[.!?;)\w"]$', stripped):
                truncated += 1

        n_sample = len(sample)
        report["quality_sample"] = {
            "n":                   n_sample,
            "has_title_pct":       round(has_title / n_sample * 100, 1),
            "empty_text_count":    empty_text,
            "possibly_truncated_count": truncated,
            "possibly_truncated_pct":   round(truncated / n_sample * 100, 1),
        }

        # ── 3. 多块文档专项检查 ──
        multi_chunk = chunks_df[chunks_df["total_chunks"] > 1]
        report["multi_chunk_docs"] = {
            "total_multi_chunk_chunks": len(multi_chunk),
            "unique_docs": int(multi_chunk["doc_id"].nunique()),
        }

        if len(multi_chunk) >= 2:
            # 按 doc_id + chunk_index 抽取一篇多块文档检查 overlap
            sample_doc_id = multi_chunk["doc_id"].value_counts().index[0]
            doc_chunks = multi_chunk[multi_chunk["doc_id"] == sample_doc_id] \
                             .sort_values("chunk_index")

            overlap_checks = []
            rows = doc_chunks.to_dict("records")
            for i in range(len(rows) - 1):
                text_a = rows[i]["text"]
                text_b = rows[i + 1]["text"]
                # 取 chunk_a 末尾 40 字符，检查是否出现在 chunk_b 开头
                tail   = text_a[-80:] if len(text_a) > 80 else text_a
                overlap_found = tail[:30] in text_b  # 粗略检查
                overlap_checks.append({
                    "chunk_a": rows[i]["chunk_index"],
                    "chunk_b": rows[i + 1]["chunk_index"],
                    "overlap_detected": overlap_found,
                    "tail_of_a": tail[:60],
                    "head_of_b": text_b[:60],
                })
            report["overlap_check"] = {
                "sample_doc_id": sample_doc_id,
                "total_chunks":  len(rows),
                "checks":        overlap_checks,
            }

        # ── 4. 分割策略分布 ──
        if "split_strategy" in chunks_df.columns:
            report["strategy_distribution"] = \
                chunks_df["split_strategy"].value_counts().to_dict()

        # ── 5. token 分布直方图（ASCII）──
        bins = [0, 50, 100, 200, 300, 400, 512, 600, 800, 9999]
        labels = ["<50","50-99","100-199","200-299","300-399","400-511","512-599","600-799","≥800"]
        hist = {}
        for j, label in enumerate(labels):
            lo, hi = bins[j], bins[j + 1]
            cnt = int((token_counts >= lo).sum() - (token_counts >= hi).sum())
            hist[label] = cnt
        report["token_histogram"] = hist

        # 保存
        quality_path = self.output_dir / f"quality_report_{data_split}.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"质量报告已保存：{quality_path}")

        # 打印摘要
        self._print_quality_summary(report)
        return report

    def _print_quality_summary(self, report: dict):
        print(f"\n{'='*60}")
        print("质量验证报告")
        print("="*60)
        ts = report["token_stats"]
        print(f"  Token 统计  mean={ts['mean']}  median={ts['median']}"
              f"  p95={ts['p95']}  p99={ts['p99']}  max={ts['max']}")
        ol = report["over_limit"]
        print(f"  超模型上限  {ol['count']} 块 ({ol['percent']}%)")
        qs = report["quality_sample"]
        print(f"  抽样 {qs['n']} 块  含标题={qs['has_title_pct']}%  "
              f"疑似截断={qs['possibly_truncated_pct']}%  空文本={qs['empty_text_count']}")
        mc = report["multi_chunk_docs"]
        print(f"  多块文档    {mc['unique_docs']} 篇 → {mc['total_multi_chunk_chunks']} 块")
        print("\n  Token 分布直方图:")
        hist = report["token_histogram"]
        max_cnt = max(hist.values()) if hist else 1
        for label, cnt in hist.items():
            bar = "█" * int(cnt / max_cnt * 25)
            print(f"    {label:<10} {cnt:>8,}  {bar}")
        if "overlap_check" in report:
            oc = report["overlap_check"]
            checks = oc.get("checks", [])
            ok = sum(1 for c in checks if c["overlap_detected"])
            print(f"\n  Overlap 检查（doc={oc['sample_doc_id']}，{len(checks)} 对）：{ok}/{len(checks)} 对检测到重叠")
        if "strategy_distribution" in report:
            print(f"\n  策略分布: {report['strategy_distribution']}")
        print("="*60)


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="PMC oa_comm 文档解析与分割 Pipeline")
    ap.add_argument("--manifest",      default=r"d:\Rag-Med\oa_comm_filelist_merged.csv")
    ap.add_argument("--extracted-dir", default=r"d:\Rag-Med\pmc_xml_extracted")
    ap.add_argument("--output-dir",    default=r"d:\Rag-Med\pipeline_output")
    ap.add_argument("--split",         default="full", help="数据集标识（train/val/test/full）")
    ap.add_argument("--limit",         type=int, default=None, help="处理文献数上限（测试用）")
    ap.add_argument("--batch-size",    type=int, default=2000)
    ap.add_argument("--force-index",   action="store_true", help="强制重建文件索引")
    ap.add_argument("--embed-limit",   type=int, default=EMBED_LIMIT)
    ap.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--preview-n",     type=int, default=3)
    ap.add_argument("--validate-n",    type=int, default=300)
    args = ap.parse_args()

    pipeline = PMCChunkingPipeline(
        manifest_csv   = args.manifest,
        extracted_dir  = args.extracted_dir,
        output_dir     = args.output_dir,
        embed_limit    = args.embed_limit,
        chunk_size     = args.chunk_size,
        chunk_overlap  = args.chunk_overlap,
    )

    # 运行
    chunks_df, stats = pipeline.run(
        data_split = args.split,
        limit      = args.limit,
        batch_size = args.batch_size,
        force_index= args.force_index,
    )

    if chunks_df.empty:
        log.error("未生成任何 chunk，请检查输入数据和文件路径。")
        sys.exit(1)

    # 保存
    pipeline.save(chunks_df, stats, data_split=args.split)

    # 预览
    pipeline.preview(chunks_df, n=args.preview_n)

    # 质量验证
    if args.validate_n > 0:
        pipeline.validate(chunks_df, data_split=args.split, sample_n=args.validate_n)


if __name__ == "__main__":
    main()
