"""
PMC oa_comm 文档解析与全文分割 Pipeline
=====================================
支持两种模式：
  默认模式  ：仅切割摘要（策略 A / B，与原版相同）
  全文模式  ：摘要 + 正文（--fulltext，新增策略 C / D）

全文切分策略（基于实测数据）：
  摘要 → 策略 A（整体不分割，≤512 tokens）/ 策略 B（Recursive 切，>512 tokens）
  正文 → 策略 C（段落合并，同 section 内合并至 merge_target=300 tokens）
          + 策略 D（合并后仍 >512 tokens → Recursive 二级切割）

输入：
  - oa_comm_filelist_merged.csv
  - pmc_xml_extracted/

输出：
  - pipeline_output/chunks_<split>.parquet
  - pipeline_output/pipeline_stats_<split>.json
  - pipeline_output/quality_report_<split>.json
  - pipeline_output/splitter_<split>.log

用法：
  # 仅摘要（原有行为）
  python pmc_document_splitter.py --split test --limit 500

  # 全文模式
  python pmc_document_splitter.py --split ft_test --limit 200 --fulltext

  # Windows 控制台需设置 UTF-8
  $env:PYTHONUTF8="1"; python pmc_document_splitter.py --split full --fulltext
"""

# ══════════════════════════════════════════════════════════════════
# 第 1 节：环境准备与数据加载
# ══════════════════════════════════════════════════════════════════

import argparse
import json
import logging
import re
import sys
import time
from itertools import groupby
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 摘要切分常量（来自数据集分析报告 §4）──────────────────────────
EMBED_LIMIT   = 512    # 嵌入模型 token 上限
CHUNK_SIZE    = 400    # 滑动窗口块大小
CHUNK_OVERLAP = 80     # 重叠 token 数（≈ 20%）
SEPARATORS    = ["\n\n", "\n", ". ", " "]

# ── 全文切分常量（基于实测：段落 p95=508 tokens，99% 段落无需切割）──
PARA_MERGE_TARGET = 300   # 同 section 内段落合并目标（tokens）

# ── 文章类型过滤 ──────────────────────────────────────────────────
KEEP_TYPES = {
    "research-article", "review-article", "systematic-review",
    "case-report", "brief-report", "methods-article", "data-paper",
    "rapid-communication", "protocol", "other",
}
DROP_TYPES = {
    "correction", "retraction", "editorial", "letter", "book-review",
    "news", "reply", "article-commentary", "discussion", "abstract",
    "product-review", "meeting-report",
}

# ── 期刊名标准化 ──────────────────────────────────────────────────
JOURNAL_ALIASES: dict[str, str] = {
    "PLOS One": "PLoS ONE",
    "PLOS ONE": "PLoS ONE",
    "Plos One": "PLoS ONE",
}

# ── 正文噪声 section 关键词（跳过这些章节）────────────────────────
SKIP_SECTIONS: frozenset[str] = frozenset({
    "reference", "acknowledge", "conflict", "competing",
    "funding", "author contribution", "supplementary",
    "appendix", "abbreviation", "declaration",
    "ethical approval", "data availability",
    "supporting information", "additional file",
})

# ── IMRaD 分类关键词 ──────────────────────────────────────────────
_IMRAD_KEYWORDS: dict[str, frozenset[str]] = {
    "introduction":  frozenset({"introduction", "background", "objective", "aim", "rationale", "hypothesis", "overview"}),
    "methods":       frozenset({"method", "material", "patient", "cohort", "procedure", "protocol",
                                 "design", "participant", "subject", "experiment", "statistical", "statistic", "analys"}),
    "results":       frozenset({"result", "finding", "outcome", "observation"}),
    "discussion":    frozenset({"discussion", "conclusion", "interpretation", "implication",
                                 "limitation", "future", "suggest", "recommend"}),
}

# ── body 中仅取 caption 的标签 ────────────────────────────────────
_CAPTION_ONLY_TAGS: frozenset[str] = frozenset({"table-wrap", "fig", "supplementary-material"})
# ── body 中完全跳过的标签 ─────────────────────────────────────────
_SKIP_BODY_TAGS: frozenset[str] = frozenset({
    "title", "label", "ref-list", "ack", "fn-group",
    "app-group", "back", "glossary", "def", "xref",
})


# ── 日志（控制台 + 文件双输出）───────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = logging.getLogger("pmc_splitter")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(logging.Formatter(fmt))
    return log


# ── XML 工具 ─────────────────────────────────────────────────────

def _all_text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _clean_title(raw: str) -> str:
    """去除章节标题中的前缀编号，如 '2.1 Study Design' → 'Study Design'。"""
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", raw).strip() or raw


def classify_imrad(title: str) -> str:
    """将章节标题归类为 introduction / methods / results / discussion / other。"""
    t = _clean_title(title).lower()
    for imrad, keywords in _IMRAD_KEYWORDS.items():
        if any(k in t for k in keywords):
            return imrad
    return "other"


def is_noise_section(title: str) -> bool:
    """判断该 section 是否为噪声章节（参考文献、致谢等），应跳过。"""
    t = _clean_title(title).lower()
    return any(k in t for k in SKIP_SECTIONS)


def parse_xml(path: Path) -> tuple[dict, ET.Element] | None:
    """
    解析单篇 JATS XML。
    返回 (metadata_dict, xml_root)；解析失败或无可用内容则返回 None。

    metadata_dict 字段：
      pmc_id, pmid, doi, title, journal, pub_year,
      article_type, abstract, abstract_source, source_url
    """
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

    # ── pub_year（三级降级）──
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

    # ── Abstract（两阶段清洗）──
    abs_el          = root.find(".//abstract")
    abstract        = _all_text(abs_el) if abs_el is not None else None
    abstract_source = "original"

    if not abstract or len(abstract) < 50:
        body_el = root.find(".//body")
        if body_el is not None:
            body_text = _all_text(body_el)
            if body_text:
                abstract        = body_text[:500]
                abstract_source = "body_fallback"
            else:
                return None
        else:
            return None

    # ── 溯源 URL（三级兜底）──
    pmid_valid = pmid and pmid.strip() not in ("", "0") and pmid.strip().isdigit()
    if pmid_valid:
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif doi:
        source_url = f"https://doi.org/{doi}"
    elif pmc_id:
        source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
    else:
        source_url = None
    if not pmid_valid:
        pmid = None

    meta = {
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
    return meta, root


# ══════════════════════════════════════════════════════════════════
# 第 2 节：文本分割策略实施
# ══════════════════════════════════════════════════════════════════

class PMCDocumentSplitter:
    """
    PMC 文献分割器。

    摘要策略（来自数据集分析报告 §4）：
      策略 A — 整体不分割：标题 + 摘要 ≤ 512 tokens（91.3% 文献）
      策略 B — Recursive 切：超出 512 tokens

    全文正文策略：
      策略 C — 段落合并：同 section 内段落合并至 merge_target（≤512 tokens）
      策略 D — Recursive 二级切：合并后仍超出 512 tokens
    """

    def __init__(
        self,
        embed_limit:   int = EMBED_LIMIT,
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        merge_target:  int = PARA_MERGE_TARGET,
    ):
        self.embed_limit  = embed_limit
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap
        self.merge_target = merge_target

        # 加载 tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # 初始化分割器（摘要策略 B 和正文策略 D 共用）
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size        = chunk_size,
            chunk_overlap     = chunk_overlap,
            length_function   = self._count_tokens,
            separators        = SEPARATORS,
            is_separator_regex= False,
        )

    # ── Token 计数 ────────────────────────────────────────────────
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int) -> str:
        return f"{doc_id}_chunk_{index:04d}"

    # ─────────────────────────────────────────────────────────────
    # 2a. 摘要切分（策略 A / B）
    # ─────────────────────────────────────────────────────────────

    def _no_split(self, document: dict) -> list[dict]:
        """策略 A：标题 + 摘要整体入库，chunk_id = doc_id。"""
        full_text = f"{document['title']}\n\n{document['abstract']}"
        return [{
            "chunk_id":      document["doc_id"],
            "text":          full_text,
            "doc_id":        document["doc_id"],
            "chunk_index":   0,
            "total_chunks":  1,
            "source_title":  document["title"],
            "token_count":   self._count_tokens(full_text),
            "chunk_type":    "abstract",
            "section_title": "Abstract",
            "section_path":  "Abstract",
            "imrad_type":    "abstract",
        }]

    def _split_smart(self, document: dict) -> list[dict]:
        """策略 B：RecursiveCharacterTextSplitter 切割摘要，每块保留标题前缀。"""
        title  = document["title"]
        texts  = self.splitter.split_text(document["abstract"])
        chunks = []
        for i, text in enumerate(texts):
            full_text = f"{title}\n\n{text}"
            chunks.append({
                "chunk_id":      self._make_chunk_id(document["doc_id"], i),
                "text":          full_text,
                "doc_id":        document["doc_id"],
                "chunk_index":   i,
                "total_chunks":  len(texts),
                "source_title":  title,
                "token_count":   self._count_tokens(full_text),
                "chunk_type":    "abstract",
                "section_title": "Abstract",
                "section_path":  "Abstract",
                "imrad_type":    "abstract",
            })
        return chunks

    def chunk_document(self, document: dict) -> tuple[list[dict], str]:
        """摘要分割决策：≤ embed_limit → A，否则 → B。"""
        full_text = f"{document['title']}\n\n{document['abstract']}"
        if self._count_tokens(full_text) <= self.embed_limit:
            return self._no_split(document), "A_no_split"
        return self._split_smart(document), "B_sliding_window"

    # ─────────────────────────────────────────────────────────────
    # 2b. 正文切分（策略 C / D）
    # ─────────────────────────────────────────────────────────────

    def _walk_body(
        self,
        el: ET.Element,
        path: list[str],
        results: list[tuple[str, str, str]],
    ) -> None:
        """
        递归遍历 JATS <body>，收集 (text, section_title, section_path) 三元组。
        - <sec> 更新 path 后递归
        - <table-wrap>/<fig> 只取 <caption> 文本
        - 其他内容标签取全文
        - 跳过 <title>、参考文献、致谢等噪声标签
        """
        for child in el:
            tag = child.tag

            if tag == "sec":
                raw_title   = _all_text(child.find("title"))
                clean_title = _clean_title(raw_title)
                if is_noise_section(clean_title):
                    continue
                new_path = path + [clean_title] if clean_title else path
                self._walk_body(child, new_path, results)

            elif tag in _SKIP_BODY_TAGS:
                continue

            elif tag in _CAPTION_ONLY_TAGS:
                cap = child.find(".//caption")
                if cap is not None:
                    text = _all_text(cap)
                    if len(text) > 20:
                        label     = "Table" if tag == "table-wrap" else "Figure"
                        sec_title = path[-1] if path else ""
                        sec_path  = " > ".join(path) if path else ""
                        results.append((f"[{label}] {text}", sec_title, sec_path))

            else:
                text = "".join(child.itertext()).strip()
                if len(text) > 30:
                    sec_title = path[-1] if path else ""
                    sec_path  = " > ".join(path) if path else ""
                    results.append((text, sec_title, sec_path))

    def _merge_paragraphs(self, texts: list[str]) -> list[str]:
        """
        将同 section 内的段落滚动合并，直到累积 token 数达到 merge_target。
        各段落之间以 '\\n\\n' 连接。
        """
        groups: list[str] = []
        current: list[str] = []
        current_toks = 0

        for text in texts:
            t = self._count_tokens(text)
            if not current:
                current      = [text]
                current_toks = t
            elif current_toks + t <= self.merge_target:
                current.append(text)
                current_toks += t
            else:
                groups.append("\n\n".join(current))
                current      = [text]
                current_toks = t

        if current:
            groups.append("\n\n".join(current))
        return groups

    def extract_body_chunks(
        self,
        root:     ET.Element,
        doc_id:   str,
        doc_meta: dict,
    ) -> list[dict]:
        """
        从 JATS XML root 提取正文 chunk 列表。

        流程：
          1. _walk_body() 递归收集所有段落 + section 路径
          2. 按 section_path 分组，组内段落合并（_merge_paragraphs）
          3. 合并后 ≤ 512 tokens → 策略 C；否则 Recursive 二级切 → 策略 D
          4. 每块文本格式："{section_title}\\n\\n{merged_text}"

        chunk 字段（在 run() 中统一附加 doc_meta 通用字段）：
          chunk_id, text, doc_id, chunk_index, total_chunks,
          source_title, token_count, chunk_type, section_title,
          section_path, imrad_type, split_strategy
        """
        body = root.find(".//body")
        if body is None:
            return []

        # 收集所有 (text, section_title, section_path)
        raw: list[tuple[str, str, str]] = []
        self._walk_body(body, path=[], results=raw)
        if not raw:
            return []

        chunks: list[dict] = []

        # 按 section_path 分组（JATS 层级保证连续性）
        for sec_path, group_iter in groupby(raw, key=lambda x: x[2]):
            group      = list(group_iter)
            sec_title  = group[0][1]
            top_section = sec_path.split(" > ")[0] if sec_path else ""
            imrad_type  = classify_imrad(top_section)

            merged_groups = self._merge_paragraphs([g[0] for g in group])

            for merged_text in merged_groups:
                prefix    = f"{sec_title}\n\n" if sec_title else ""
                full_text = prefix + merged_text

                if self._count_tokens(full_text) <= self.embed_limit:
                    sub_list = [(full_text, "C_body_merged")]
                else:
                    # 二级切割：对 merged_text 切（避免重复拼接 prefix）
                    sub_texts = self.splitter.split_text(merged_text)
                    sub_list  = [(prefix + s, "D_body_recursive") for s in sub_texts]

                for text, strategy in sub_list:
                    idx = len(chunks)
                    chunks.append({
                        "chunk_id":      f"{doc_id}_body_{idx:05d}",
                        "text":          text,
                        "doc_id":        doc_id,
                        "chunk_index":   idx,
                        "total_chunks":  -1,          # 批量更新
                        "source_title":  doc_meta["title"],
                        "token_count":   self._count_tokens(text),
                        "chunk_type":    "body",
                        "section_title": sec_title,
                        "section_path":  sec_path,
                        "imrad_type":    imrad_type,
                        "split_strategy": strategy,
                    })

        # 回填 total_chunks
        total = len(chunks)
        for c in chunks:
            c["total_chunks"] = total

        return chunks


# ══════════════════════════════════════════════════════════════════
# 第 3 节：分割文档，保存结果
# ══════════════════════════════════════════════════════════════════

class PMCChunkingPipeline:
    """全量 Pipeline：加载 → 解析 → 分割（摘要 ± 正文）→ 保存 → 验证。"""

    FILE_INDEX_NAME = "file_index.json"

    def __init__(
        self,
        manifest_csv:  str  = r"d:\Rag-Med\data\oa_comm_filelist_merged.csv",
        extracted_dir: str  = r"d:\Rag-Med\pmc_xml_extracted",
        output_dir:    str  = r"d:\Rag-Med\pipeline_output",
        embed_limit:   int  = EMBED_LIMIT,
        chunk_size:    int  = CHUNK_SIZE,
        chunk_overlap: int  = CHUNK_OVERLAP,
        merge_target:  int  = PARA_MERGE_TARGET,
        include_body:  bool = False,
        log: logging.Logger | None = None,
    ):
        self.manifest_csv  = Path(manifest_csv)
        self.extracted_dir = Path(extracted_dir)
        self.output_dir    = Path(output_dir)
        self.include_body  = include_body
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log = log or logging.getLogger("pmc_splitter")

        self.doc_splitter = PMCDocumentSplitter(
            embed_limit   = embed_limit,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            merge_target  = merge_target,
        )
        mode = "全文模式（摘要 + 正文）" if include_body else "摘要模式"
        self.log.info(
            f"分割器初始化：{mode}  embed_limit={embed_limit}  "
            f"chunk_size={chunk_size}  chunk_overlap={chunk_overlap}  "
            f"merge_target={merge_target}"
        )

    # ── 数据加载 ─────────────────────────────────────────────────
    def load_manifest(self, limit: int | None = None) -> pd.DataFrame:
        self.log.info(f"加载清单：{self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            "AccessionID":                       "pmc_id",
            "PMID":                              "pmid_csv",
            "Article File":                      "article_file",
            "LastUpdated (YYYY-MM-DD HH:MM:SS)": "last_updated",
        })
        n_before = len(df)
        df = df[df["Retracted"].str.lower() != "yes"].copy()
        self.log.info(f"过滤撤稿：{n_before:,} → {len(df):,} 篇（移除 {n_before - len(df):,} 篇）")
        missing_pmc = df["pmc_id"].isna().sum()
        self.log.info(f"唯一 ID 检查：pmc_id 缺失 {missing_pmc} 篇（应为 0）")
        if limit:
            df = df.head(limit)
            self.log.info(f"限制处理量：{len(df):,} 篇")
        return df.reset_index(drop=True)

    # ── 文件索引 ─────────────────────────────────────────────────
    def build_file_index(self, force_rebuild: bool = False) -> dict:
        cache_path = self.output_dir / self.FILE_INDEX_NAME
        if not force_rebuild and cache_path.exists():
            self.log.info(f"加载文件索引缓存：{cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                idx = json.load(f)
            self.log.info(f"文件索引已加载：{len(idx):,} 个 XML 文件")
            return idx

        self.log.info(f"扫描 {self.extracted_dir}，构建文件索引（首次约需 1–2 分钟）…")
        idx: dict[str, str] = {}
        for xml_path in self.extracted_dir.rglob("*.xml"):
            idx[xml_path.stem] = str(xml_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(idx, f)
        self.log.info(f"文件索引构建完成：{len(idx):,} 个文件 → {cache_path}")
        return idx

    # ── 断点续传：checkpoint 辅助 ────────────────────────────────
    def _batch_dir(self, data_split: str) -> Path:
        d = self.output_dir / f"batches_{data_split}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _checkpoint_path(self, data_split: str) -> Path:
        return self.output_dir / f"checkpoint_{data_split}.json"

    def _load_checkpoint(self, data_split: str) -> dict:
        p = self._checkpoint_path(data_split)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, data_split: str, state: dict) -> None:
        with open(self._checkpoint_path(data_split), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, indent=2)

    # ── 主运行逻辑 ───────────────────────────────────────────────
    def run(
        self,
        data_split:  str        = "full",
        limit:       int | None = None,
        batch_size:  int        = 2000,
        force_index: bool       = False,
        resume:      bool       = False,
    ) -> tuple[None, dict]:
        """
        批量写盘模式：每批处理完后立即写入独立 Parquet，不在内存中累积。
        支持 resume=True 从上次断点继续（跳过已完成的批次）。

        调用 merge_batches(data_split) 将批次文件合并为最终 Parquet。
        调用 sample_chunks(data_split) 获取小样本用于预览和验证。
        """
        t0 = time.time()

        file_index = self.build_file_index(force_rebuild=force_index)
        df_raw     = self.load_manifest(limit=limit)
        batch_dir  = self._batch_dir(data_split)

        # ── 加载断点 ──────────────────────────────────────────
        if resume:
            ckpt = self._load_checkpoint(data_split)
            next_batch_start  = ckpt.get("next_batch_start", 0)
            parsed_ok         = ckpt.get("parsed_ok", 0)
            skipped_type      = ckpt.get("skipped_type", 0)
            skipped_nofile    = ckpt.get("skipped_nofile", 0)
            skipped_parse     = ckpt.get("skipped_parse", 0)
            n_abstract_chunks = ckpt.get("n_abstract_chunks", 0)
            n_body_chunks     = ckpt.get("n_body_chunks", 0)
            strategy_counts   = ckpt.get("strategy_counts", {})
            elapsed_prev      = ckpt.get("elapsed_seconds", 0.0)
            self.log.info(
                f"断点续传：从第 {next_batch_start:,} 篇继续  "
                f"（已完成 {parsed_ok:,} 篇，已有 chunks={n_abstract_chunks+n_body_chunks:,}）"
            )
        else:
            next_batch_start  = 0
            parsed_ok         = 0
            skipped_type      = 0
            skipped_nofile    = 0
            skipped_parse     = 0
            n_abstract_chunks = 0
            n_body_chunks     = 0
            strategy_counts: dict[str, int] = {}
            elapsed_prev      = 0.0

        total = len(df_raw)
        self.log.info(f"开始处理 {total:,} 篇文献，batch_size={batch_size}，include_body={self.include_body}")

        for batch_start in range(0, total, batch_size):
            # 跳过已完成的批次
            if batch_start < next_batch_start:
                continue

            batch_file = batch_dir / f"batch_{batch_start:08d}.parquet"

            # 如果该批次文件已存在（断点续传时），也跳过
            if resume and batch_file.exists():
                self.log.info(f"  跳过已有批次：batch_{batch_start:08d}.parquet")
                continue

            batch        = df_raw.iloc[batch_start : batch_start + batch_size]
            batch_chunks: list[dict] = []

            for _, row in batch.iterrows():
                pmc_id = row["pmc_id"]

                xml_path_str = file_index.get(pmc_id)
                if xml_path_str is None:
                    skipped_nofile += 1
                    continue

                result = parse_xml(Path(xml_path_str))
                if result is None:
                    skipped_parse += 1
                    continue
                parsed, xml_root = result

                art_type = (parsed.get("article_type") or "other").lower()
                if art_type in DROP_TYPES:
                    skipped_type += 1
                    continue

                if not parsed.get("pmid") and pd.notna(row.get("pmid_csv")):
                    csv_pmid = str(row["pmid_csv"]).strip()
                    if csv_pmid and csv_pmid not in ("", "0") and csv_pmid.isdigit():
                        parsed["pmid"]       = csv_pmid
                        parsed["source_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{csv_pmid}/"

                parsed["doc_id"] = pmc_id

                common_meta = {
                    "pmc_id":          parsed["pmc_id"],
                    "pmid":            parsed["pmid"],
                    "doi":             parsed["doi"],
                    "journal":         parsed["journal"],
                    "pub_year":        parsed["pub_year"],
                    "article_type":    parsed["article_type"],
                    "abstract_source": parsed["abstract_source"],
                    "source_url":      parsed["source_url"],
                }

                abs_chunks, abs_strategy = self.doc_splitter.chunk_document(parsed)
                for c in abs_chunks:
                    c.update(common_meta)
                    c["split_strategy"] = abs_strategy
                batch_chunks.extend(abs_chunks)
                n_abstract_chunks += len(abs_chunks)
                strategy_counts[abs_strategy] = strategy_counts.get(abs_strategy, 0) + 1

                if self.include_body:
                    body_chunks = self.doc_splitter.extract_body_chunks(xml_root, pmc_id, parsed)
                    for c in body_chunks:
                        c.update(common_meta)
                    batch_chunks.extend(body_chunks)
                    n_body_chunks += len(body_chunks)
                    for c in body_chunks:
                        s = c["split_strategy"]
                        strategy_counts[s] = strategy_counts.get(s, 0) + 1

                parsed_ok += 1

            # ── 批次写盘（每批独立 Parquet）────────────────────
            if batch_chunks:
                batch_df = pd.DataFrame(batch_chunks)
                for col in ["chunk_index", "total_chunks", "token_count", "pub_year"]:
                    if col in batch_df.columns:
                        batch_df[col] = pd.to_numeric(batch_df[col], errors="coerce").astype("Int64")
                batch_df.to_parquet(batch_file, index=False, compression="snappy")

            # ── 保存断点 ────────────────────────────────────────
            elapsed_now = time.time() - t0
            ckpt_state  = {
                "next_batch_start":  batch_start + batch_size,
                "parsed_ok":         parsed_ok,
                "skipped_type":      skipped_type,
                "skipped_nofile":    skipped_nofile,
                "skipped_parse":     skipped_parse,
                "n_abstract_chunks": n_abstract_chunks,
                "n_body_chunks":     n_body_chunks,
                "strategy_counts":   strategy_counts,
                "elapsed_seconds":   round(elapsed_prev + elapsed_now, 1),
            }
            self._save_checkpoint(data_split, ckpt_state)

            done    = min(batch_start + batch_size, total)
            speed   = done / elapsed_now if elapsed_now > 0 else 0
            eta     = (total - done) / speed if speed > 0 else 0
            self.log.info(
                f"进度 {done:>7,}/{total:,} ({done / total * 100:5.1f}%) | "
                f"chunks={n_abstract_chunks + n_body_chunks:>8,} | "
                f"速度={speed:.0f}篇/s | ETA={eta:.0f}s"
            )

        elapsed_total = elapsed_prev + (time.time() - t0)
        total_chunks  = n_abstract_chunks + n_body_chunks

        stats = {
            "processed_date":        pd.Timestamp.now().isoformat(),
            "data_split":            data_split,
            "include_body":          self.include_body,
            "original_documents":    len(df_raw),
            "parsed_ok":             parsed_ok,
            "skipped_no_file":       skipped_nofile,
            "skipped_article_type":  skipped_type,
            "skipped_parse_error":   skipped_parse,
            "total_chunks":          total_chunks,
            "abstract_chunks":       n_abstract_chunks,
            "body_chunks":           n_body_chunks,
            "chunks_per_doc":        round(total_chunks / parsed_ok, 3) if parsed_ok > 0 else 0,
            "embed_limit":           self.doc_splitter.embed_limit,
            "chunk_size":            self.doc_splitter.chunk_size,
            "chunk_overlap":         self.doc_splitter.chunk_overlap,
            "merge_target":          self.doc_splitter.merge_target,
            "separators":            SEPARATORS,
            "strategy_distribution": strategy_counts,
            "elapsed_seconds":       round(elapsed_total, 1),
            "batch_dir":             str(batch_dir),
        }

        self.log.info(
            f"\n{'=' * 60}\n"
            f"  批次处理完成\n"
            f"  原始文献      : {len(df_raw):,}\n"
            f"  成功解析      : {parsed_ok:,}\n"
            f"  无 XML 文件   : {skipped_nofile:,}\n"
            f"  类型过滤      : {skipped_type:,}\n"
            f"  解析失败      : {skipped_parse:,}\n"
            f"  摘要 chunks   : {n_abstract_chunks:,}\n"
            f"  正文 chunks   : {n_body_chunks:,}\n"
            f"  合计 chunks   : {total_chunks:,}\n"
            f"  chunks/篇     : {stats['chunks_per_doc']}\n"
            f"  策略分布      : {strategy_counts}\n"
            f"  耗时          : {elapsed_total:.1f}s\n"
            f"  批次目录      : {batch_dir}\n"
            f"  → 运行 merge_batches('{data_split}') 合并为最终 Parquet\n"
            f"{'=' * 60}"
        )
        # 清除已完成的断点文件
        ckpt_path = self._checkpoint_path(data_split)
        if ckpt_path.exists():
            ckpt_path.unlink()
            self.log.info(f"断点文件已清除：{ckpt_path}")

        return None, stats

    # ── 合并批次 Parquet（PyArrow 流式，不占用大量内存）─────────
    def merge_batches(self, data_split: str) -> Path:
        """
        将 batches_{data_split}/ 下的所有批次文件流式合并为
        chunks_{data_split}.parquet（snappy 压缩）。

        使用 PyArrow ParquetWriter 逐批写出，内存占用约等于单批文件大小。
        """
        batch_dir   = self._batch_dir(data_split)
        batch_files = sorted(batch_dir.glob("batch_*.parquet"))
        if not batch_files:
            raise FileNotFoundError(f"未找到批次文件：{batch_dir}")

        output_path = self.output_dir / f"chunks_{data_split}.parquet"
        self.log.info(f"合并 {len(batch_files):,} 个批次文件 → {output_path}")

        writer: pq.ParquetWriter | None = None
        total_rows = 0
        for i, bf in enumerate(batch_files):
            table = pq.read_table(bf)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(table)
            if (i + 1) % 100 == 0 or (i + 1) == len(batch_files):
                self.log.info(f"  合并进度 {i+1:,}/{len(batch_files):,}  累计行数={total_rows:,}")

        if writer:
            writer.close()

        size_mb = output_path.stat().st_size / 1e6
        self.log.info(f"合并完成：{output_path}  ({size_mb:.1f} MB，共 {total_rows:,} 行)")
        return output_path

    # ── 加载小样本（用于预览和验证）──────────────────────────────
    def sample_chunks(self, data_split: str, n: int = 1000) -> pd.DataFrame:
        """
        从已完成的批次或合并文件中加载小样本，供 preview / validate 使用。
        优先读取合并后的文件；若合并文件不存在，则读取第一个批次文件。
        """
        merged = self.output_dir / f"chunks_{data_split}.parquet"
        if merged.exists():
            return pd.read_parquet(merged).head(n)

        batch_dir   = self._batch_dir(data_split)
        batch_files = sorted(batch_dir.glob("batch_*.parquet"))
        if batch_files:
            return pd.read_parquet(batch_files[0]).head(n)

        return pd.DataFrame()

    # ── 保存统计 JSON ─────────────────────────────────────────────
    def save(self, stats: dict, data_split: str = "full") -> dict:
        stats_path = self.output_dir / f"pipeline_stats_{data_split}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)
        self.log.info(f"统计报告已保存：{stats_path}")
        return {"stats": str(stats_path)}


# ══════════════════════════════════════════════════════════════════
# 第 4 节：预览结果
# ══════════════════════════════════════════════════════════════════

def preview_chunks(
    chunks_df: pd.DataFrame,
    n:   int = 5,
    log: logging.Logger | None = None,
):
    _p = log.info if log else print
    _p(f"\n{'=' * 60}")
    _p(f"数据集预览（前 {n} 行 / 共 {len(chunks_df):,} 行）")
    _p(f"列：{list(chunks_df.columns)}")
    _p("=" * 60)

    for i, row in chunks_df.head(n).iterrows():
        _p(f"\n[{i}] chunk_id      : {row['chunk_id']}")
        _p(f"    doc_id        : {row['doc_id']}")
        _p(f"    chunk_type    : {row.get('chunk_type', '?')}")
        _p(f"    chunk_index   : {row['chunk_index']} / {row['total_chunks']}")
        _p(f"    token_count   : {row['token_count']}")
        _p(f"    split_strategy: {row.get('split_strategy', '?')}")
        _p(f"    section_title : {row.get('section_title', '?')}")
        _p(f"    section_path  : {row.get('section_path', '?')}")
        _p(f"    imrad_type    : {row.get('imrad_type', '?')}")
        _p(f"    journal       : {row.get('journal', '?')}")
        _p(f"    pub_year      : {row.get('pub_year', '?')}")
        _p(f"    source_url    : {row.get('source_url', '?')}")
        _p(f"    text[:200]    : {str(row['text'])[:200]}…")

    _p("=" * 60)


# ══════════════════════════════════════════════════════════════════
# 第 5 节：质量验证
# ══════════════════════════════════════════════════════════════════

def validate_quality(
    chunks_df:   pd.DataFrame,
    embed_limit: int               = EMBED_LIMIT,
    data_split:  str               = "full",
    output_dir:  Path              = Path(r"d:\Rag-Med\pipeline_output"),
    sample_n:    int               = 300,
    log:         logging.Logger | None = None,
) -> dict:
    """
    质量抽样检查，生成并保存验证报告。

    全体检查：
      - 总块数 & 超模型上限比率
      - Token 分布统计与直方图
      - 按 chunk_type / imrad_type / split_strategy 分布

    抽样检查（sample_n 块）：
      - 空文本 / 标题或 section 标题包含率 / 末尾截断率

    多块文档专项：
      - 对摘要多块文档检查 overlap（策略 B）
    """
    _p = log.info if log else print
    report: dict = {"sample_size": min(sample_n, len(chunks_df))}

    # ── 1. Token 统计 ─────────────────────────────────────────────
    tc = chunks_df["token_count"].dropna().astype(int)
    report["token_stats"] = {
        "mean":   round(float(tc.mean()),             1),
        "median": round(float(tc.median()),           1),
        "p90":    round(float(tc.quantile(0.90)),     1),
        "p95":    round(float(tc.quantile(0.95)),     1),
        "p99":    round(float(tc.quantile(0.99)),     1),
        "max":    int(tc.max()),
        "min":    int(tc.min()),
    }
    over_limit = int((tc > embed_limit).sum())
    report["over_limit"] = {
        "count":   over_limit,
        "percent": round(over_limit / len(chunks_df) * 100, 2),
    }

    # ── 2. 分布统计 ───────────────────────────────────────────────
    if "chunk_type" in chunks_df.columns:
        report["chunk_type_distribution"] = chunks_df["chunk_type"].value_counts().to_dict()
    if "imrad_type" in chunks_df.columns:
        report["imrad_distribution"] = chunks_df["imrad_type"].value_counts().to_dict()
    if "split_strategy" in chunks_df.columns:
        report["strategy_distribution"] = chunks_df["split_strategy"].value_counts().to_dict()

    # ── 3. 抽样质量检查 ───────────────────────────────────────────
    sample = chunks_df.sample(min(sample_n, len(chunks_df)), random_state=42)
    has_prefix = 0
    truncated  = 0
    empty_text = 0

    for _, row in sample.iterrows():
        text       = str(row.get("text",          ""))
        chunk_type = str(row.get("chunk_type",    "abstract"))
        sec_title  = str(row.get("section_title", ""))
        src_title  = str(row.get("source_title",  ""))

        if not text.strip():
            empty_text += 1
            continue

        # 前缀检查：body 块看 section_title，abstract 块看 article title
        if chunk_type == "body":
            words = [w for w in sec_title.split()[:3] if len(w) > 3]
            if not words or any(w in text for w in words):
                has_prefix += 1
        else:
            words = [w for w in src_title.split()[:4] if len(w) > 3]
            if words and any(w in text for w in words):
                has_prefix += 1

        # 截断检查：末尾不应在单词中间断开
        stripped = text.rstrip()
        if stripped and not re.search(r'[.!?;)\w"]$', stripped):
            truncated += 1

    n_s = len(sample)
    report["quality_sample"] = {
        "n":                        n_s,
        "has_prefix_pct":           round(has_prefix / n_s * 100, 1),
        "empty_text_count":         empty_text,
        "possibly_truncated_count": truncated,
        "possibly_truncated_pct":   round(truncated / n_s * 100, 1),
    }

    # ── 4. 摘要多块 overlap 检查（策略 B）───────────────────────
    abs_df    = chunks_df[chunks_df.get("chunk_type", pd.Series("abstract", index=chunks_df.index)) == "abstract"] \
                if "chunk_type" in chunks_df.columns else chunks_df
    multi     = abs_df[abs_df["total_chunks"] > 1]
    report["multi_chunk_abstract"] = {
        "unique_docs":   int(multi["doc_id"].nunique()),
        "total_chunks":  len(multi),
    }

    if len(multi) >= 2:
        sample_doc = multi["doc_id"].value_counts().index[0]
        doc_rows   = multi[multi["doc_id"] == sample_doc].sort_values("chunk_index").to_dict("records")
        checks     = []
        for i in range(len(doc_rows) - 1):
            ta = doc_rows[i]["text"]
            tb = doc_rows[i + 1]["text"]
            tail   = ta[-80:] if len(ta) > 80 else ta
            needle = tail[10:30].strip()
            found  = len(needle) >= 5 and needle in tb
            checks.append({
                "chunk_a": doc_rows[i]["chunk_index"],
                "chunk_b": doc_rows[i + 1]["chunk_index"],
                "overlap_detected": found,
                "tail_of_a": tail[:60],
                "head_of_b": tb[:60],
            })
        report["overlap_check"] = {
            "sample_doc_id": sample_doc,
            "chunks":        len(doc_rows),
            "checks":        checks,
        }

    # ── 5. Token 分布直方图 ───────────────────────────────────────
    bins   = [0, 50, 100, 200, 300, 400, 512, 600, 800, 9999]
    labels = ["<50","50-99","100-199","200-299","300-399","400-511","512-599","600-799",">=800"]
    hist   = {}
    for j, label in enumerate(labels):
        lo, hi    = bins[j], bins[j + 1]
        hist[label] = int((tc >= lo).sum() - (tc >= hi).sum())
    report["token_histogram"] = hist

    # 保存
    quality_path = output_dir / f"quality_report_{data_split}.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    # 打印摘要
    _p(f"\n{'=' * 60}")
    _p("质量验证报告")
    _p("=" * 60)
    ts = report["token_stats"]
    _p(f"  Token 统计  mean={ts['mean']}  median={ts['median']}  p90={ts['p90']}  p95={ts['p95']}  max={ts['max']}")
    ol = report["over_limit"]
    _p(f"  {'⚠️' if ol['count'] else '✅'}  超模型上限  {ol['count']:,} 块 ({ol['percent']}%)")

    if "chunk_type_distribution" in report:
        _p(f"  chunk_type  : {report['chunk_type_distribution']}")
    if "imrad_distribution" in report:
        _p(f"  IMRaD 分布  : {report['imrad_distribution']}")

    qs = report["quality_sample"]
    _p(f"  抽样 {qs['n']} 块  含前缀={qs['has_prefix_pct']}%  疑似截断={qs['possibly_truncated_pct']}%  空文本={qs['empty_text_count']}")

    mc = report["multi_chunk_abstract"]
    _p(f"  摘要多块文档  {mc['unique_docs']:,} 篇 → {mc['total_chunks']:,} 块")

    _p("\n  Token 分布直方图:")
    max_cnt = max(hist.values()) if hist else 1
    for label, cnt in hist.items():
        bar = "█" * int(cnt / max_cnt * 30)
        _p(f"    {label:<10}  {cnt:>8,}  {bar}")

    if "overlap_check" in report:
        oc    = report["overlap_check"]
        ok    = sum(1 for c in oc["checks"] if c["overlap_detected"])
        total = len(oc["checks"])
        flag  = "✅" if ok == total else "⚠️"
        _p(f"\n  {flag} Overlap 检查（doc={oc['sample_doc_id']}）：{ok}/{total} 对检测到重叠")

    if "strategy_distribution" in report:
        _p(f"\n  策略分布: {report['strategy_distribution']}")

    _p(f"\n  质量报告已保存：{quality_path}")
    _p("=" * 60)
    return report


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="PMC oa_comm 文档解析与全文分割 Pipeline（批量写盘 + 断点续传）"
    )
    ap.add_argument("--manifest",       default=r"d:\Rag-Med\data\oa_comm_filelist_merged.csv")
    ap.add_argument("--extracted-dir",  default=r"d:\Rag-Med\pmc_xml_extracted")
    ap.add_argument("--output-dir",     default=r"d:\Rag-Med\pipeline_output")
    ap.add_argument("--split",          default="full",   help="数据集标识（train/val/test/full）")
    ap.add_argument("--limit",          type=int,  default=None,  help="处理文献数上限（测试用）")
    ap.add_argument("--batch-size",     type=int,  default=2000)
    ap.add_argument("--force-index",    action="store_true")
    ap.add_argument("--embed-limit",    type=int,  default=EMBED_LIMIT)
    ap.add_argument("--chunk-size",     type=int,  default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap",  type=int,  default=CHUNK_OVERLAP)
    ap.add_argument("--merge-target",   type=int,  default=PARA_MERGE_TARGET,
                    help="正文段落合并目标 token 数（默认 300）")
    ap.add_argument("--fulltext",       action="store_true",
                    help="启用全文模式：同时切分摘要和正文")
    ap.add_argument("--resume",         action="store_true",
                    help="从上次断点继续（跳过已完成的批次）")
    ap.add_argument("--merge-only",     action="store_true",
                    help="跳过切分，只合并已有批次文件为最终 Parquet")
    ap.add_argument("--no-merge",       action="store_true",
                    help="切分完成后不合并，批次文件留在 batches_<split>/ 目录供 embedding 直接消费")
    ap.add_argument("--preview-n",      type=int,  default=3)
    ap.add_argument("--validate-n",     type=int,  default=100)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"splitter_{args.split}.log"
    log      = setup_logging(log_path)
    log.info(f"日志文件：{log_path}")
    log.info(f"Python：{sys.version.split()[0]}  tiktoken={tiktoken.__version__}")
    log.info(f"全文模式：{'是' if args.fulltext else '否（仅摘要）'}  续传：{'是' if args.resume else '否'}")

    pipeline = PMCChunkingPipeline(
        manifest_csv  = args.manifest,
        extracted_dir = args.extracted_dir,
        output_dir    = args.output_dir,
        embed_limit   = args.embed_limit,
        chunk_size    = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        merge_target  = args.merge_target,
        include_body  = args.fulltext,
        log           = log,
    )

    # ── 切分阶段 ──────────────────────────────────────────────────
    if not args.merge_only:
        _, stats = pipeline.run(
            data_split  = args.split,
            limit       = args.limit,
            batch_size  = args.batch_size,
            force_index = args.force_index,
            resume      = args.resume,
        )
        pipeline.save(stats, data_split=args.split)
    else:
        log.info("--merge-only 模式：跳过切分，直接合并批次文件")
        stats = {}

    # ── 合并阶段（可跳过）────────────────────────────────────────
    batch_dir = pipeline._batch_dir(args.split)
    if args.no_merge:
        log.info(
            f"--no-merge 模式：跳过合并，批次文件保留在 {batch_dir}\n"
            f"下一步运行 embedding：\n"
            f"  python pmc_vector_index.py --input-dir {batch_dir} "
            f"--collection pmc_{args.split} --resume"
        )
        sample_df = pipeline.sample_chunks(args.split, n=max(args.preview_n, args.validate_n))
    else:
        log.info("开始合并批次文件…")
        merged_path = pipeline.merge_batches(args.split)
        sample_df   = pipeline.sample_chunks(args.split, n=max(args.preview_n, args.validate_n))
        log.info(f"合并完成：{merged_path}")

    # ── 预览 & 验证 ───────────────────────────────────────────────
    if sample_df.empty:
        log.error("样本为空，无法预览或验证。")
        sys.exit(1)

    if args.preview_n > 0:
        preview_chunks(sample_df, n=args.preview_n, log=log)

    if args.validate_n > 0:
        validate_quality(
            sample_df,
            embed_limit = args.embed_limit,
            data_split  = args.split,
            output_dir  = output_dir,
            sample_n    = min(args.validate_n, len(sample_df)),
            log         = log,
        )

    log.info("\n切分完成。")


if __name__ == "__main__":
    main()
