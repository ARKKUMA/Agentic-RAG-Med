# PMC oa_comm 文本分割策略

> 依据前序分析数据制定，所有阈值均来自实测（7248 篇，tiktoken cl100k_base）

---

## 1. 数据驱动的策略选择

### 1.1 三种策略的适用条件对照

| 策略 | 触发条件 | 本数据集情况 | 是否适用 |
|------|---------|------------|---------|
| 整体不分割 | p95 < 嵌入上限（512） | p95 = **560** > 512 | ❌ 不完全适用 |
| 重叠滑动窗口 | 存在长尾文档 | 8.4% 超出 512，max=1912 | ✅ 适用于长尾 |
| 语义章节分割 | 有明确章节标题 | 正文 section 结构 100% 清晰；摘要结构化仅 4.6% | ✅ 适用于正文层 |

### 1.2 决策树

```
输入文本
  ├─ 摘要（Abstract）
  │    ├─ ≤ 512 tokens（91.6%）→ 【策略 A】整体不分割
  │    └─ > 512 tokens（8.4%） → 【策略 B】重叠滑动窗口
  │
  └─ 正文（Body）
       ├─ 第一步：按 <sec> 标题切成 section 块
       └─ 第二步：对 > 512 tokens 的 section → 【策略 B】滑动窗口
```

**结论：摘要以"整体不分割"为主，正文以"语义章节 + 滑动窗口"双阶段为主。**

---

## 2. 策略 A — 整体不分割（摘要主体，91.6%）

### 适用范围

| 条件 | 实测值 |
|------|--------|
| 摘要 ≤ 512 tokens 的比例 | **91.6%** |
| 中位数 | 323 tokens |
| p90 | 496 tokens（仍在 512 内）|

### Document 构造

```python
# 每篇文章构造一个 Document，不切割
doc = Document(
    page_content=f"{title}\n\n{abstract}",
    metadata={
        "pmc_id":      pmc_id,
        "pmid":        pmid,           # 可空
        "doi":         doi,            # 可空
        "journal":     journal,        # 标准化后
        "pub_year":    pub_year,       # int，用于时间过滤
        "article_type": article_type,
        "source_url":  source_url,     # 三级兜底链接
        "chunk_type":  "abstract",
        "token_count": token_count,
    }
)
```

### 优点 / 缺点

| 项目 | 说明 |
|------|------|
| ✅ 上下文完整 | 标题 + 摘要在同一 chunk，语义连贯 |
| ✅ 检索精度高 | 无跨 chunk 信息割裂 |
| ✅ 实现最简单 | 无需 Splitter |
| ⚠️ 覆盖 91.6% | 剩余 8.4% 需策略 B 补充 |

---

## 3. 策略 B — 重叠滑动窗口（摘要长尾，8.4%）

### 适用范围

| 条件 | 实测值 |
|------|--------|
| 摘要 > 512 tokens | 8.4%（609 / 7248） |
| p95 | 560 tokens（仅超 48 tokens）|
| p99 | 704 tokens |
| max | 1912 tokens |

### 参数设定

```
chunk_size    = 400    # 留出余量，给 title prefix 和 overlap 空间
chunk_overlap = 80     # overlap ≈ chunk_size × 0.2，保留上下文衔接
```

**参数推导：**
- `chunk_size = 400`：嵌入上限 512，减去 title（约 30–60 tokens）后余量约 450，再留 50 tokens 缓冲
- `chunk_overlap = 80`：约 20%，保证关键语句不因边界断裂而丢失

### 实现（LangChain）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=lambda t: len(enc.encode(t)),   # token 计数，非字符数
    separators=["\n\n", "\n", ". ", " ", ""],        # 优先按句子断
)

def split_abstract(title, abstract, metadata):
    chunks = splitter.split_text(abstract)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=f"{title}\n\n{chunk}",     # 每块都保留 title
            metadata={**metadata,
                      "chunk_type": "abstract",
                      "chunk_index": i,
                      "chunk_total": len(chunks)},
        ))
    return docs
```

> **关键细节**：`length_function` 必须用 tokenizer 而非 `len()`，
> 因为中文或专业术语一个单词可能占多个 token。

---

## 4. 策略 C — 语义章节分割（正文 Body）

### 适用范围

| 条件 | 实测值 |
|------|--------|
| research-article 比例 | 75.8% |
| 有明确 section 结构 | ≈100%（JATS `<sec>` 标签）|
| 单章节中位数 | 416 tokens |
| 单章节 p90 | **2003 tokens**（须二次切割）|
| 全文中位数 | 12170 tokens |

### 两阶段流程

```
阶段 1：语义章节切割
  按 JATS <sec> 标签提取每个 section（标题 + 正文）
  → 每个 section 成为一个候选 chunk

阶段 2：对超长 section 做滑动窗口
  if section_tokens > 512:
      → RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
```

### 实现

```python
def extract_body_chunks(root, base_metadata, enc):
    docs = []
    for sec in root.findall(".//body//sec"):
        # 跳过嵌套子节（只取顶层 section，避免重复）
        if sec.getparent() is not None and sec.getparent().tag == "sec":
            continue

        title_el = sec.find("title")
        sec_title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        sec_text  = "".join(sec.itertext()).strip()
        sec_tok   = len(enc.encode(sec_text))

        metadata = {**base_metadata,
                    "chunk_type":    "body_section",
                    "section_title": sec_title}

        if sec_tok <= 512:
            # 阶段 1：直接作为一个 chunk
            docs.append(Document(
                page_content=f"{sec_title}\n\n{sec_text}",
                metadata={**metadata, "chunk_index": 0, "chunk_total": 1},
            ))
        else:
            # 阶段 2：滑动窗口二次切割
            chunks = splitter.split_text(sec_text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=f"{sec_title}\n\n{chunk}",
                    metadata={**metadata,
                              "chunk_index": i,
                              "chunk_total": len(chunks)},
                ))
    return docs
```

---

## 5. 整体策略汇总

### 5.1 每篇文章产生的 Document 数量估算

| 来源 | 策略 | 估算 chunk 数 / 篇 |
|------|------|-----------------|
| 摘要（≤512） | 策略 A | 1 |
| 摘要（>512） | 策略 B | 2–4 |
| 正文各 section | 策略 C | 4–8（视期刊） |
| **合计（典型文章）** | | **5–9 chunks** |

### 5.2 Metadata 设计（检索过滤器）

```python
metadata = {
    # 身份标识（用于去重 / 溯源）
    "pmc_id":        "PMC3089640",
    "pmid":          "21573075",
    "source_url":    "https://pubmed.ncbi.nlm.nih.gov/21573079/",

    # 检索过滤器
    "journal":       "PLOS One",     # 支持：journal == "Nature Communications"
    "pub_year":      2011,           # 支持：pub_year >= 2021（近5年）
    "article_type":  "research-article",

    # chunk 定位
    "chunk_type":    "abstract",     # "abstract" | "body_section"
    "section_title": "",             # body_section 时填入，如 "Methods"
    "chunk_index":   0,              # 第几块（0-based）
    "chunk_total":   1,              # 该文档共几块
}
```

### 5.3 参数速查表

| 参数 | 摘要（策略 B）| 正文（策略 C）|
|------|-------------|-------------|
| `chunk_size` | 400 tokens | 400 tokens |
| `chunk_overlap` | 80 tokens | 80 tokens |
| `length_function` | tiktoken | tiktoken |
| `separators` | `["\n\n", "\n", ". ", " "]` | 同左 |
| title 前缀 | ✅ 每块保留 | ✅ 每块保留 section 标题 |

---

## 6. 实际部署注意事项

| 问题 | 说明 |
|------|------|
| **Tokenizer 替换** | 上线前将 `cl100k_base` 换成目标嵌入模型的 tokenizer，参数小幅重校 |
| **abstract_source 字段** | 9.5% 摘要缺失，已用 body 首段填充，chunk_type 仍为 `"abstract"`，可通过 `abstract_source="body_fallback"` 区分 |
| **无 body 文档** | 4.7% 无正文，跳过策略 C，仅保留策略 A/B 的摘要 chunk |
| **correction/editorial 过滤** | 建议入库前过滤，避免噪声 chunk 影响检索质量 |
| **chunk_total > 1 的摘要** | 检索命中时，将同一 `pmc_id` 的多个 chunk 合并去重后再传入 LLM |
