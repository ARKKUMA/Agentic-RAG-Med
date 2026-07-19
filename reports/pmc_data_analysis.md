# PMC oa_comm XML 数据结构分析报告

> 数据集：175 万篇 JATS XML（oa_comm，商用 CC-BY 许可）  
> 分析样本：2000 篇随机抽样 + 500 篇字段粒度验证  
> 格式：JATS（Journal Archiving and Interchange Tag Suite）

---

## 1. 字段完整性检查

### 1.1 各字段缺失率

| 字段 | XML 路径 | 缺失率 | 状态 |
|------|----------|--------|------|
| `pmc_id` | `<article-id pub-id-type="pmc">` | 0.00% | ✅ 可靠锚点 |
| `title` | `<article-title>` | 0.00% | ✅ 可靠锚点 |
| `journal` | `<journal-title>` | 0.00% | ✅ 可靠锚点 |
| `article_type` | `<article article-type="…">` | 0.00% | ✅ 可靠锚点 |
| `doi` | `<article-id pub-id-type="doi">` | 5.20% | ⚠️ 需兜底 |
| `pub_date` | `<pub-date>` | 5.95% | ⚠️ 需降级 |
| `abstract` | `<abstract>` | **9.50%** | ❌ 超出阈值，需清洗 |
| `pmid` | `<article-id pub-id-type="pmid">` | **19.90%** | ❌ 需三级兜底 |
| `keywords` | `<kwd-group>` | **39.00%** | ❌ 不可依赖 |

---

### 1.2 Abstract 清洗策略（缺失率 9.5% >> 1% 阈值）

**缺失原因分析**

Abstract 缺失并非数据质量差，而是文章类型决定的：

```
correction(31) + editorial(30) + letter(24) + book-review(10)
+ article-commentary(4) + news(4) + reply/retraction... ≈ 103/2000 = 5.2%
（此类文章本身无摘要，属正常现象）

剩余 ~4.3% 属于 research-article 本身缺失 abstract
```

**清洗策略（分两阶段）**

```
阶段 1 — 按 article_type 前置过滤
  丢弃：correction / retraction / editorial / letter /
        book-review / news / reply / article-commentary
  保留：research-article / review-article / systematic-review /
        case-report / brief-report / methods-article / data-paper

阶段 2 — 对保留文章的 abstract 缺失处理
  ├─ 有 abstract                → 直接使用（abstract_source = "original"）
  ├─ 无 abstract，有 body       → 取 body 前 500 字作代理摘要
  │                               （abstract_source = "body_fallback"）
  └─ 无 abstract，无 body       → 丢弃（无可检索内容）
```

> **不建议直接丢弃全部无摘要文章**：research-article 中缺 abstract 的文章正文通常完整，
> 直接丢弃会损失约 4.3% 的有效文献。

---

## 2. 基础质量分析

### 2.1 极短摘要

| 指标 | 数值 |
|------|------|
| 极短摘要（< 50 字符）数量 | 3 篇 / 2000 |
| 极短摘要比率 | 0.15% |

**处理策略**：与"无摘要"同等对待，用 body 首段替代，标记 `abstract_source = "body_fallback"`。

### 2.2 无正文（body 为空）

| 指标 | 数值 |
|------|------|
| 无正文文章数量 | 94 篇 / 2000 |
| 无正文比率 | **4.70%** |

**处理策略**：直接丢弃。无正文意味着无可分块、无可检索内容，保留无意义。

### 2.3 XML 解析错误

| 指标 | 数值 |
|------|------|
| 解析失败数量 | 0 篇 / 2000 |
| 解析失败率 | 0.00% |

文件结构完整，但处理代码仍需 `try-except`，以应对边缘情况。

### 2.4 HTML 实体 / 特殊字符编码

JATS XML 中广泛使用 XML 实体表示特殊字符（如希腊字母、数学符号）：

```xml
&#x003bc;  →  μ（微）
&#x003b1;  →  α（alpha）
&#x003ba;  →  κ（kappa）
&#x000d7;  →  ×（乘号）
```

**处理策略**：使用 Python `xml.etree.ElementTree` 解析时自动还原，**无需手动处理**。
若后续使用正则提取文本，需先调用 `html.unescape()` 或完整解析 XML 树。

---

## 3. 关键字段分析

### 3.1 title

- **完整率：100%**，无需任何清洗。
- 部分 title 包含 XML 子标签（如 `<italic>`、`<sub>`），提取时应使用 `itertext()` 拼接全部文本。

```python
title = "".join(root.find(".//article-title").itertext()).strip()
```

### 3.2 journal — 可作为元数据过滤器，需标准化

**完整率：100%**，字段本身可靠。

**主要问题：同一期刊存在多种拼写变体**

| 变体名称 | 样本中数量 | 实际期刊 |
|----------|-----------|---------|
| `"PLoS ONE"` | 184 | 同一期刊 |
| `"PLOS One"` | 13 | 同一期刊 |

**oa_comm 数据集中几乎没有《Nature》主刊文章**

样本 Top 20 中仅出现 `"Nature Communications"`（19 篇），无 `"Nature"` 主刊。
原因：《Nature》主刊极少采用 CC-BY 商用许可；oa_comm 数据主要来自
PLoS、Frontiers、BMC、MDPI 等全 OA 出版商。

**结论：`journal` 字段可作为过滤器**，但"检索《Nature》文献"在本数据集命中极少，
更实际的场景为过滤 `"Nature Communications"`、`"PLoS ONE"`、`"BMC …"` 等期刊。

**入库时做标准化：**

```python
JOURNAL_ALIASES = {
    "PLOS One":  "PLoS ONE",
    "PLOS ONE":  "PLoS ONE",
    # 可扩展
}
journal = JOURNAL_ALIASES.get(raw_journal, raw_journal)
```

### 3.3 pub_date — 可作为时间过滤器，提取 pub_year 字段

**500 篇实测缺失率：**

| 字段 | 缺失率 |
|------|--------|
| `pub_date`（完整日期） | 0.0% |
| `pub_year`（仅年份） | **0.0%** |

`pub_year` 在所有样本中均存在，比完整 `pub_date` 更可靠（有些文章只有年份，无月/日）。

**数据集时间跨度提示：**

```
最早：1840 年（历史期刊数字化归档）
最新：2026 年
"近5年"（2021–2026）在样本中约占 34%
```

**结论：** 将 `pub_year` 单独提取为 `int` 字段，完全支持时间范围过滤。

```python
# 提取
pub_year = int(pub_date_str[:4])  # 如 "2021-05-20" → 2021

# 查询示例："近5年 Nature Communications 文献"
pub_year >= 2021 AND journal == "Nature Communications"
```

### 3.4 pmid — 可作为溯源链接，需三级兜底

**500 篇实测：**

| 指标 | 数值 |
|------|------|
| pmid 存在 | 367 篇（73.4%） |
| pmid 缺失 | 133 篇（**26.6%**） |

pmid 缺失常见于新发表文章（PMC 先于 PubMed 收录）或部分小型期刊。

**示例验证（PMC3089640）：**

```
pmid   = 21573075   → https://pubmed.ncbi.nlm.nih.gov/21573075/
doi    = 10.1371/journal.pone.0019771
pmc_id = PMC3089640 → https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3089640/
```

**结论：** `pmid` 可作为溯源链接，但约 1/4 文章无 pmid，**必须配合三级降级链**
才能保证 100% 有可点击链接：

```python
def get_source_url(doc: dict) -> str:
    if doc.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/"
    if doc.get("doi"):
        return f"https://doi.org/{doc['doi']}"
    # pmc_id 缺失率 0%，永远有兜底
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{doc['pmc_id']}/"
```

---

## 汇总：入库字段清单

```python
{
    # 必填（缺失率 0%）
    "pmc_id":          "PMC3089640",
    "title":           "MITOSTATIN in Prostate Cancer",
    "journal":         "PLOS One",               # 标准化后存储
    "article_type":    "research-article",
    "pub_year":        2011,                      # int，用于时间范围过滤

    # 可空（有兜底逻辑）
    "pmid":            "21573075",
    "doi":             "10.1371/journal.pone.0019771",
    "pub_date":        "2011-05-18",
    "keywords":        ["MITOSTATIN", "prostate"],

    # RAG 核心内容
    "abstract":        "…",
    "abstract_source": "original",               # "original" | "body_fallback"
    "body_text":       "…",                       # 用于分块检索

    # 运行时计算
    "source_url":      "https://pubmed.ncbi.nlm.nih.gov/21573075/",
}
```
