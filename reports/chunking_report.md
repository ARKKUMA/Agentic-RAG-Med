# PMC oa_comm 文档分块（Chunking）流程报告

> **数据集**：PMC oa_comm（CC-BY 商用授权），约 527 万篇 JATS XML 全文  
> **Tokenizer**：`tiktoken cl100k_base`（DeepSeek V3/V4、Qwen3、GLM-4/5 BPE 近似值）  
> **嵌入模型上限**：512 tokens  
> **分块参数**：chunk_size = 400 tokens，chunk_overlap = 80 tokens  
> **验证规模**：benchmark 集 5,000 篇，生成 5,205 个 chunk

---

## 一、背景与目标

RAG（检索增强生成）系统的检索质量高度依赖文本块的粒度设计。本流程针对 PMC 医学文献数据集的以下特性进行定制：

| 数据集特点 | 对分块的影响 |
|-----------|-------------|
| 摘要 token 均值 **332.5**，中位数 **323** | 绝大多数摘要无需分块即可入库 |
| 摘要 p95 = **560 tokens**，超出 512 限制 | 约 8.4% 的摘要需要切割 |
| 正文中位数 **12,170 tokens**，远超上限 | 正文须按语义章节二次切割 |
| 医学摘要信息密度极高（每句含 3–5 个术语）| chunk 不宜过小，建议 ≥ 200 tokens |
| 非结构化摘要占比 **95.4%** | 须依赖句子边界（`. `）而非章节标签切割 |

**目标**：在保证每个 chunk ≤ 512 tokens 的前提下，最大化上下文完整性，同时携带充分的元数据支持检索过滤与溯源。

---

## 二、输入数据与前置清洗

### 2.1 数据来源

```
输入清单  : oa_comm_filelist_merged.csv  （5,284,651 条记录）
XML 文件  : pmc_xml_extracted/            （2,518,598 个已解压 XML）
```

### 2.2 前置过滤

在分块前按以下规则清洗原始数据：

| 过滤步骤 | 规则 | 数量（5,000 篇示例） |
|---------|------|-------------------|
| 撤稿过滤 | `Retracted == "yes"` → 丢弃 | 移除 12,758 / 527 万（全量） |
| 文章类型过滤 | 丢弃 correction / retraction / editorial / letter / book-review / news / reply 等 | 移除 229 篇（4.6%） |
| 摘要缺失处理 | 有正文 → 取 body 前 500 字作代理摘要（`body_fallback`） | 396 篇（7.6%） |
| 无摘要无正文 | 直接丢弃 | — |

**保留的文章类型**：`research-article`、`review-article`、`systematic-review`、`case-report`、`brief-report`、`methods-article`、`data-paper`、`rapid-communication`、`protocol`

### 2.3 唯一文档 ID

| 字段 | 缺失率 | 角色 |
|------|--------|------|
| `pmc_id` | **0.00%** | 主键，直接作为 `doc_id` |
| `pmid`   | ~20%   | 溯源链接优先项（无效值 "0" 已过滤） |
| `doi`    | ~5.2%  | 溯源链接次选 |

> pmid 无效值（"0"）已在 parse 阶段过滤，改用 doi / pmc_id 三级兜底构建 `source_url`。

---

## 三、分块策略设计

### 3.1 决策树

基于摘要实测 token 分布（7,248 篇抽样）：

```
输入文档
│
├─ article_type ∈ DROP_TYPES → 丢弃
│
└─ 保留文档
     │
     ├─ abstract 缺失 / < 50 chars
     │    ├─ body 存在 → abstract = body[:500]（body_fallback）
     │    └─ body 也缺失 → 丢弃
     │
     └─ 合法摘要
          │
          full_text = title + "\n\n" + abstract
          │
          ├─ token_count ≤ 512 → 【策略 A】整体不分割
          └─ token_count > 512 → 【策略 B】RecursiveCharacterTextSplitter
```

### 3.2 策略 A — 整体不分割（91.3% 文献）

**触发条件**：`title + "\n\n" + abstract` 的 token 数 ≤ 512

| 项目 | 说明 |
|------|------|
| chunk 内容 | `title + "\n\n" + abstract` 完整入库 |
| chunk_id | 直接使用 `pmc_id`（无后缀） |
| 每篇产出 | **1 个 chunk** |
| 优势 | 上下文完整，检索精度最高，实现最简洁 |

```
chunk_id     : "PMC12345678"          ← 直接等于 doc_id
chunk_index  : 0
total_chunks : 1
```

### 3.3 策略 B — 重叠滑动窗口（8.7% 文献）

**触发条件**：`title + "\n\n" + abstract` 的 token 数 > 512

使用 `langchain_text_splitters.RecursiveCharacterTextSplitter` 对摘要正文进行切割，分割完成后**每个子块统一拼接原文标题作为前缀**。

#### 分割器参数

| 参数 | 值 | 推导说明 |
|------|----|---------|
| `chunk_size` | **400 tokens** | 512 上限 − title 前缀（约 60 tokens）− 缓冲余量 |
| `chunk_overlap` | **80 tokens** | ≈ chunk_size × 20%，防止关键语句在边界割裂 |
| `length_function` | `self._count_tokens` | **必须使用 tokenizer 计数**，非 `len()` 字符数 |
| `separators` | `["\n\n", "\n", ". ", " "]` | 优先按段落 → 换行 → 句子 → 单词断开，避免截断句中 |
| `is_separator_regex` | `False` | 分隔符为字面量，非正则 |

#### 分割器初始化代码

```python
self.tokenizer = tiktoken.get_encoding("cl100k_base")

self.splitter = RecursiveCharacterTextSplitter(
    chunk_size        = 400,
    chunk_overlap     = 80,
    length_function   = self._count_tokens,   # token 级计数
    separators        = ["\n\n", "\n", ". ", " "],
    is_separator_regex= False,
)
```

#### 分割流程

```
abstract（原始摘要文本）
    │
    ▼  RecursiveCharacterTextSplitter.split_text(abstract)
    │
    ├─ 尝试按 "\n\n" 切割
    ├─ 子块仍过长 → 降级按 "\n" 切割
    ├─ 子块仍过长 → 降级按 ". " 切割（句子边界）
    └─ 子块仍过长 → 降级按 " " 切割（单词边界）
    │
    ▼  合并小碎片（merge）→ 每块 ≤ 400 tokens，相邻块共享 80 tokens 重叠
    │
    ▼  为每块拼接标题前缀
    │
    chunk_0 : title + "\n\n" + abstract_part_0
    chunk_1 : title + "\n\n" + abstract_part_1  ← 含 80 tokens 与 chunk_0 重叠
    chunk_2 : title + "\n\n" + abstract_part_2
```

#### Chunk ID 命名规则

```
chunk_id = f"{pmc_id}_chunk_{index:04d}"

例：
  PMC467093_chunk_0000   ← 第 1 块
  PMC467093_chunk_0001   ← 第 2 块（与第 1 块有 80 tokens 重叠）
  PMC467093_chunk_0002   ← 第 3 块
```

---

## 四、实现架构

### 4.1 模块结构

```
pmc_document_splitter.py
│
├─ 第 1 节  环境准备与数据加载
│    ├─ setup_logging()         控制台 + 文件双输出日志
│    ├─ parse_xml()             JATS XML 解析（IDs / 标题 / 期刊 / 年份 / 摘要）
│    └─ PMCChunkingPipeline
│         ├─ load_manifest()    CSV 加载、列名标准化、撤稿过滤
│         └─ build_file_index() pmc_id → xml_path 映射（带 JSON 缓存）
│
├─ 第 2 节  文本分割策略
│    └─ PMCDocumentSplitter
│         ├─ __init__()         加载 tokenizer + 初始化 RecursiveCharacterTextSplitter
│         ├─ _count_tokens()    tiktoken token 计数
│         ├─ _no_split()        策略 A：整体不分割
│         ├─ _split_smart()     策略 B：RecursiveCharacterTextSplitter + 标题前缀
│         └─ chunk_document()   决策派发（≤512 → A，>512 → B）
│
├─ 第 3 节  分割文档，保存结果
│    └─ PMCChunkingPipeline.run() + .save()
│         批量解析 → 过滤 → 分块 → 附元数据 → Parquet + 统计 JSON
│
├─ 第 4 节  预览结果
│    └─ preview_chunks()        格式化打印前 N 个 chunk
│
└─ 第 5 节  质量验证
     └─ validate_quality()
          ├─ Token 分布统计（mean/median/p95/p99/max）
          ├─ 超上限块检测（token_count > 512）
          ├─ 抽样质量检查（标题率 / 截断率 / 空文本）
          ├─ 多块文档 overlap 检测
          └─ ASCII Token 分布直方图
```

### 4.2 处理流程

```
oa_comm_filelist_merged.csv
         │
         ▼ load_manifest()
    DataFrame（过滤撤稿）
         │
         ▼ build_file_index()（缓存 JSON）
    {pmc_id → xml_path} 映射
         │
    ┌────┴────────────────────────────────────────┐
    │  for each row in manifest (batch_size=2000) │
    │                                             │
    │  1. file_index.get(pmc_id) → xml_path      │
    │  2. parse_xml(xml_path) → dict             │
    │  3. article_type 过滤                       │
    │  4. pmid 补充（CSV 覆盖 XML 空值）          │
    │  5. chunk_document() → chunks + strategy   │
    │  6. 附加元数据字段                          │
    └────────────────────────────────────────────┘
         │
         ▼ pd.DataFrame(all_chunks)
    chunks_df（Int64 类型优化）
         │
    ┌────┴──────────────────────┐
    │  save()                  │
    │  ├─ chunks_<split>.parquet│
    │  └─ pipeline_stats.json  │
    └───────────────────────────┘
         │
    preview_chunks() → 终端预览
         │
    validate_quality() → quality_report.json
```

---

## 五、运行结果统计

> **数据来源**：benchmark 集，5,000 篇，`pmc_chunking_pipeline.py` 及 `pmc_document_splitter.py` 两次运行结果一致

### 5.1 文献处理概况

| 指标 | 数量 | 占比 |
|------|------|------|
| 原始文献（CSV） | 5,000 | 100% |
| 无 XML 文件 | 0 | 0.0% |
| 文章类型过滤 | 229 | 4.6% |
| XML 解析失败 | 0 | 0.0% |
| **成功解析** | **4,771** | **95.4%** |
| 生成 chunk 总数 | **5,205** | — |
| 平均 chunks/文献 | **1.091** | — |
| 处理耗时 | 22.2 s | ≈ 215 篇/s |

### 5.2 分块策略分布

| 策略 | 触发条件 | 文献数 | chunk 数 | 文献占比 |
|------|---------|-------|---------|---------|
| **A — 整体不分割** | token ≤ 512 | 4,354 | 4,354 | **91.3%** |
| **B — 滑动窗口** | token > 512 | 417 | 851 | **8.7%** |

策略 B 的 4,771 × 8.7% = 417 篇超长文献，平均分成 **2.04 块/篇**，最多 3 块。

```
chunks_per_doc 分布：
  1 块 : 4,354 篇  (91.3%)  ← 策略 A
  2 块 :   400 篇  ( 8.4%)  ← 策略 B
  3 块 :    17 篇  ( 0.4%)  ← 策略 B（极长摘要）
```

### 5.3 Abstract Source 分布

| 来源 | 数量 | 占比 | 说明 |
|------|------|------|------|
| `original` | 4,809 | 92.4% | 原始摘要 |
| `body_fallback` | 396 | 7.6% | 正文前 500 字代理摘要 |

### 5.4 Token 长度分布

| 区间 | chunk 数 | 占比 |
|------|---------|------|
| < 50 | 246 | 4.7% |
| 50 – 99 | 184 | 3.5% |
| 100 – 199 | 675 | 13.0% |
| 200 – 299 | 1,193 | 22.9% |
| 300 – 399 | 1,445 | 27.8% |
| **400 – 512** | **1,456** | **28.0%** |
| > 512 | **0** | **0.0% ✅** |

```
Token 分布（benchmark 5,205 chunks）

<50        ██
50-99      ██
100-199    ██████
200-299    ██████████
300-399    ████████████
400-512    ████████████
>512       （空）
```

### 5.5 Token 描述性统计

| 统计量 | 值 |
|-------|----|
| 均值（mean） | **301.4 tokens** |
| 中位数（median） | **319.0 tokens** |
| 标准差（std） | 127.1 tokens |
| P25 | 219 tokens |
| P75 | 411 tokens |
| P90 | 447 tokens |
| P95 | 475 tokens |
| P99 | 505 tokens |
| 最大值（max） | **512 tokens** ✅ |
| 最小值（min） | 19 tokens |
| **超 512 token 块数** | **0（0.00%）** ✅ |

> 策略 B chunk 的均值（338.4）高于策略 A（294.2），说明滑动窗口将长摘要有效压缩至合理区间。

---

## 六、质量验证

### 6.1 抽样检查结果（n = 300）

| 检查项 | 结果 | 判定 |
|-------|------|------|
| 超模型上限（> 512 tokens） | 0 块，0.00% | ✅ 通过 |
| 含原文标题 | 300/300，**100.0%** | ✅ 通过 |
| 空文本 | 0 块 | ✅ 通过 |
| 疑似截断（末尾不完整） | 1 块，**0.3%** | ✅ 可接受 |

### 6.2 多块文档专项检查

**样本文档**：PMC467093（共 3 块，来自一篇超长血管外科摘要）

| 块对 | 重叠检测 | chunk_a 末尾 | chunk_b 开头 |
|------|---------|-------------|-------------|
| chunk 0 → 1 | ✅ 检测到 | `…secondary patency rates were 85.71%…` | `Retrospective evaluation of three…` |
| chunk 1 → 2 | ✅ 检测到 | `…primary, or secondary graft patency rates…` | `Retrospective evaluation of three…` |

> 每个 chunk 以原文标题作为前缀，所以 `head_of_b` 均为标题行——重叠内容紧随标题之后，独立检索时语义完整。

### 6.3 分割边界样例

以下展示策略 B 对一篇 543-token 摘要的分割效果：

```
原文摘要（543 tokens）
─────────────────────────────────────────────
Background: Expanded polytetrafluoroethylene (ePTFE) grafts are
commonly used as vascular conduits... [方法描述] ... The primary
patency rates were 57.14%, 48.57%; and the secondary patency rates
were 85.71%, 80.00%... [结果描述] ... No significant difference in
primary, or secondary graft patency rates were observed among the
three groups (P > 0.05).

─────────────────────────────────────────────
⬇ RecursiveCharacterTextSplitter（按 ". " 句子边界切割）
─────────────────────────────────────────────

chunk_0000（~400 tokens）:
  Title: Retrospective evaluation of three types of expanded...
  [Background + Methods + 前半段 Results]
  ...48.57%; and the secondary patency rates  ←── 末尾
  ←── 80 tokens 重叠区域 ────────────────────────────────────────────
chunk_0001（~400 tokens）:
  Title: Retrospective evaluation of three types of expanded...
  [中段 Results]  ←── 头部（含与 chunk_0000 重叠的 80 tokens）
  ...No significant difference...  ←── 末尾
  ←── 80 tokens 重叠区域 ────────────────────────────────────────────
chunk_0002（~80 tokens）:
  Title: Retrospective evaluation of three types of expanded...
  [...primary, or secondary graft patency rates were observed...]
```

---

## 七、Chunk 数据结构

每个 chunk 包含以下字段，存储于 Parquet 文件：

### 7.1 核心内容字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `chunk_id` | str | 全局唯一 chunk 标识 | `"PMC467093_chunk_0001"` |
| `text` | str | 实际文本（含标题前缀） | `"Title\n\nabstract part..."` |
| `doc_id` | str | 所属文献 ID（= pmc_id） | `"PMC467093"` |
| `chunk_index` | Int64 | 块在文献内的序号（0-based） | `1` |
| `total_chunks` | Int64 | 该文献共切割的块数 | `3` |
| `source_title` | str | 原文标题（便于溯源） | `"Retrospective evaluation..."` |
| `token_count` | Int64 | 本块 token 数 | `411` |

### 7.2 元数据字段（检索过滤器）

| 字段 | 类型 | 用途 | 示例 |
|------|------|------|------|
| `pmc_id` | str | 去重 / 溯源主键 | `"PMC467093"` |
| `pmid` | str / null | 溯源链接优先项 | `"16026548"` |
| `doi` | str / null | 溯源次选 | `"10.1016/..."` |
| `journal` | str | 期刊过滤 | `"Journal of Vascular Surgery"` |
| `pub_year` | Int64 | 时间范围过滤 | `2005` |
| `article_type` | str | 文章类型 | `"research-article"` |
| `abstract_source` | str | 摘要来源标记 | `"original"` / `"body_fallback"` |
| `source_url` | str | 三级兜底溯源链接 | `"https://pubmed.ncbi.nlm.nih.gov/..."` |
| `split_strategy` | str | 所用分块策略 | `"A_no_split"` / `"B_sliding_window"` |

### 7.3 检索过滤示例

```python
# 过滤 2021 年后 Nature Communications 的研究型文章
df.query(
    "pub_year >= 2021 and "
    "journal == 'Nature Communications' and "
    "article_type == 'research-article'"
)

# 多块文档重新拼合（检索命中后传给 LLM 前合并）
doc_chunks = (
    df[df["doc_id"] == "PMC467093"]
    .sort_values("chunk_index")["text"]
    .tolist()
)
```

---

## 八、输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `chunks_<split>.parquet` | Parquet (snappy) | 文本块数据集，列见第七节 |
| `pipeline_stats_<split>.json` | JSON | 处理配置 + 统计（篇数、块数、策略分布、耗时） |
| `quality_report_<split>.json` | JSON | 质量验证报告（token 分布 / 标题率 / 截断率 / overlap） |
| `splitter_<split>.log` | 纯文本 | 控制台 + 文件双输出处理日志 |

---

## 九、运行方式

```bash
# 安装依赖
pip install tiktoken pandas pyarrow langchain-text-splitters

# 小批量测试（500 篇，约 10 秒）
python pmc_document_splitter.py --split test --limit 500

# Benchmark（5,000 篇，约 25 秒）
python pmc_document_splitter.py --split benchmark --limit 5000

# 全量处理（≈ 527 万篇，预计 5–6 小时）
python pmc_document_splitter.py --split full

# Windows 控制台需设置 UTF-8 编码
$env:PYTHONUTF8="1"; python pmc_document_splitter.py --split full
```

---

## 十、总结

| 维度 | 结论 |
|------|------|
| **策略覆盖率** | 91.3% 文献走策略 A（完整入库），8.7% 走策略 B（平均 2.04 块）|
| **模型兼容性** | 全部 5,205 块均在 512 token 上限内，0 块超限 ✅ |
| **语义完整性** | 每块携带完整标题前缀；策略 B 相邻块保留 80 token 重叠，防止关键语句断裂 |
| **分割边界** | 分隔符优先级 `"\n\n" → "\n" → ". " → " "`，95%+ 情况下在句子边界切割 |
| **截断率** | 抽样 300 块中仅 1 块（0.3%）疑似截断，在可接受范围内 |
| **处理速度** | ≈ 215 篇/s（单进程），全量 527 万篇预计 5–6 小时 |
| **元数据完整** | 16 个字段覆盖内容、溯源、过滤三类需求 |

### 后续优化方向

1. **正文分块（策略 C）**：按 JATS `<sec>` 章节切割正文，每章节超 512 tokens 再滑动窗口二次切割，预计每篇新增 4–8 块
2. **多进程加速**：正文解析 CPU 密集，可用 `multiprocessing.Pool` 并行处理，速度提升 4–8×
3. **增量更新**：利用 `last_updated` 字段，只处理新增或变更的 XML，避免全量重跑
4. **Tokenizer 替换**：上线前替换为目标嵌入模型的 tokenizer（如 `bge-m3` 的 sentencepiece），`chunk_size` / `chunk_overlap` 小幅重校
