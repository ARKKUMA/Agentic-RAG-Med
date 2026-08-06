# PMC 医学文献 RAG 系统

基于 PubMed Central（PMC）oa_comm 开放获取子集构建的端到端医学文献检索增强生成（RAG）系统。覆盖从原始数据获取、文档切分、向量化索引，到多路检索、重排序、提示词工程与本地 LLM 生成的完整链路。

> **本仓库只包含源码。** `data/`、`pipeline_output/`、`logs/` 均已 `.gitignore`（原始清单、切分产出、ChromaDB、embedding、BM25 索引等体积巨大，单机上就有上百 GB），需要按下文"快速开始"重新生成。

> 已封装为 FastAPI 服务（`api/`），接口调用说明见 [API.md](reports/API.md)，部署说明见 [DEPLOYMENT.md](reports/DEPLOYMENT.md)。

> 面向多跳检索/多工具场景的 Agentic 架构：设计见 [AGENT_ARCHITECTURE.md](reports/AGENT_ARCHITECTURE.md)，Week 1 实现（LangGraph 状态机 + 工具调度引擎 + 会话轨迹扩展）与测试/真实验证结果见 [AGENT_WEEK1_REPORT.md](reports/AGENT_WEEK1_REPORT.md)，代码在 [`agent/`](agent/)。四节点状态机（entry → tool_execution → answer_generation → termination）已可跑通端到端，但规划/反思/校验/整合四个组件仍是接口占位、图仍是纯串行无真实迭代循环，且尚未接入 API 层（`mode=agent` 路由未实现）。

## 目录结构

```
Rag-Med/
├── pmc_oa_bulk_downloader.py   # 数据获取：批量下载 PMC oa_comm XML
├── pmc_document_splitter.py    # 文档切分：摘要+正文分块（当前主力 pipeline）
├── pmc_vector_index.py         # 向量化 + ChromaDB 索引构建
├── watch_and_write_chroma.py   # 大批量嵌入时的异步 ChromaDB 写入器（与 GPU encode 解耦）
├── validate_index.py           # ChromaDB 索引质量验证脚本
├── build_bm25_sample.py        # 从全量语料中采样构建 BM25 索引
│
├── retrieval/                  # 检索模块（详见下文）
├── generation/                 # 生成模块（详见下文）
├── api/                        # FastAPI 服务层（问答/会话/统计/文档接口，见 reports/API.md）
├── agent/                      # Agentic RAG：LangGraph 状态机 + 工具调度引擎（Week 1 已实现，详见下文与 reports/AGENT_WEEK1_REPORT.md），未接入 api/
│
├── tests/                      # 单元/集成测试（unittest + TestClient）：test_api.py（API 层）、test_agent.py（Agent 状态机/调度引擎，29 项全 mock）
├── test_query_processor.py     # 查询理解模块测试
├── test_retrieval_pipeline.py  # 检索流水线测试（小规模 test_dir_mode 集合）
├── test_retrieval_pipeline_full.py  # 检索流水线测试（全量 pmc_full 集合，内存开销大）
├── test_embedding.py           # 嵌入模型测试
├── test_generation_pipeline.py # 生成流水线端到端测试
├── test_evaluation_cache_batch.py   # 答案评估/缓存/批量处理测试
├── test_hard_constraints.py    # 强约束/幻觉抑制对抗测试
├── test_agent_pipeline.py      # Agent 状态机端到端验证（真实检索+生成，含与单轮 RAG 的耗时对照）
│
├── utils/                      # 数据获取/分析辅助脚本
├── legacy/                     # 已被取代但保留存档的旧版本代码
├── notebooks/                  # 云端（AutoDL）GPU 训练/嵌入用 Jupyter Notebook
├── reports/                    # 数据分析类中文报告 + 接口/部署/架构设计文档（API.md / DEPLOYMENT.md / AGENT_ARCHITECTURE.md / AGENT_WEEK1_REPORT.md）
├── data/                       # 原始清单文件（filelist csv/txt）
├── pipeline_output/            # 流水线产出物：chunks、ChromaDB、BM25 索引、日志等
├── logs/                       # 各模块运行日志（JSONL + 可读文本）
│
├── openapi.json / postman_collection.json           # OpenAPI 快照 / Postman 测试集合
├── .env.example                # 环境变量模板（真实 .env 已 gitignore）
└── requirements.txt
```

## 数据流水线总览

```
下载 XML (pmc_oa_bulk_downloader.py)
    ↓
解压 + 清单整理 (utils/extract_pmc.py, utils/download_filelist_csv.py)
    ↓
文档切分 (pmc_document_splitter.py) → pipeline_output/chunks_<split>.parquet
    ↓
向量化 + 建索引 (pmc_vector_index.py) → ChromaDB (pipeline_output/chroma_db)
    ↓
检索 (retrieval/) —— 向量检索 + BM25 关键词检索 → 融合 → 交叉编码器重排序
    ↓
生成 (generation/) —— 上下文组装 → 证据评估 → 答案生成 → 批判性审查 → 最终答案
```

## 模块说明

### 数据获取与预处理

| 文件 | 作用 |
|---|---|
| `pmc_oa_bulk_downloader.py` | 批量下载 PMC oa_comm（商用许可）XML，支持断点续传、md5 校验 |
| `utils/download_filelist_csv.py` | 下载并合并 PMC 官方 filelist csv，生成 `data/oa_comm_filelist_merged.csv` |
| `utils/extract_pmc.py` | 批量解压下载得到的 tar.gz |
| `utils/analyze_pmc_xml.py` | XML 数据质量抽样分析 |
| `utils/analyze_domain_content.py` | 领域内容理解分析（文章类型、章节结构等） |
| `utils/analyze_token_length.py` | 摘要 token 长度统计（用于确定切分阈值） |

对应分析结论见 `reports/` 下的中文报告（数据集分析、领域分析、token 分析、切分策略）。

### 文档切分

`pmc_document_splitter.py` 是当前主力切分 pipeline，支持：
- 默认模式：仅切摘要（策略 A：≤512 token 不切；策略 B：超长用 RecursiveCharacterTextSplitter 滑窗切）
- 全文模式 `--fulltext`：额外对正文按 JATS section 做段落合并（策略 C）+ 二级递归切割（策略 D）

`legacy/pmc_chunking_pipeline.py` 是被取代的旧版本（仅支持摘要切分），保留作历史存档，当前代码中已无任何模块引用它。

Chunk 输出 schema（parquet）包含：`chunk_id / text / doc_id / chunk_type / imrad_type / section_path / split_strategy / pmc_id / pmid / doi / journal / pub_year / token_count` 等字段，详见脚本内文档字符串。

### 向量化索引

`pmc_vector_index.py`：
- `BGEEmbedder`：封装 `BAAI/bge-base-en-v1.5`，文档侧无指令前缀，查询侧自动加检索指令前缀
- `PMCVectorIndex`：ChromaDB 持久化索引封装，支持单文件/批次目录两种构建模式（后者支持断点续传，用于百万级语料）
- 内置质量验证（数量核对、自相似性召回率、边界情况）

`watch_and_write_chroma.py` 是大规模嵌入时使用的异步写入器，与 GPU encode 解耦以提升吞吐；`validate_index.py` 是独立的索引质量验证脚本。

当前已建好两个 ChromaDB 集合（`pipeline_output/chroma_db`）：

| 集合名 | 规模 | 说明 |
|---|---|---|
| `test_dir_mode` | 1,854 chunks | 快速测试用小样本（恰好全部来自 2003-2004 年 PLoS Biology，主题覆盖有限） |
| `pmc_full` | 5,811,866 chunks | 全量 PMC oa_comm 语料，覆盖真实年份/期刊分布 |

**注意：** `pmc_full` 的 HNSW 索引加载进内存约需 18-21GB RSS，在 33GB 内存的机器上会挤占绝大部分可用内存，导致查询延迟从毫秒级升到分钟级。日常开发/联调请使用 `test_dir_mode`；需要看真实检索质量数据时才用 `pmc_full`（对应 `test_retrieval_pipeline_full.py`）。

### 检索模块 `retrieval/`

| 文件 | 作用 |
|---|---|
| `query_processor.py` | `MedicalQueryProcessor`：中英文查询清洗、医学缩写展开、同义词扩展、实体识别、时间/研究类型过滤条件提取 |
| `medical_vocab.py` | 缩写表、同义词表、实体识别正则、中英词汇映射（词典数据） |
| `umls_builder.py` | 从 UMLS REST API 拉取医学同义词，可选补充 `medical_vocab.py`（需 API key，输出到 `vocabulary/`，尚未运行） |
| `bm25_index.py` | `BM25Index`：基于 `rank_bm25` 的关键词检索索引，支持从 ChromaDB/DataFrame 构建 + pickle 持久化 |
| `multipath_retriever.py` | `MultiPathRetriever`：向量检索 + BM25 双路召回融合，支持 `rrf`/`weighted`/`simple` 三种融合策略 |
| `reranker.py` | `MedicalReranker`：`BAAI/bge-reranker-base` 交叉编码器重排序，相关性(0.6)+时效性(0.25，指数衰减)+权威性(0.15，期刊权重表)加权 |
| `pipeline.py` | `RetrievalPipeline`：串联以上全部组件的统一入口 `.retrieve(query, top_k, fusion_strategy)` |

### 生成模块 `generation/`

| 文件 | 作用 |
|---|---|
| `context_assembler.py` | `ContextAssembler`：检索结果 → LLM 上下文。Jaccard 去重、相关性排序+来源多样性控制、token 预算内按句子边界截断 |
| `prompt_templates.py` | `MEDICAL_PROMPT_STAGES`：四阶段提示词模板——证据评估器 / 答案生成器 / 批判性审查器 / 最终组装器 |
| `llm_generator.py` | `LLMGenerator`：本地 Ollama `/api/generate` 封装，支持 JSON 模式约束 + 容错解析（修复多余逗号、缺失括号等） |
| `pipeline.py` | `MedicalGenerationPipeline`：检索 → 上下文组装 → 证据评估 → 答案草稿 → 批判性审查 → 按需重新组装 → 后处理（引用列表+免责声明） |

引用编号使用 `ContextAssembler` 分配的 `_citation_id`（存于每个 chunk 的 metadata），而非列表位置——因为证据评估阶段会从候选列表中剔除不相关来源，若按位置重新编号会与正文中模型引用的 `[来源 N]` 错位。

### Agent 编排模块 `agent/`

面向多跳检索/多工具场景的编排层，Week 1 已落地一条可运行的 LangGraph 状态机，架构设计与实现细节见 [AGENT_ARCHITECTURE.md](reports/AGENT_ARCHITECTURE.md) / [AGENT_WEEK1_REPORT.md](reports/AGENT_WEEK1_REPORT.md)。

| 文件 | 作用 |
|---|---|
| `state.py` | `AgentState`（`TypedDict`）：查询/会话信息、工具调用历史与检索结果（累加字段用 `Annotated[list, operator.add]`）、最终答案、迭代/超时控制字段；`should_terminate()` 判断是否超时或达最大迭代轮次 |
| `nodes.py` | 四个节点：`entry`（清洗查询/语言检测/读取会话历史）→ `tool_execution`（调工具、按 `chunk_id` 去重）→ `answer_generation`（复用 `ContextAssembler` + `answer_generator` 提示词阶段）→ `termination`（整理来源、判定 `DONE`/`FAILED`） |
| `graph.py` | 用 LangGraph 把四节点串成图；`tool_execution → answer_generation` 用条件边，`path_map` 预留"补充检索""重新规划"两条出边（当前路由函数恒定返回同一目的地）；`recursion_limit` 兜底防意外循环 |
| `tool_registry.py` | 工具注册表：`register`/`unregister`/`get`/`list_tools`/`describe_all`，支持导出真正可用的 `langchain_core.tools.StructuredTool` |
| `tool_dispatcher.py` | `ToolDispatcherEngine`：按 pydantic `param_schema` 从 `AgentState` 自动填参 → 校验 → 调用，区分可重试（网络超时等，指数退避最多 3 次）与不可重试异常（参数错误等，不重试） |
| `retrieval_tool.py` | 把 `RetrievalPipeline.retrieve()` 注册为第一个真实工具（`ToolDispatcherEngine` 之外的所有工具都照此模式接入，不需要改动调度引擎/图本身） |
| `interfaces.py` | `TaskPlanner`/`RetrievalReflector`/`AnswerValidator`/`ResultIntegrator` 等六个组件的 `Protocol` 接口，目前仍是占位，未接入图 |

`api/session.py` 的 `ConversationTurn` 新增可选字段 `agent_trace`（默认 `None`，对现有 RAG 模式零改动），`SessionManager.get_agent_trace()` 支持按会话 ID 取全部轨迹或按步骤名筛选。

**明确未完成**：规划/反思/校验/整合四个组件仍是接口占位；图是纯串行，两条预留分支边无判断逻辑驱动，不构成真正的迭代循环；API 层 `mode=agent` 路由未实现；工具结果缓存层未实现。端到端验证（`test_agent_pipeline.py`，5 条查询）确认 Agent 执行成功率 5/5，节点级耗时稳定，但"Agent 相对 RAG 总耗时开销"目前样本量太小、噪声大，不具备统计意义，详见 [AGENT_WEEK1_REPORT.md](reports/AGENT_WEEK1_REPORT.md) §5。

## 快速开始

### 环境准备

```powershell
# 1. 安装 PyTorch（CUDA 12.8）
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 2. 安装其余依赖
pip install -r requirements.txt

# 3. 安装并启动 Ollama（本地 LLM 服务，生成模块依赖）
# https://ollama.com 下载安装后：
ollama pull qwen2.5:7b-instruct
```

### 运行检索测试（小规模，秒级）

```powershell
$env:PYTHONUTF8="1"; python test_retrieval_pipeline.py
```

### 运行完整生成流水线测试

```powershell
$env:PYTHONUTF8="1"; python test_generation_pipeline.py
```

会依次打印每条测试查询的检索结果、各阶段耗时、最终答案与引用来源，并写入：
- `logs/generation_pipeline_run.log`（可读日志）
- `logs/generation_pipeline.jsonl`（结构化日志，每条查询一行）

### 运行 Agent 状态机测试

```powershell
# 单元测试（全 mock，秒级）
$env:PYTHONUTF8="1"; python -m unittest tests.test_agent -v

# 端到端验证（真实检索+生成，与单轮 RAG 流水线做耗时对照）
$env:PYTHONUTF8="1"; python test_agent_pipeline.py
```

### 从零构建切分 + 索引（如需重新处理数据）

```powershell
# 切分（全文模式）
$env:PYTHONUTF8="1"; python pmc_document_splitter.py --split full --fulltext

# 构建向量索引
python pmc_vector_index.py --input-dir pipeline_output/batches_full --collection pmc_full --resume
```

## 已知限制

- **`pmc_full` 内存开销大**：见上文"向量化索引"一节，非必要不要在开发机上常驻查询全量集合。
- **磁盘空间紧张**：`pipeline_output/batches_full.zip`（约 34.6GB）是 `batches_full/` 目录解压前的归档，内容完全重复且已用于构建索引，磁盘紧张时可删除以释放空间（未删除，需要时手动确认清理）。
- **BM25 无法覆盖全量语料**：`rank_bm25` 需要将分词后的语料全部驻留内存，570 万 chunk 级别不现实，因此 `pmc_full` 配套的 BM25 索引只用了跨语料随机采样的 23.6 万 chunk（`build_bm25_sample.py`），向量检索路径仍覆盖全量。
- **`retrieval/umls_builder.py` 尚未运行**：`retrieval/vocabulary/` 目前为空，如需真实 UMLS 同义词表需自行申请 API key 后运行。
