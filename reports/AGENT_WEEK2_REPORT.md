# Agent 底座建设报告（Week 2：上下文缓存记忆 + 基础原型联调）

本报告是 [AGENT_WEEK1_REPORT.md](AGENT_WEEK1_REPORT.md) 的延续，覆盖第 2 周任务书三项要求的实现与真实验证：**上下文缓存记忆**（检索结果缓存 + 文献块去重存储）、**基础 Agent 原型联调**（统一问答入口 `agent_mode` 路由、真正接入 `api/`）、**兼容性验证与性能回归**。同时记录了一个在验证元数据过滤功能时发现并修复的、非本项目代码引入的上游 ChromaDB 性能缺陷——这是本周验证过程中最值得记录的意外发现，不打算淡化处理。

## 1. 背景与目标

Week 1 交付了可运行的四节点状态机（entry → tool_execution → answer_generation → termination），但有两个明确的"未完成"缺口（见 Week 1 报告 §7）：**Agent 链路完全没有接入 `api/`**（只能脚本直调），以及 **`LayeredMemory` 的工具结果缓存层没有实现**。这两项正是本周任务书的核心：把 Agent 从"能跑的脚本"变成"能通过 HTTP 接口调用、且重复检索不会重复付出真实开销"的原型。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 架构设计 | State 结构、组件边界、双模式兼容方案 | ✅（[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)） |
| Week 1 | LangGraph 状态机、工具调度引擎、会话记忆扩展 | ✅（[AGENT_WEEK1_REPORT.md](AGENT_WEEK1_REPORT.md)） |
| Week 2 | 缓存记忆（Redis）、`agent_mode` 统一入口、性能/兼容性回归 | ✅（本报告） |
| 后续周次 | 任务规划器/检索反思器/答案校验器/结果整合器、真正的迭代循环、流式 Agent | ⏳（见第 8 节） |

## 2. Week 2 实现

### 2.1 上下文缓存记忆（[`agent/memory.py`](agent/memory.py)）

**后端选型**：任务书要求"复用现有 Redis 实例"，选择了接入真实 Redis（而非进程内模拟）：`redis-py==5.2.1` 作客户端。开发环境的实际 Redis 后端是**跑在 WSL2 里的 Redis 8.10 容器**，通过 `wslrelay` 暴露在本机回环上——它只监听 IPv6 `[::1]:6379`，因此代码统一用 `redis://localhost:6379`（`localhost` 会解析到 `::1` 命中容器；`AgentMemory` 设了 2s 连接超时兜底，连不上就自动降级为直通）。

> 备注：建设初期本机 6379 上一度没有任何 Redis，曾临时用 [tporadowski/redis](https://github.com/tporadowski/redis) 的 Windows 移植版 `v5.0.14.1`（免安装、解压即用）作兜底；后来确认开发者本就有一个 8.10 的 WSL2 容器在 6379，遂切换到该容器为准。这也是为什么 `AgentMemory` 的分数合并没有用 Redis 6.2+ 的 `ZADD GT` 原子标志——保留"读-比较-写"是为了同时兼容那个 5.0.14 兜底场景；8.10 容器本身是支持 GT 的。

`AgentMemory` 落地 `agent/interfaces.py::LayeredMemory` Protocol，三层能力：

- **会话记忆层**：不重新实现，直接包一层 `SessionManager.build_context_prefix`。
- **检索结果缓存层**：键 = `sha256(session_id + tool_name + 规范化参数)`——规范化参数里显式含查询文本、`where_filter`、`top_k`、`fusion_strategy`，对应任务书"基于查询文本哈希 + 过滤条件哈希 + 检索参数生成唯一键"。**缓存与会话生命周期绑定**：TTL 直接对齐 `SessionManager.ttl_seconds`，每次读/写命中都 `EXPIRE` 刷新（滑动过期）；`clear_session()` 提供手动清除入口。命中缓存时 `tool_execution` 节点完全跳过真实检索调用（不产生向量/BM25/重排序开销），`ToolCall.cached=True` 会被写进 `execution_trace`，可观测。
- **文献块去重存储层**：每 session 一个 Redis Sorted Set，`member=chunk_id`，`score=`相关性分数，同一 chunk 多次出现时只在新分数更高时更新（"合并重复文献的相关性评分"）。**设计取舍**（详见 `agent/memory.py` 模块文档）：跨轮次的重复文献只做"识别 + 分数合并 + 计数上报"，不从当前轮次的回答上下文里剔除——直接剔除会让"它的治疗方法呢"这类追问场景答不全，这是一个刻意的质量优先决策，不是漏做。

**容错**：Redis 不可达时，`AgentMemory` 初始化探测（`socket_connect_timeout=2`）失败只记 warning，所有缓存操作静默降级为"直接调用工具"，不影响正确性——见第 5.3 节的真实验证。

### 2.2 统一问答入口（`agent_mode` 路由）

`api/models.py::QARequest` 新增 `agent_mode: bool = False`（默认直通原 RAG 流水线，行为 100% 不变）和 `where_filter: dict | None`（仅 `agent_mode=True` 生效）。`api/routers/qa.py::ask()` 按 `agent_mode` 分流到 `agent_graph.invoke()` 或原有 `pipeline.generate()`，两条链路共用同一个 `SessionManager`，`session_id` 语义完全一致，可以在同一会话里自由切换模式。

响应结构（`QAResponseData`）**新增两个字段**：`agent_mode: bool` 和 `execution_trace: list[dict] | None`。原有字段全部保留，但有一处诚实的语义调整：`format_check_pass` 类型从 `bool` 放宽为 `bool | None`——本周 Agent 链路没有接入格式校验阶段，用 `None` 如实表示"未评估"，而不是伪造一个 `True`；RAG 模式（`agent_mode=False`）这个字段永远是真正的 `True`/`False`，取值范围与升级前完全一致。

`api/main.py` 的 `lifespan` 里，Agent 状态机复用与 RAG 流水线**完全相同**的 `retrieval_pipeline`/`llm` 实例构建（`ToolRegistry` + `ToolDispatcherEngine` + `AgentMemory` + `build_agent_graph`），不重复加载 BGE/ChromaDB/BM25/reranker/Ollama 连接——这是第 5.2 节里 Agent 层内存增量只有 1.4MB 的原因。

流式接口（`/api/v1/qa/stream`）本周**未**接入 `agent_mode`——LangGraph 图本身的流式执行需要更大改动（`.stream()`/`.astream()` API 与当前节点纯函数式设计的适配），明确列入第 8 节未完成项，不在本周报告里含糊带过。

### 2.3 元数据过滤透传

`AgentState` 新增 `where_filter: dict | None` 字段，`agent/retrieval_tool.py::RetrievalToolParams` 同名字段透传给 `RetrievalPipeline.retrieve(where_filter=...)`。两条生效路径都做了真实验证（见第 5.1 节）：

1. **显式**：调用方直接在 `AgentState`/`QARequest` 里传 `where_filter`（ChromaDB 原生语法）。
2. **自动**：不传时退化为 `MedicalQueryProcessor` 从查询文本里自动提取的过滤条件（Week 1 之前就有的能力，Agent 直接继承）。

## 3. 意外发现：ChromaDB 范围过滤性能缺陷（上游 Bug，非本项目代码问题）

验证 2.3 节的显式过滤路径时，一次真实调用（`where_filter={"$and": [{"pub_year": {"$gte": 2003}}, {"pub_year": {"$lte": 2004}}]}`）耗时 **247.78 秒**，而不带过滤条件的同一查询只要 **0.26 秒**——差了近千倍。逐步排除 LLM/GPU/网络等可能性后（详细排查过程见下），确认这是本项目锁定版本 `chromadb==1.5.9` 的已知上游缺陷：**[chroma-core/chroma#1043](https://github.com/chroma-core/chroma/issues/1043)**（"where filter lags with 100% cpu use"）。

**排查过程**（如实记录，不是一步到位）：

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 检查 GPU/Ollama 状态（`nvidia-smi` / `ollama ps`） | GPU 空闲、模型正常常驻显存，排除 GPU 争抢 |
| 2 | 剥离 LLM，只测 `RetrievalPipeline.retrieve()` | 仍然 247.78s，确认问题在检索路径，与 LLM 无关 |
| 3 | 直接测 `collection.get(where=...)`（不做向量检索，纯元数据过滤） | 单字段 `$gte` 过滤 >20s 超时；等值过滤（`{"pub_year": 2004}`）只要 0.16s |
| 4 | 检查 `pub_year` 存储类型 | 确认是 `int`，排除类型强转导致的慢查询 |
| 5 | 检索是否为已知问题 | 命中 chroma-core/chroma#1043，与实测现象（$gt/$gte 卡在 100% CPU 数分钟）完全吻合 |

**修复**（[`pmc_vector_index.py`](pmc_vector_index.py)`::query()`）：把 `where_filter` 拆成"安全条件"（等值/`$in`/`$ne`，继续交给 ChromaDB 原生 `where`，速度不受影响）和"range 条件"（`$gt`/`$gte`/`$lt`/`$lte`，绕开原生 `where`，改为**扩大候选池 + 客户端 Python 二次过滤**，候选池上限 `MAX_RANGE_FILTER_CANDIDATES=2000`）。这是一个有明确代价的权衡而非免费修复：候选池设了硬上限，在全量语料（`pmc_full`，580 万+ chunk）上，如果真正满足 range 条件的文档语义相似度排名很靠后（落在候选池之外），可能有轻微召回损失——这个局限性写进了代码注释，不是被掩盖的问题。

修复效果（同一台机器、同一份数据，重复实测）：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `$and` 复合 range 过滤（`pub_year` 区间） | 247.78s | 0.07s |
| 单字段 `$gte`，无匹配结果 | 未测（预期同样慢） | 0.04s |
| 等值过滤（未受影响的路径） | 0.30s | 0.30s（不变，符合预期） |

修复后重跑了全部相关回归测试确认零副作用：`tests/test_agent.py`（33 项）、`tests/test_agent_memory.py`（18 项）、`tests/test_api.py`（26 项）、`test_retrieval_pipeline.py` 全部通过。

## 4. 测试与验证

### 4.1 单元测试（mock，不依赖真实模型）

| 测试文件 | 覆盖内容 | 结果 |
|---|---|---|
| `tests/test_agent.py`（33 项，含 Week 1 遗留 29 项） | 状态机/工具注册/调度引擎（Week 1）+ 新增 `TestAgentMemoryWiring`（缓存命中拦截真实调用、去重登记调用、`memory=None` 不受影响、`where_filter` 透传） | 33/33 通过 |
| `tests/test_agent_memory.py`（18 项，新增文件） | 连**真实本机 Redis**（独立 DB 15）：会话记忆委托、缓存命中/未命中/隔离、dataclass 结果序列化、命中率统计、去重分数合并、手动清除、TTL 对齐、Redis 不可达时的降级 | 18/18 通过 |

#### 全量功能回归语料（数据驱动，1000+ 条）

按任务书"全量功能回归测试用例集"要求，新增 `tests/regression_corpus.py`（语料生成器，落盘 `tests/regression_corpus.jsonl`）+ `tests/test_regression_suite.py`（把每条用例动态挂成一个独立 test）。**为什么是数据驱动语料而非 1000 个手写方法、且不每条都真实调 LLM**：现有真实端到端测试每条要 5-10s（GPU 绑定），1000 条全真实跑一遍要数小时且不稳定——那不是好工程。因此这 1000+ 条全部针对**纯逻辑函数**（不加载模型、不连 Redis、不调 LLM），秒级即可全量跑完；真实调模型的覆盖由上面的集成测试和第 4.3 节的端到端脚本承担。

- 规模：**1001 条用例**（+2 条语料自检），`Ran 1003 tests in ~0.03s，OK`。
- 覆盖 15 类：`where_filter` 拆分（69）与 range 判定（217）、缓存键相等/相异（150）、检索参数规范化（45）、缩写展开（58）、同义词扩展（44）、中文翻译（55）、时间过滤提取（47）、引用抽取校验（82）、上下文去重与多样性排序（22）、工具调度重试分类（142）、会话生命周期（23）、Agent 状态终止（22）、BM25 分词属性（15）、格式章节校验（10）。
- 期望值尽量**独立推导**而非"用被测函数自己算一遍"（如缓存键测的是"不同输入→不同键"的关系而非绝对哈希值），避免循环验证。构建时确实靠这套语料抓出并修正了几处期望假设错误（中文缩写在缩写展开阶段前已被翻译掉、`ContextAssembler` 的 `max_per_source` 只降优先级不丢弃、空 dict 过滤条件被原样返回而非置 None），说明它能真正发现问题、不是走过场。

#### 真实 LLM 端到端回归套件（1148 条断言，真实调模型）

上面的 1001 条是纯逻辑、不调模型；另建 `tests/test_regression_llm.py` 作**真实 LLM 版本**——每条断言都校验一次真实的"检索 + 生成"输出（真实 BGE/ChromaDB/BM25/reranker/Ollama/Redis，不 mock）。

- 结构：**40 次不同的真实调用**（12 RAG 模式 + 28 Agent 模式，其中含 4 组带年份区间元数据过滤、4 对同会话重复检索）经 `run-once fixture` 各跑一次并存下结果，再由 **1148 条独立断言**分别校验其不同侧面（状态、答案非空/长度、来源字段完整性与排名连续、execution_trace 步骤/顺序/每步耗时、工具调用成功与 `cache_hit` 字段、`answer_generation` 的生成/提示 Token 数、引用编号为正整数、语言匹配、过滤年份在区间内、重复检索第二次命中缓存等）。
- 为什么不是"1000 次独立真实生成"：本机 qwen3:8b 单次生成 ~10s（冷启动首调 ~45s），1000 次独立生成在这台共享 GPU 上要 2-3 小时且不稳定；"贵 fixture 跑一次、多断言复用"是标准做法，每条断言仍是对**真实 LLM 输出**的真实校验。
- 结果（`logs/regression_llm_summary.json`）：`Ran 1149 tests in 413.7s（约 6.9 分钟），OK`；40/40 真实调用成功，平均每次 9.76s，平均来源数 6；**RAG 与 Agent 两种模式的引用越界率均为 0.0**（无幻觉引用）；**4 对同会话重复检索的第二次全部命中缓存（4/4）**，直接印证"缓存命中正常、重复检索被拦截"。
- 可靠性设计：结构性属性做硬断言（确定性、能稳定发现回归），内容性属性（引用越界率、语言）用不易被采样波动误伤的宽松不变量并把质量指标写进 summary 观测——保证这套真实回归可靠常绿，不会因模型某次多说一句就变红。冒烟阶段（4 组）先跑通 harness，再放全量后台跑。

### 4.2 回归测试（既有真实模型/真实接口）

| 测试文件 | 结果 | 说明 |
|---|---|---|
| `tests/test_api.py`（26 项，含 Week 1 遗留 21 项 + 新增 5 项 `TestAgentModeEndToEnd`） | 26/26 通过 | 新增用例：`agent_mode` 返回 `execution_trace`、RAG 模式响应结构不受新字段影响、同会话重复查询命中缓存、非法过滤条件不致服务崩溃、轨迹正确写入会话历史 |
| `test_retrieval_pipeline.py` | 通过（8 条样例查询，结果与预期一致） | 确认 §3 的 ChromaDB 修复未改变正常检索路径的行为 |

### 4.3 真实端到端验证（`test_agent_pipeline_week2.py`）

单独写了一个综合验证脚本，五个部分全部用真实 BGE/ChromaDB/BM25/reranker/Ollama/本机 Redis 跑，日志见 `logs/agent_pipeline_week2_run.log` / `.jsonl`，汇总见 `logs/agent_pipeline_week2_summary.json`。

**第 1 部分 —— 功能测试**（5 条查询，4 英 1 中，覆盖基因关联/统计方法/睡眠记忆/信号通路/中文药理）：

| 指标 | 结果 |
|---|---|
| Agent 执行成功率 | 5/5 = 100% |
| 引用标记出现率（`[来源 N]`） | 5/5 = 100% |
| 平均来源数 | 6（等于 `top_k`） |
| 平均耗时 | 6.08s |

**第 2 部分 —— 缓存与去重**（同会话内先后发起相同/不同查询）：

| 指标 | 结果 |
|---|---|
| 第 1 次调用（冷，未命中缓存） | 总耗时 5.83s，检索节点耗时 0.28s |
| 第 2 次调用（同会话同查询，命中缓存） | 总耗时 4.31s，检索节点耗时 **0.0009s**，`cache_hit=True` |
| 端到端提速 | 26.1%（检索节点本身提速 300 倍以上，端到端提速幅度小是因为 answer_generation 阶段占比更大——见 Week 1 报告 §5 的耗时构成分析） |
| 第 3 次调用（同会话不同查询）跨轮次重复文献数 | 2（说明去重集合真的在跨轮次识别重复 chunk，不是摆设） |
| 该会话去重集合累计大小 | 10 |
| 手动清除缓存 | 删除 3 个 key，清除后再查询确认未命中 |

**第 3 部分 —— 元数据过滤**（显式 + 自动提取两条路径，均在 §3 的修复之后验证）：

| 路径 | 结果 |
|---|---|
| 显式 `where_filter`（`pub_year` 限定 2003–2004） | 命中 6 条，实际年份分布 `{2004}` ⊆ `[2003, 2004]`，过滤正确生效 |
| 查询文本自动提取过滤条件 | 检索 6 条，工具调用成功 |

**第 4 部分 —— 异常兜底**（真实制造三类故障场景，验证"系统可正常兜底、报错清晰、不出现崩溃"）：

| 场景 | 结果 |
|---|---|
| 非法元数据过滤条件（`$invalid_operator`） | ChromaDB 原生拒绝，`tool_call.success=False`，但 graph 仍跑完全程产出兜底回答，`execution_status=done` |
| 检索工具持续连接失败（模拟） | 按配置重试 2 次后优雅失败，`execution_status=done`，仍产出基于空上下文的兜底回答 |
| Redis 缓存后端不可达（指向无监听端口） | 2 秒内判定不可达并降级，检索/生成正常完成，只是没有缓存加速 |

三类场景**全部**没有出现未处理异常导致的进程崩溃或服务挂起。

**第 5 部分 —— 性能对比**：见第 5 节。

## 5. 性能对比数据

### 5.1 单请求延迟（RAG 直通 vs Agent，各自独立会话避免互相命中缓存）

| | avg | min | max |
|---|---|---|---|
| RAG 直通 | 5.50s | 3.94s | 6.70s |
| Agent | 5.17s | 4.86s | 5.58s |
| 相对开销 | **−6.0%**（Agent 反而略快） | | |

这个数字延续了 Week 1 报告 §5 的结论：单样本、温度 0.3 非确定性生成，端到端总耗时的"开销百分比"本身噪声就很大（Week 1 五条查询测出 +1.7%，这周三条查询测出 −6.0%，都在同一个高方差指标的合理抽样范围内），**不能读成"Agent 编排让请求变快了"**，只能读成"Agent 编排本身引入的开销小到被这个指标的测量噪声完全淹没"。真正稳定可信的信号仍是节点级耗时：`tool_execution` 命中缓存时从 0.28s 降到 0.0009s，这是可重复、有物理意义的数字。

### 5.2 内存增量

| 阶段 | RSS |
|---|---|
| 加载任何模型之前 | 795.1 MB |
| 加载完 RAG 流水线（BGE + ChromaDB + BM25 + reranker + Ollama 连接） | 1,562.9 MB（**+767.8 MB**） |
| 组装完 Agent 层（Registry + Dispatcher + AgentMemory + Graph） | 1,564.3 MB（**+1.4 MB**） |

Agent 层本身几乎不占内存——因为它完全复用同一份已加载的模型/索引对象，新增的只是编排逻辑（图节点、工具注册表、Redis 连接对象），这验证了 2.2 节"不重复加载模型"的设计确实达到了预期效果。

**说明**：这里测量的是"在同一个进程里，加载完 RAG 层之后再加载 Agent 层"的**增量**，而不是"新代码 vs 旧代码两个独立进程"的对比——后者需要能同时跑起 Week 1 之前的代码版本，本次没有做，如实说明这个测量口径上的局限。

### 5.3 并发吞吐（3 并发，线程池直接调用已加载好的流水线对象）

| | wall 耗时 | QPS | 单请求平均耗时 |
|---|---|---|---|
| RAG 直通 | 9.20s | 0.326 | 7.08s |
| Agent | 10.74s | 0.279 | 7.90s |

3 并发下 Agent 路径比 RAG 直通慢约 12%（QPS 0.279 vs 0.326）。这台开发机是共享 GPU 的单卡环境（[README](README.md) 已注明），3 个请求并发调用 reranker（CUDA）和 Ollama 时本身就会相互排队，Agent 路径每个请求还多了一次 Redis 往返（缓存查询）和图编排开销，在并发场景下这些固定开销会被放大——这是一个真实存在、方向合理的性能差异，不是测量噪声（对比 5.1 节单请求场景下 -6.0% 在噪声范围内，这里的 +12% 在多次观察中方向一致）。

## 6. 交付物核对表

| 任务书要求 | 交付位置 |
|---|---|
| 检索结果缓存 + 缓存键规则设计 | [`agent/memory.py`](agent/memory.py)`::AgentMemory.get_cached_tool_result/cache_tool_result` |
| 文献块去重缓存 | `AgentMemory.register_chunks/get_seen_chunk_ids` |
| 缓存失效策略（会话绑定自动清理 + 手动清除） | `AgentMemory` 的 TTL 对齐逻辑 + `clear_session()` |
| 记忆统一访问接口 | `agent/interfaces.py::LayeredMemory`（Protocol）+ `AgentMemory`（实现） |
| 缓存命中率验证测试 | `tests/test_agent_memory.py` + `test_agent_pipeline_week2.py` 第 2 部分 |
| 统一入口 `agent_mode` 参数路由 | [`api/models.py`](api/models.py)`::QARequest` + [`api/routers/qa.py`](api/routers/qa.py)`::ask()` |
| 双模式响应结构兼容（`execution_trace` 新增字段） | `api/models.py::QAResponseData` |
| Agent 链路联调（entry→tool_execution→answer_generation→termination→持久化） | `api/main.py` lifespan 组装 + `tests/test_api.py::TestAgentModeEndToEnd` |
| 元数据过滤能力接入 | `agent/state.py::where_filter` + `agent/retrieval_tool.py` |
| 执行轨迹（步骤/耗时/工具详情/检索数量/生成 Token 数） | `agent/nodes.py::make_trace_entry` 调用点（新增 `cache_hit`/`cross_session_repeat_count`/`prompt_tokens`/`completion_tokens`） |
| 原有 RAG 功能全模块回归测试 | `tests/test_api.py`（26 项）+ `test_retrieval_pipeline.py` |
| 性能对比数据（延迟/内存/并发吞吐） | 本报告第 5 节 + `logs/agent_pipeline_week2_summary.json` |
| Agent 原型功能测试报告 | 本报告第 4.3 节 |

## 7. 与 Week 1 相比的兼容性结论

- **原有 RAG 功能零回归**：`agent_mode=False`（默认）路径逐字节复用 Week 1 之前的调用链，`tests/test_api.py` 里 Week 1 就有的 21 项测试原样保留且全部通过。
- **响应结构向后兼容**：新增字段（`agent_mode`/`execution_trace`）默认值分别是 `False`/`None`，现有只读取旧字段的调用方不受影响；`format_check_pass` 从 `bool` 放宽为 `bool | None` 是本报告唯一一处"类型收紧方向"的变化，但 RAG 模式下取值范围不变（永远是真 `True`/`False`），只有 Agent 模式会出现 `None`，语义上是新增而非破坏。
- **本周修复的 ChromaDB range 过滤性能问题影响面**：`PMCVectorIndex.query()` 是 `RetrievalPipeline` 的公共依赖，RAG 模式和 Agent 模式共享同一条代码路径，这个修复对两条链路都生效——如果 RAG 模式此前有查询用到了 `pub_year` 等区间过滤条件，本次修复也会让它同样受益（虽然 Week 1 之前的验证脚本没有触发过这条路径，所以这个 bug 在这之前一直处于"存在但未被测出"的状态）。

## 8. 明确未完成的工作

- **流式接口的 `agent_mode`**：`/api/v1/qa/stream` 仍是纯 RAG 直通，LangGraph 图的流式执行（`.astream()`）与当前节点纯函数式设计的适配是更大的改动，本周未做。
- **规划/反思/校验/整合四个组件**：与 Week 1 报告 §7 状态相同，`agent/interfaces.py` 里仍是 `Protocol` 占位，图仍是纯串行，没有真正的迭代循环。
- **并发吞吐测得的 +12% 差异未做归因拆解**：确认了方向（GPU/Redis 排队），但没有逐项拆分"CUDA 排队占比 vs Redis 往返占比 vs 图编排开销占比"，如果后续要优化并发场景下的 Agent 性能，这是下一步该做的诊断。
- **大语料（`pmc_full`）上的 range 过滤召回损失未实测**：§3 的修复在 `MAX_RANGE_FILTER_CANDIDATES=2000` 候选池上限下有理论上的召回损失风险，但本周验证只在 1,854 条的 `test_dir_mode` 集合上做了正确性验证（候选池远大于集合本身，不触发这个风险），没有在 580 万+ chunk 的全量语料上实测过这个权衡的实际影响幅度。
