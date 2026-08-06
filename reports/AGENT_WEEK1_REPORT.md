# Agent 底座建设报告（架构设计 + Week 1 实现）

本报告汇总 Agentic RAG 架构从设计到落地的完整过程：顶层架构设计（State 结构、执行闭环、组件边界、双模式兼容方案）与 Week 1 的真实实现（LangGraph 状态机、工具调度引擎、会话记忆扩展），以及全部测试/真实端到端验证结果。设计层面的完整细节见 [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)；本报告聚焦"做了什么、测没测过、结果如何"。

## 1. 背景与目标

现有系统（[`retrieval/`](retrieval/) + [`generation/`](generation/) + [`api/`](api/)）是一条工作良好、已上线验证过的**单轮 RAG 流水线**：检索 → 证据评估 → 答案生成 → 批判性审查 → 最终组装，全部在一次调用内线性走完，无法应对"检索一次发现证据不够，需要再查一轮"的多跳场景。本次工作的目标是在**不改动、不拖慢**现有单轮 RAG 能力的前提下，加一层可迭代的 Agent 编排。

范围分两个阶段：

| 阶段        | 内容                                                                                       | 状态                                                       |
| ----------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| 架构设计    | State 结构、执行闭环、状态流转规则、六个核心组件接口、双模式兼容方案                       | ✅ 已完成（[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)） |
| Week 1 实现 | LangGraph 状态机落地、工具调度引擎、会话记忆 agent_trace 扩展                              | ✅ 已完成（本报告）                                        |
| 后续周次    | 任务规划器/检索反思器/答案校验器/结果整合器实现、真正的迭代循环、API 层`mode=agent` 路由 | ⏳ 未开始（见第 6 节）                                     |

## 2. 架构设计阶段成果（回顾）

- **全局 State**：覆盖原始查询、会话信息、任务/子任务（预留）、工具调用历史、检索结果集、中间结论、反思记录（预留）、合规校验结果（预留）、最终答案、控制字段（迭代计数/超时/最大轮次）。
- **执行闭环**：规划 → 执行 → 反思 → 整合 → 校验，五个环节间的分支判断与兜底机制（超时、最大迭代轮次）在架构文档 §1.3-1.4 有完整状态转移表。
- **六个核心组件边界**：`TaskPlanner`/`ToolDispatcher`/`LayeredMemory`/`RetrievalReflector`/`AnswerValidator`/`ResultIntegrator`，每个都明确标注了要复用哪一块现有代码，不重新发明。
- **双模式兼容方案**：`mode: "rag"|"agent"` 字段（默认 `rag`，现状零改动）、响应结构新增 `execution_trace` 可选字段、全部复用现有日志/统计/健康检查/缓存基础设施。

这一阶段的交付物是**设计文档 + 类型契约**（`agent/interfaces.py` 六个 `Protocol`，`agent/state.py` 当时是纯 dataclass），刻意不含可执行逻辑。

## 3. Week 1 实现

### 3.1 LangGraph 状态机

**关键决策**：`AgentState` 从设计阶段的 dataclass 改成 `TypedDict`。原因是 LangGraph 节点函数的约定是"输入 State，返回一个部分更新字典"，框架自己按字段做合并；`tool_call_history`/`execution_trace`/`reflections` 用 `Annotated[list[X], operator.add]` 声明为累加字段，`retrieval_results` **刻意不累加**（它是"去重后的当前完整集合"，每次由节点整体重算覆盖，用累加会导致同一条 chunk 被反复计入）。这个合并行为在写业务代码之前先用最小示例单独验证过，确认可行才用到正式设计里。

四个节点（`agent/nodes.py`）串成一条链（`agent/graph.py`）：

```
entry → tool_execution → answer_generation → termination
```

- `entry`：清洗查询、检测语言、按 `session_id` 读取历史对话上下文
- `tool_execution`：通过工具调度引擎调用检索工具，按 `chunk_id` 去重合并结果
- `answer_generation`：复用上周的 `ContextAssembler` 和 `answer_generator` 提示词阶段，产出"初始答案"（不接证据评估/批判性审查/最终组装——那是预留给未来周次的能力）
- `termination`：整理 `sources`、判定最终状态（`DONE`/`FAILED`）

`tool_execution` 到 `answer_generation` 之间用**条件边**而非普通边连接，路由函数当前恒定返回同一个目的地，但 `path_map` 里已经声明好"补充检索""重新规划"两条预留出边——未来加反思迭代只需要改路由函数内部判断逻辑，图的拓扑结构不用重新布线。

兜底机制：LangGraph 的 `recursion_limit`（默认 25，四节点串行图正常只需 4~5 步）防止意外循环；`agent.state.should_terminate()`（超时/达最大迭代轮次判断）已实现并留在 `state.py` 里，供未来出现真正的循环分支时调用。

### 3.2 工具调度引擎

`agent/tool_registry.py` + `agent/tool_dispatcher.py`：

- **注册表**：`register`/`unregister`/`get`/`list_tools`/`describe_all`，外加 `as_langchain_tool()` 导出真正的 `langchain_core.tools.StructuredTool`（不是"声称兼容"，是真的构造出对象并调用 `.invoke()` 验证过）。
- **参数自动填充**：按工具的 pydantic `param_schema` 字段名，从当前 `AgentState` 里同名字段直接取值；预留了 `_llm_generate_params()` 扩展位供未来复杂场景用 LLM 动态生成参数，当前会显式抛 `NotImplementedError` 而不是静默返回空结果。
- **参数校验**：pydantic 模型校验，失败即 `NonRetryableError`（参数不合法不会因为重试变合法）。
- **调用执行 + 重试**：区分 `RetryableError`（网络超时/模型过载/向量库连接失败等瞬时性问题）与不可重试异常（参数错误、工具不存在、以及任何未预期的普通异常——对未知 bug 做无意义重试只会掩盖问题），可重试异常按指数退避重试（默认最多 3 次）。

`agent/retrieval_tool.py` 把 `RetrievalPipeline.retrieve()` 注册为第一个真实工具，是"工具抽象"这个设计理念第一次变成可运行代码——未来加新工具只需要照这个文件的模式写一个 `handler + param_schema` 再注册，不需要改动调度引擎或图本身。

### 3.3 会话记忆扩展

`api/session.py` 的改动是**纯增量**：`ConversationTurn` 新增 `agent_trace: list[dict] | None = None` 字段，`append_turn()` 新增同名可选参数，两处默认值都是 `None`——现有 RAG 模式的调用方（`api/routers/qa.py`）一行代码都不用改，行为完全不变。新增 `get_agent_trace(session_id, step=None)` 支持按会话 ID 取全部轨迹（跨轮次展平）或按步骤名筛选。

## 4. 测试与验证

### 4.1 单元测试（`tests/test_agent.py`，29 项，全 mock，0.2 秒）

| 测试类                       | 覆盖内容                                                                               | 结果     |
| ---------------------------- | -------------------------------------------------------------------------------------- | -------- |
| `TestAgentState`           | 默认值、超时判断、最大迭代轮次判断、trace 记录结构                                     | 5/5 通过 |
| `TestToolRegistry`         | 注册/注销/查询/JSON Schema 导出/LangChain 兼容导出                                     | 5/5 通过 |
| `TestToolDispatcher`       | 成功、重试后成功、重试耗尽、不可重试异常不重试、参数校验失败、工具未注册、参数自动填充 | 7/7 通过 |
| `TestGraphNodeTransitions` | 完整图运行、语言检测、会话上下文加载、检索失败不崩溃、上下文正确传入 LLM、去重逻辑     | 6/6 通过 |
| `TestSessionAgentTrace`    | RAG 模式轨迹为空、Agent 模式轨迹存储、跨轮次展平、按步骤筛选、边界情况                 | 6/6 通过 |

写测试过程中发现并修复了一处**测试自身的 bug**（不是被测代码的 bug）：直接调用节点函数后用 `dict.update()` 手动模拟状态合并，会绕过 LangGraph 的 `operator.add` reducer 导致 `tool_call_history` 被覆盖而非累加；修正为手动模拟 reducer 语义后测试通过。这个过程本身说明了"节点函数在图外单独测试"和"节点在真实图里跑"存在语义差异，是后续维护这套代码时需要留意的一点。

### 4.2 回归测试（既有 `tests/test_api.py`）

由于本周新增代码全部在新增文件里（`agent/` 整个包是新的），对现有代码的改动只有 `api/session.py` 的纯增量字段——重新跑了一遍现有的 21 项 API 集成测试确认零回归：**21/21 通过**，与本次改动前完全一致。

### 4.3 端到端验证（`test_agent_pipeline.py`）

5 条测试查询（4 条英文 + 1 条中文，覆盖基因关联、统计方法、睡眠记忆、信号通路、中文药理机制五个主题）跑完整状态机，并与现有单轮 RAG 流水线做同等步骤量级的对照。

| 指标                    | 结果                                                                        |
| ----------------------- | --------------------------------------------------------------------------- |
| Agent 执行成功率        | 5/5 = 100%                                                                  |
| 检索工具调用成功率      | 5/5 = 100%                                                                  |
| 累计工具重试次数        | 0                                                                           |
| 平均检索结果数          | 6（等于配置的`top_k`）                                                    |
| Agent 耗时              | avg 5.95s / min 3.87s / max 9.00s                                           |
| RAG 对照耗时            | avg 5.93s / min 4.55s / max 7.33s                                           |
| 各节点平均耗时          | entry ≈0s，tool_execution 0.36s，answer_generation 5.59s，termination ≈0s |
| 会话 agent_trace 持久化 | 5 轮全部成功写入；展平后 20 条记录；按步骤筛选正确返回 5 条                 |

## 5. 指标可靠性分析

逐条查询的"Agent 相对 RAG 开销"分别是 **+49.1% / −28.0% / −23.9% / −6.1% / +17.6%**，平均下来约 +1.7%——但这个平均值具有误导性：单条查询的波动幅度远大于平均值本身，是正负抵消凑出来的，不能解读为"Agent 编排开销稳定在 1.7%"。

真正可信、稳定的信号是**节点级耗时**：`tool_execution`（真实检索）稳定在 0.30–0.55 秒区间，`entry`/`termination` 全部约等于 0 秒（纯字典操作，符合预期）。占总耗时 90%+ 的 `answer_generation` 波动幅度是 `tool_execution` 的十几倍，根因有三个：

1. `answer_generator` 阶段温度 0.3（非确定性采样），同一问题每次生成的 token 数量不固定，耗时基本正比于 token 数；
2. 本机 GPU 与桌面环境共享，Ollama 推理速度受同时段其它进程负载影响；
3. 每条查询只跑一次（样本量 1），无法把上述噪声平均掉——上一轮 3 条查询测出的开销是"约 15-20%"，这一轮 5 条测出"约 1.7%"，两个数字都不是真实开销，只是同一个高噪声指标的两次不同抽样。

**结论**：结构性指标（成功率、重试次数、trace 步骤顺序、检索节点耗时）合理可信；端到端总耗时的"开销百分比"目前不具备统计意义，不应作为"Agent 编排几乎零开销"的结论引用。要拿到可信数字，需要把生成阶段温度设为 0、固定输出长度，同一查询重复跑 10-20 次取均值。

## 6. 交付物核对表

| 任务书要求                                | 交付位置                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| 全局 AgentState 数据结构                  | [`agent/state.py`](agent/state.py)                                             |
| 入口/工具执行/答案生成/终止四个节点       | [`agent/nodes.py`](agent/nodes.py)                                             |
| 节点串行流转 + 预留条件分支               | [`agent/graph.py`](agent/graph.py)                                             |
| 最大执行步数 + 超时机制                   | `recursion_limit`（graph.py）+ `should_terminate()`（state.py）             |
| 工具注册表（兼容 LangChain Tool）         | [`agent/tool_registry.py`](agent/tool_registry.py)                             |
| 参数自动填充 + 校验                       | `ToolDispatcherEngine.auto_fill_params/validate_params`                       |
| 统一结果格式（成功/失败/错误）            | `ToolCall` dataclass（state.py）                                              |
| 异常重试（可重试/不可重试区分，指数退避） | `ToolDispatcherEngine.dispatch`                                               |
| agent_trace 字段扩展会话存储              | [`api/session.py`](api/session.py)                                             |
| 按会话 ID / 步骤查询轨迹                  | `SessionManager.get_agent_trace()`                                            |
| 会话上下文共享（多轮继承）                | `SessionManager.build_context_prefix()`（已有，本周未改，entry 节点直接复用） |
| 节点流转逻辑单元测试                      | [`tests/test_agent.py`](tests/test_agent.py)`::TestGraphNodeTransitions`     |
| 单工具调用测试用例                        | `tests/test_agent.py::TestToolDispatcher`                                     |
| 轨迹查询测试用例                          | `tests/test_agent.py::TestSessionAgentTrace`                                  |

## 7. 明确未完成的工作

- **规划/反思/校验/整合四个组件**：`agent/interfaces.py` 里的 `TaskPlanner`/`RetrievalReflector`/`AnswerValidator`/`ResultIntegrator` 仍是 `Protocol` 占位，`subtasks`/`reflections`/`compliance_check` 字段本周节点不消费。
- **真正的迭代循环**：图目前是纯串行，`route_after_tool_execution` 恒定返回同一目的地，"补充检索""重新规划"两条边已声明但无判断逻辑驱动。
- **API 层 `mode=agent` 路由**：`api/models.py`/`api/routers/qa.py` 未改动，双模式兼容方案目前只是设计文档里的示意代码，尚未接入真实 HTTP 接口。
- **LayeredMemory 的工具结果缓存层**：会话记忆（`SessionManager`）已复用，但"工具结果缓存"（泛化 `GenerationCache`）这部分还没实现，重复调用同一检索目前不会命中缓存。
