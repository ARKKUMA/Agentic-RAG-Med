"""
agent/interfaces.py — 核心组件接口定义（边界定义，不含实现）

六个核心组件的职责边界与输入输出契约，全部用 typing.Protocol 定义——
描述"必须暴露什么方法"而不规定具体基类，方便未来替换实现（甚至换成跨进程/
跨语言的实现）而不破坏调用方代码。每个 Protocol 的 docstring 里注明了它与
现有 retrieval/generation/api 模块的复用关系——新组件是对现有能力的封装/
泛化，不是重新发明。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .state import AgentState, ComplianceCheckResult, ReflectionRecord, SubTask, ToolCall


@runtime_checkable
class TaskPlanner(Protocol):
    """
    医学任务规划器：输入用户查询，输出结构化子任务列表与执行优先级。

    复用关系：规划时可调用现有 retrieval.query_processor.MedicalQueryProcessor
    做实体识别/缩写展开/过滤条件提取，把提取出的信号（如"需要时间范围过滤"、
    "涉及多个药物比较"）转化为拆分子任务的依据，而不是从零设计一套新的
    查询理解逻辑。
    """

    def plan(self, state: AgentState) -> list[SubTask]:
        """
        基于 state.query（以及可选的 state.session_id 关联的历史对话）生成
        子任务列表。首轮调用时 state.subtasks 为空列表；补充规划场景下会带着
        已有的 state.reflections 再次调用，此时应返回"新增或修订"的子任务，
        而不是从头重新规划已完成的部分。
        """
        ...


@runtime_checkable
class ToolDispatcher(Protocol):
    """
    工具调度器：统一管理工具注册、参数校验、异常重试与执行结果记录。

    这是本次任务标题"工具抽象"的核心落点——现有的
    retrieval.pipeline.RetrievalPipeline.retrieve() 不再被当作流水线里
    硬编码的一步，而是被当作"第一个注册的工具"，参数用 pydantic 模型校验。
    未来可以用同样的方式注册更多工具（如药物相互作用查询、单位换算、
    面向 PubMed 的实时检索），Agent 的规划/执行循环本身不需要改动。

    示例（设计示意，供未来实现参考，不属于本次交付代码）：

        class RetrievalToolParams(BaseModel):
            query: str
            top_k: int = 8
            fusion_strategy: str = "rrf"

        def retrieval_tool_handler(params: RetrievalToolParams) -> dict:
            return retrieval_pipeline.retrieve(
                params.query, top_k=params.top_k, fusion_strategy=params.fusion_strategy,
            )

        dispatcher.register_tool("retrieval", retrieval_tool_handler, RetrievalToolParams)
    """

    def register_tool(self, name: str, handler: Any, param_schema: type, max_retries: int = 2) -> None:
        """注册一个工具：名称、可调用对象、参数校验模型、失败重试次数上限。"""
        ...

    def dispatch(self, tool_name: str, arguments: dict, subtask_id: str | None = None) -> ToolCall:
        """
        校验参数 -> 调用工具 -> 记录结果；调用异常时按注册时配置的重试次数退避重试，
        重试耗尽后仍失败则返回 success=False 的 ToolCall（不抛出异常中断整个
        Agent 循环——单个工具失败应可降级，不应让整个查询失败）。
        返回值即为应当 append 进 state.tool_call_history 的记录。
        """
        ...

    def list_tools(self) -> list[str]:
        ...


@runtime_checkable
class LayeredMemory(Protocol):
    """
    分层记忆模块：会话记忆 + 工具结果缓存两层。

    复用关系：
      - 会话记忆层直接复用 api.session.SessionManager（已实现的 TTL 会话存储 +
        build_context_prefix 指代消解渲染），不重新实现一遍。
      - 工具结果缓存层复用/泛化 generation.cache.GenerationCache 的设计思路
        （键哈希 + TTL 淘汰），但第 2 周实现（agent/memory.py::AgentMemory）
        换成了 Redis 后端而不是进程内 LRU——原因是任务书明确要求"缓存与会话
        生命周期绑定、会话过期自动清理"，Redis 的 EXPIRE/TTL 原生支持这个
        语义，且为未来跨进程/多实例部署留出了空间。

    第 2 周落地实现（agent/memory.py::AgentMemory）：
      - get_cached_tool_result / cache_tool_result 新增可选 session_id 参数——
        缓存与会话生命周期绑定后，键里必须包含 session_id 才能实现"会话过期
        自动清理对应缓存"（同一 Redis key 的 TTL 直接对齐 session 的 TTL）。
        不传 session_id 时退化为全局命名空间（不与任何会话绑定，需要调用方
        自行管理失效），保持对未来非会话场景工具调用的兼容。
      - 额外提供 register_chunks / get_seen_chunk_ids（文献块去重存储层），
        Protocol 未声明这两个方法——结构化子类型（duck typing）允许具体实现
        提供比 Protocol 更多的方法，不影响 isinstance 检查。
    """

    def get_session_context(self, session_id: str) -> str:
        """渲染历史对话为简短前缀，语义与现有 SessionManager.build_context_prefix 一致。"""
        ...

    def get_cached_tool_result(
        self, tool_name: str, arguments: dict, session_id: str | None = None,
    ) -> Any | None:
        ...

    def cache_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        ttl_seconds: float | None = None,
        session_id: str | None = None,
    ) -> None:
        ...


@runtime_checkable
class RetrievalReflector(Protocol):
    """
    检索反思器：输入已检索结果与子任务，输出信息充足度判断与补检索建议。

    复用关系：是现有 generation.prompt_templates 里 evidence_evaluator 提示词
    阶段的泛化版本——现有阶段只对"单次检索的上下文"判断一次 sufficient/
    partial/insufficient；这里需要支持"同一子任务可能经过多轮/多工具检索"，
    每轮都产出一条 ReflectionRecord（state.reflections 是列表而非单值，
    就是为了保留这个多轮历史）。
    """

    def reflect(self, subtask: SubTask, tool_calls: list[ToolCall], round: int) -> ReflectionRecord:
        """评估与该 subtask 相关的 tool_calls 结果是否足以支撑回答，给出该轮的判断记录。"""
        ...


@runtime_checkable
class AnswerValidator(Protocol):
    """
    答案校验器：输入生成答案，输出合规性校验结果与修正意见。

    复用关系：直接组合现有三个已实现的校验器，不重新实现：
      - generation.citation_validator.CitationValidator —— 引用编号有效性
      - generation.format_checker.FormatChecker —— 章节结构/缩写全称/参考文献完整性
      - generation.answer_evaluator.AnswerEvaluator.evaluate_hallucination_risk —— 幻觉信号评分
    ComplianceCheckResult 的三个布尔/分数字段就是这三者各自结果的直接映射。
    """

    def validate(self, answer: str, state: AgentState) -> ComplianceCheckResult:
        ...


@runtime_checkable
class ResultIntegrator(Protocol):
    """
    结果整合器：输入多轮检索结果与推理结论，输出结构化最终答案与溯源信息。

    复用关系：是现有 generation.pipeline.MedicalGenerationPipeline 里
    final_assembler 阶段 + _postprocess 引用格式化逻辑的泛化版本——区别在于
    输入可能来自多个子任务、多轮工具调用累积的 state.intermediate_conclusions
    与 state.tool_call_history，而不是单一的 draft_answer。
    """

    def integrate(self, state: AgentState) -> tuple[str, list[dict]]:
        """返回 (final_answer, sources)；sources 结构与现有 QAResponseData.sources 保持一致。"""
        ...
