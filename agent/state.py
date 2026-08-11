"""
agent/state.py — Agent 全局状态（AgentState）定义

第 1 周实现说明：上周（架构设计阶段）AgentState 是一个纯 dataclass，仅用于
边界定义。本周基于 LangGraph 落地状态机，LangGraph 的 StateGraph 按节点
返回的"部分更新"字典去合并状态，因此容器类型改为 TypedDict（节点函数签名
天然是 "输入 State，返回 dict"），可累加字段用 Annotated[..., operator.add]
声明 reducer；子结构（SubTask/ToolCall/ReflectionRecord/ComplianceCheckResult）
的字段语义与上周设计完全一致，仍是普通 dataclass（作为 list 元素类型使用，
不受 TypedDict 的合并规则约束）。

覆盖任务书本周要求的字段分组：
  基础信息：原始查询、会话 ID、运行配置参数（top_k/融合策略/最大迭代轮次等）
  任务信息：子问题列表（预留）、当前执行步骤、执行状态标记
  执行记录：工具调用历史（调用 ID/工具名称/入参/返回结果/耗时/调用状态）
  业务数据：去重后的检索结果集、中间答案文本、反思记录（预留）、合规校验结果（预留）
  异常字段：错误信息、重试次数标记
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict


class AgentStatus(str, Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    INTEGRATING = "integrating"
    VALIDATING = "validating"
    DONE = "done"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ══════════════════════════════════════════════════════════════════
# 子结构（列表元素类型，语义与架构设计阶段一致）
# ══════════════════════════════════════════════════════════════════

@dataclass
class SubTask:
    """任务规划器（TaskPlanner，未来周次实现）产出的子任务——本周为预留字段，节点不消费。"""

    description: str
    priority: int = 0
    id: str = field(default_factory=_new_id)
    status: Literal["pending", "in_progress", "done", "failed", "skipped"] = "pending"
    depends_on: list[str] = field(default_factory=list)
    assigned_tool: str | None = None
    result_ref: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCall:
    """工具调度器（ToolDispatcher）的每一次调用记录。"""

    tool_name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=_new_id)
    subtask_id: str | None = None
    result: Any | None = None
    success: bool = False
    error: str | None = None
    retryable: bool = False
    retry_count: int = 0
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float | None = None
    cached: bool = False   # 第 2 周新增：命中 AgentMemory 检索结果缓存时为 True，未真正调用工具

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReflectionRecord:
    """检索反思器（未来周次实现）的输出记录——本周为预留字段。"""

    subtask_id: str | None
    round: int
    sufficiency: Literal["sufficient", "partial", "insufficient"]
    reasoning: str
    id: str = field(default_factory=_new_id)
    follow_up_suggestion: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComplianceCheckResult:
    """答案校验器（未来周次实现）的输出——本周为预留字段。"""

    passed: bool
    citation_valid: bool
    format_valid: bool
    hallucination_risk_score: float
    issues: list[str] = field(default_factory=list)
    revision_instructions: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
# LangGraph 状态容器
# ══════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """
    LangGraph StateGraph 的状态 schema。`total=False` 允许节点只返回发生变化
    的字段（LangGraph 按字段合并，未出现的键保持不变）。

    reducer 说明：
      - `tool_call_history` / `execution_trace` / `reflections` 用
        `operator.add` 累加——每个节点只需返回"本节点新增的部分"，
        不需要自己读旧值再拼接。
      - `retrieval_results` **不用** `operator.add`：它存的是"去重后的
        当前完整集合"，每次由 tool_execution_node 重新计算整体去重结果并
        整体覆盖，如果用 operator.add 会导致同一 chunk 在多轮里被重复相加。
      - 标量字段（query/current_step/execution_status/...）默认覆盖语义，
        节点返回最新值即可。
    """

    # ── 基础信息 ──
    query: str
    session_id: str | None
    top_k: int
    fusion_strategy: str
    where_filter: dict | None                                   # 第 2 周新增：显式元数据过滤条件，透传给检索工具
    max_iterations: int
    timeout_seconds: float
    language: str
    conversation_context: str                                   # entry 节点渲染的历史对话前缀，仅供消解指代

    # ── 任务信息 ──
    subtasks: list[SubTask]                                    # 预留字段
    current_step: str                                          # 当前执行到的节点名
    execution_status: AgentStatus

    # ── 执行记录（累加）──
    tool_call_history: Annotated[list[ToolCall], operator.add]

    # ── 业务数据 ──
    retrieval_results: list[dict]                               # 去重后的检索结果集，整体覆盖写回
    intermediate_answer: str | None
    reflections: Annotated[list[ReflectionRecord], operator.add]        # 预留
    compliance_check: ComplianceCheckResult | None                       # 预留

    # ── 异常字段 ──
    error: str | None
    retry_count: int                                            # 本次运行累计触发的工具重试次数

    # ── 输出与轨迹 ──
    final_answer: str | None
    sources: list[dict]
    execution_trace: Annotated[list[dict], operator.add]        # 每节点一条记录，供 session 持久化

    # ── 控制（兜底机制）──
    created_at: float
    iteration_count: int


DEFAULT_MAX_ITERATIONS = 5
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_TOP_K = 8
DEFAULT_FUSION_STRATEGY = "rrf"


def new_agent_state(
    query: str,
    session_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    fusion_strategy: str = DEFAULT_FUSION_STRATEGY,
    where_filter: dict | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    language: str = "en",
) -> AgentState:
    """构造初始 AgentState（TypedDict 没有构造方法/classmethod，用工厂函数代替）。"""
    return AgentState(
        query=query,
        session_id=session_id,
        top_k=top_k,
        fusion_strategy=fusion_strategy,
        where_filter=where_filter,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        language=language,
        conversation_context="",
        subtasks=[],
        current_step="entry",
        execution_status=AgentStatus.PLANNING,
        tool_call_history=[],
        retrieval_results=[],
        intermediate_answer=None,
        reflections=[],
        compliance_check=None,
        error=None,
        retry_count=0,
        final_answer=None,
        sources=[],
        execution_trace=[],
        created_at=time.time(),
        iteration_count=0,
    )


# ══════════════════════════════════════════════════════════════════
# 终止条件判断（模块级函数——TypedDict 运行时就是普通 dict，没有方法）
# ══════════════════════════════════════════════════════════════════

def is_elapsed_timeout(state: AgentState) -> bool:
    created_at = state.get("created_at", 0.0)
    timeout_seconds = state.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    return (time.time() - created_at) > timeout_seconds


def is_max_iterations_reached(state: AgentState) -> bool:
    return state.get("iteration_count", 0) >= state.get("max_iterations", DEFAULT_MAX_ITERATIONS)


def should_terminate(state: AgentState) -> tuple[bool, AgentStatus | None]:
    """返回 (是否应终止, 终止时应置的状态)；不应终止时第二个值为 None。"""
    if is_elapsed_timeout(state):
        return True, AgentStatus.TIMEOUT
    if is_max_iterations_reached(state):
        return True, AgentStatus.MAX_ITERATIONS_REACHED
    return False, None


def pending_subtasks(state: AgentState) -> list[SubTask]:
    return [t for t in state.get("subtasks", []) if t.status == "pending"]


def latest_reflection_for(state: AgentState, subtask_id: str) -> ReflectionRecord | None:
    candidates = [r for r in state.get("reflections", []) if r.subtask_id == subtask_id]
    return max(candidates, key=lambda r: r.round) if candidates else None


def make_trace_entry(
    step: str,
    inputs: dict,
    outputs: dict,
    elapsed_seconds: float,
    status: Literal["success", "failed"] = "success",
    error: str | None = None,
) -> dict:
    """
    统一构造 execution_trace 里的一条记录（每个节点执行完毕后调用）。
    结构化字段与任务书"按步骤记录输入、输出、耗时、执行状态"一一对应，
    供 api/session.py 持久化与前端按步骤筛选展示。
    """
    return {
        "step": step,
        "inputs": inputs,
        "outputs": outputs,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "status": status,
        "error": error,
        "timestamp": time.time(),
    }
