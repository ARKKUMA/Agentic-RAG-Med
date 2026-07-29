"""
agent/state.py — Agent 全局状态（AgentState）定义

这是架构设计"边界定义"阶段的产物：只定义数据结构与终止条件判断，
不包含规划/执行/反思的业务逻辑（那属于未来 AgentExecutor 实现的范畴，
设计方案见项目根目录 AGENT_ARCHITECTURE.md）。

覆盖任务书要求的全链路字段：原始查询、子任务列表、工具调用历史、检索结果集、
中间推理结论、反思记录、合规校验结果、最终答案。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


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


@dataclass
class SubTask:
    """任务规划器（TaskPlanner）产出的子任务。"""

    description: str
    priority: int = 0                                      # 数字越小优先级越高
    id: str = field(default_factory=_new_id)
    status: Literal["pending", "in_progress", "done", "failed", "skipped"] = "pending"
    depends_on: list[str] = field(default_factory=list)     # 依赖的其它 subtask id
    assigned_tool: str | None = None                        # 规划器建议的工具名，调度器最终决定是否采用
    result_ref: str | None = None                           # 指向 tool_call_history 中对应 ToolCall.id


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
    retry_count: int = 0
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float | None = None


@dataclass
class RetrievalResultRef:
    """
    检索结果集的轻量引用。不直接内联完整 chunk 内容（避免 State 随迭代轮次
    无限膨胀）——完整结果仍留在对应 ToolCall.result 里，这里只存关键统计量，
    供规划器/反思器快速判断"这个子任务的检索结果大概什么样"而无需反序列化全量数据。
    """

    tool_call_id: str
    subtask_id: str | None
    n_results: int
    top_relevance_score: float | None = None


@dataclass
class ReflectionRecord:
    """检索反思器（RetrievalReflector）的输出记录。"""

    subtask_id: str | None
    round: int                                              # 同一子任务可能被反思多次，round 从 1 开始
    sufficiency: Literal["sufficient", "partial", "insufficient"]
    reasoning: str
    id: str = field(default_factory=_new_id)
    follow_up_suggestion: str | None = None                 # 建议的补充检索方向/查询改写
    created_at: float = field(default_factory=time.time)


@dataclass
class ComplianceCheckResult:
    """答案校验器（AnswerValidator）的输出。"""

    passed: bool
    citation_valid: bool
    format_valid: bool
    hallucination_risk_score: float
    issues: list[str] = field(default_factory=list)
    revision_instructions: str | None = None


@dataclass
class AgentState:
    """
    Agent 执行全生命周期的全局状态，在 规划->执行->反思->整合->校验 各环节间
    传递、累积、更新（状态流转规则见 AGENT_ARCHITECTURE.md 第 2.3 节）。

    字段分组：
      - 输入与元信息：query / session_id / language / created_at
      - 规划：subtasks
      - 执行：tool_call_history / retrieval_results
      - 推理：intermediate_conclusions
      - 反思：reflections
      - 校验：compliance_check
      - 输出：final_answer / sources
      - 控制：status / iteration_count / max_iterations / timeout_seconds / error
    """

    # 输入与元信息
    query: str
    session_id: str | None = None
    language: str = "en"
    created_at: float = field(default_factory=time.time)

    # 规划
    subtasks: list[SubTask] = field(default_factory=list)

    # 执行
    tool_call_history: list[ToolCall] = field(default_factory=list)
    retrieval_results: list[RetrievalResultRef] = field(default_factory=list)

    # 推理
    intermediate_conclusions: list[str] = field(default_factory=list)

    # 反思
    reflections: list[ReflectionRecord] = field(default_factory=list)

    # 校验
    compliance_check: ComplianceCheckResult | None = None

    # 输出
    final_answer: str | None = None
    sources: list[dict] = field(default_factory=list)       # 溯源信息；结构复用 generation 层已有的 source item

    # 控制（兜底机制）
    status: AgentStatus = AgentStatus.PLANNING
    iteration_count: int = 0
    max_iterations: int = 5
    timeout_seconds: float = 90.0
    error: str | None = None

    @staticmethod
    def new(
        query: str,
        session_id: str | None = None,
        language: str = "en",
        max_iterations: int = 5,
        timeout_seconds: float = 90.0,
    ) -> "AgentState":
        return AgentState(
            query=query,
            session_id=session_id,
            language=language,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )

    # ── 终止条件判断（供 AgentExecutor 在每轮循环开始前调用）──────
    def is_elapsed_timeout(self) -> bool:
        return (time.time() - self.created_at) > self.timeout_seconds

    def is_max_iterations_reached(self) -> bool:
        return self.iteration_count >= self.max_iterations

    def should_terminate(self) -> tuple[bool, AgentStatus | None]:
        """返回 (是否应终止, 终止时应置的状态)；不应终止时第二个值为 None。"""
        if self.is_elapsed_timeout():
            return True, AgentStatus.TIMEOUT
        if self.is_max_iterations_reached():
            return True, AgentStatus.MAX_ITERATIONS_REACHED
        return False, None

    def pending_subtasks(self) -> list[SubTask]:
        return [t for t in self.subtasks if t.status == "pending"]

    def latest_reflection_for(self, subtask_id: str) -> ReflectionRecord | None:
        candidates = [r for r in self.reflections if r.subtask_id == subtask_id]
        return max(candidates, key=lambda r: r.round) if candidates else None
