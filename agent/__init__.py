"""
agent — Agentic RAG 架构：全局 State（TypedDict） + LangGraph 状态机 +
工具调度引擎 + 核心组件接口边界定义。

第 1 周实现范围：LangGraph 状态机落地（entry -> tool_execution ->
answer_generation -> termination）+ 工具调度引擎（注册/参数校验/重试）+
会话记忆 agent_trace 扩展。规划/反思/校验节点为预留字段，尚未实现
（见 agent/interfaces.py 的 Protocol 定义与 AGENT_ARCHITECTURE.md）。
"""

from .graph import build_agent_graph, route_after_tool_execution, run_agent
from .interfaces import (
    AnswerValidator,
    LayeredMemory,
    ResultIntegrator,
    RetrievalReflector,
    TaskPlanner,
)
from .interfaces import ToolDispatcher as ToolDispatcherProtocol
from .retrieval_tool import RetrievalToolParams, register_retrieval_tool
from .state import (
    AgentState,
    AgentStatus,
    ComplianceCheckResult,
    ReflectionRecord,
    SubTask,
    ToolCall,
    make_trace_entry,
    new_agent_state,
    should_terminate,
)
from .tool_dispatcher import NonRetryableError, RetryableError, ToolDispatcherEngine
from .tool_registry import ToolRegistry, ToolSpec

__all__ = [
    "AgentState",
    "AgentStatus",
    "SubTask",
    "ToolCall",
    "ReflectionRecord",
    "ComplianceCheckResult",
    "new_agent_state",
    "should_terminate",
    "make_trace_entry",
    "TaskPlanner",
    "ToolDispatcherProtocol",
    "LayeredMemory",
    "RetrievalReflector",
    "AnswerValidator",
    "ResultIntegrator",
    "ToolRegistry",
    "ToolSpec",
    "ToolDispatcherEngine",
    "RetryableError",
    "NonRetryableError",
    "register_retrieval_tool",
    "RetrievalToolParams",
    "build_agent_graph",
    "run_agent",
    "route_after_tool_execution",
]
