"""
agent — Agentic RAG 架构的边界定义（State 数据结构 + 核心组件接口）。

这是设计阶段的产物：只有数据结构与 Protocol 接口，不含 AgentExecutor 的
执行逻辑实现。完整设计说明见项目根目录 AGENT_ARCHITECTURE.md。
"""

from .interfaces import AnswerValidator, LayeredMemory, ResultIntegrator, RetrievalReflector, TaskPlanner, ToolDispatcher
from .state import (
    AgentState,
    AgentStatus,
    ComplianceCheckResult,
    ReflectionRecord,
    RetrievalResultRef,
    SubTask,
    ToolCall,
)

__all__ = [
    "AgentState",
    "AgentStatus",
    "SubTask",
    "ToolCall",
    "RetrievalResultRef",
    "ReflectionRecord",
    "ComplianceCheckResult",
    "TaskPlanner",
    "ToolDispatcher",
    "LayeredMemory",
    "RetrievalReflector",
    "AnswerValidator",
    "ResultIntegrator",
]
