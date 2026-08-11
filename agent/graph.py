"""
agent/graph.py — LangGraph 状态机组装
串联 entry -> tool_execution -> answer_generation -> termination 四个节点，
预留条件分支（本周恒定路由到下一个固定节点；未来周次替换判断函数即可接入
反思迭代/异常兜底分支，不需要重新布线整张图的拓扑结构）。

兜底机制：
  - LangGraph 自带的 recursion_limit（invoke() 时通过 config 传入）是防止
    节点间出现意外循环导致无限执行的最后一道防线——本周是纯串行图，正常
    情况下远用不到，属于防御性兜底。
  - agent.state.should_terminate()（超时/达最大迭代轮次）供未来引入循环
    分支时在条件路由函数里调用；本周主链路是无条件走完四个节点，不会
    触发中途终止，先把判断函数留在 state.py 里备用。
"""

from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from generation.context_assembler import ContextAssembler
from generation.llm_generator import LLMGenerator

from .memory import AgentMemory
from .nodes import make_answer_generation_node, make_entry_node, make_termination_node, make_tool_execution_node
from .state import AgentState
from .tool_dispatcher import ToolDispatcherEngine

log = logging.getLogger("agent.graph")

DEFAULT_RECURSION_LIMIT = 25   # 四节点串行图正常只需 4~5 步，25 是防御性上限，非正常预期值


def route_after_tool_execution(state: AgentState) -> str:
    """
    条件分支占位：本周恒定路由到 answer_generation。

    预留给未来周次替换为"信息是否充分"的判断——不充分时可路由回
    tool_execution（补充检索）或 entry（重新规划），充分则路由到
    answer_generation。图的拓扑结构本身不需要因此改变，只需要替换这个
    函数内部的判断逻辑，path_map 里已经声明了全部三条可能的出边。
    """
    return "answer_generation"


def build_agent_graph(
    retrieval_dispatcher: ToolDispatcherEngine,
    llm: LLMGenerator,
    session_context_fn: Callable[[str], str] | None = None,
    context_assembler: ContextAssembler | None = None,
    memory: AgentMemory | None = None,
) -> CompiledStateGraph:
    """
    组装并编译 Agent 状态机。

    Args:
        retrieval_dispatcher: 已注册好 "retrieval" 工具的 ToolDispatcherEngine
        llm: 生成初始答案用的 LLMGenerator
        session_context_fn: 可选，entry 节点读取历史会话上下文的函数
            （典型传入 api.session.SessionManager.build_context_prefix）
        context_assembler: 可选，answer_generation 节点用的上下文组装器
        memory: 可选，第 2 周新增的 AgentMemory（检索结果缓存 + 文献块去重）；
            不传时 tool_execution 节点行为与第 1 周完全一致
    """
    graph = StateGraph(AgentState)

    graph.add_node("entry", make_entry_node(session_context_fn=session_context_fn))
    graph.add_node("tool_execution", make_tool_execution_node(retrieval_dispatcher, memory=memory))
    graph.add_node("answer_generation", make_answer_generation_node(llm, context_assembler))
    graph.add_node("termination", make_termination_node())

    graph.add_edge(START, "entry")
    graph.add_edge("entry", "tool_execution")
    graph.add_conditional_edges(
        "tool_execution",
        route_after_tool_execution,
        {
            "answer_generation": "answer_generation",
            "tool_execution": "tool_execution",   # 预留：未来补充检索
            "entry": "entry",                     # 预留：未来重新规划
        },
    )
    graph.add_edge("answer_generation", "termination")
    graph.add_edge("termination", END)

    return graph.compile()


def run_agent(
    compiled_graph: CompiledStateGraph,
    initial_state: AgentState,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> AgentState:
    """执行编译好的状态机；recursion_limit 是 LangGraph 层面的防御性兜底（见模块顶部说明）。"""
    return compiled_graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
