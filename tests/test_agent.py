"""
tests/test_agent.py — Agent 底座单元测试（标准库 unittest，无需真实模型）

覆盖任务书要求的三类测试：
  - 节点流转逻辑单元测试（TestAgentState, TestGraphNodeTransitions）
  - 单工具调用测试用例（TestToolRegistry, TestToolDispatcher）
  - 轨迹查询测试用例（TestSessionAgentTrace）

本文件全部使用 mock 的检索/LLM 组件，不加载真实 BGE/ChromaDB/reranker/
Ollama——这些是纯逻辑/编排正确性测试，运行应是毫秒级。真实组件的端到端
回归验证（确认单轮 RAG 功能无退化）是单独的手动/集成验证脚本，不在这里。

运行：
    PYTHONUTF8=1 python -m unittest tests.test_agent -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from agent import (
    NonRetryableError,
    RetryableError,
    ToolDispatcherEngine,
    ToolRegistry,
    ToolSpec,
    build_agent_graph,
    new_agent_state,
    should_terminate,
)
from agent.state import make_trace_entry
from api.session import SessionManager


# ══════════════════════════════════════════════════════════════════
# AgentState 单元测试
# ══════════════════════════════════════════════════════════════════

class TestAgentState(unittest.TestCase):
    def test_new_agent_state_defaults(self):
        state = new_agent_state(query="What is GWAS?")
        self.assertEqual(state["query"], "What is GWAS?")
        self.assertEqual(state["iteration_count"], 0)
        self.assertEqual(state["max_iterations"], 5)
        self.assertEqual(state["tool_call_history"], [])
        self.assertEqual(state["retrieval_results"], [])
        self.assertEqual(state["execution_trace"], [])
        self.assertIsNone(state["final_answer"])

    def test_should_terminate_false_initially(self):
        state = new_agent_state(query="test", max_iterations=5, timeout_seconds=90.0)
        terminate, status = should_terminate(state)
        self.assertFalse(terminate)
        self.assertIsNone(status)

    def test_should_terminate_on_timeout(self):
        state = new_agent_state(query="test", timeout_seconds=0.01)
        time.sleep(0.02)
        terminate, status = should_terminate(state)
        self.assertTrue(terminate)
        self.assertEqual(status.value, "timeout")

    def test_should_terminate_on_max_iterations(self):
        state = new_agent_state(query="test", max_iterations=1)
        state["iteration_count"] = 1
        terminate, status = should_terminate(state)
        self.assertTrue(terminate)
        self.assertEqual(status.value, "max_iterations_reached")

    def test_make_trace_entry_structure(self):
        entry = make_trace_entry(
            step="entry", inputs={"a": 1}, outputs={"b": 2}, elapsed_seconds=0.123,
        )
        self.assertEqual(entry["step"], "entry")
        self.assertEqual(entry["inputs"], {"a": 1})
        self.assertEqual(entry["outputs"], {"b": 2})
        self.assertEqual(entry["status"], "success")
        self.assertIsNone(entry["error"])
        self.assertIn("timestamp", entry)


# ══════════════════════════════════════════════════════════════════
# 工具注册表单元测试
# ══════════════════════════════════════════════════════════════════

class EchoParams(BaseModel):
    query: str


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(ToolSpec(
            name="echo", handler=lambda query: {"echo": query},
            param_schema=EchoParams, description="echo tool", max_retries=2,
        ))

    def test_register_and_get(self):
        spec = self.registry.get("echo")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "echo")

    def test_list_tools(self):
        self.assertEqual(self.registry.list_tools(), ["echo"])

    def test_unregister(self):
        self.assertTrue(self.registry.unregister("echo"))
        self.assertIsNone(self.registry.get("echo"))
        self.assertFalse(self.registry.unregister("echo"))  # 第二次注销同一个，应返回 False

    def test_describe_all_includes_json_schema(self):
        described = self.registry.describe_all()
        self.assertEqual(len(described), 1)
        self.assertIn("param_schema", described[0])
        self.assertIn("properties", described[0]["param_schema"])

    def test_as_langchain_tool_compat(self):
        lc_tool = self.registry.as_langchain_tool("echo")
        self.assertEqual(lc_tool.name, "echo")
        result = lc_tool.invoke({"query": "hi"})
        self.assertEqual(result, {"echo": "hi"})


# ══════════════════════════════════════════════════════════════════
# 工具调度引擎单元测试（单工具调用测试用例）
# ══════════════════════════════════════════════════════════════════

class TestToolDispatcher(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.dispatcher = ToolDispatcherEngine(self.registry, backoff_base_seconds=0.01)

    def _register(self, name, handler, max_retries=3):
        self.registry.register(ToolSpec(name=name, handler=handler, param_schema=EchoParams, max_retries=max_retries))

    def test_dispatch_success(self):
        self._register("echo", lambda query: {"echo": query})
        call = self.dispatcher.dispatch("echo", {"query": "hi"})
        self.assertTrue(call.success)
        self.assertEqual(call.result, {"echo": "hi"})
        self.assertEqual(call.retry_count, 0)

    def test_dispatch_retries_then_succeeds(self):
        counter = {"n": 0}

        def flaky(query):
            counter["n"] += 1
            if counter["n"] < 3:
                raise RetryableError(f"transient #{counter['n']}")
            return {"ok": True}

        self._register("flaky", flaky, max_retries=3)
        call = self.dispatcher.dispatch("flaky", {"query": "hi"})
        self.assertTrue(call.success)
        self.assertEqual(call.retry_count, 2)

    def test_dispatch_exhausts_retries(self):
        def always_down(query):
            raise RetryableError("permanently down")

        self._register("always_down", always_down, max_retries=2)
        call = self.dispatcher.dispatch("always_down", {"query": "hi"})
        self.assertFalse(call.success)
        self.assertTrue(call.retryable)
        self.assertEqual(call.retry_count, 2)
        self.assertIn("permanently down", call.error)

    def test_dispatch_non_retryable_no_retry(self):
        attempts = {"n": 0}

        def broken(query):
            attempts["n"] += 1
            raise ValueError("bug, not transient")

        self._register("broken", broken, max_retries=3)
        call = self.dispatcher.dispatch("broken", {"query": "hi"})
        self.assertFalse(call.success)
        self.assertFalse(call.retryable)
        self.assertEqual(call.retry_count, 0)
        self.assertEqual(attempts["n"], 1)  # 只应尝试一次，不重试

    def test_dispatch_invalid_params_non_retryable(self):
        self._register("echo", lambda query: {"echo": query})
        call = self.dispatcher.dispatch("echo", {})  # 缺少必填字段 query
        self.assertFalse(call.success)
        self.assertFalse(call.retryable)

    def test_dispatch_unregistered_tool(self):
        call = self.dispatcher.dispatch("does_not_exist", {"query": "hi"})
        self.assertFalse(call.success)
        self.assertIn("未注册", call.error)

    def test_auto_fill_params_filters_by_schema(self):
        self._register("echo", lambda query: {"echo": query})
        state = {"query": "hello", "top_k": 6, "fusion_strategy": "rrf", "unrelated": "x"}
        args = self.dispatcher.auto_fill_params("echo", state)
        self.assertEqual(args, {"query": "hello"})


# ══════════════════════════════════════════════════════════════════
# LangGraph 节点流转测试（用 mock 检索/LLM，不加载真实模型）
# ══════════════════════════════════════════════════════════════════

class MockRetrievalPipeline:
    """模拟 RetrievalPipeline.retrieve()，避免加载真实 BGE/ChromaDB/reranker。"""

    def __init__(self, results=None, raise_error: Exception | None = None):
        self._results = results if results is not None else [
            {"chunk_id": "PMC1_body_0", "text": "Metformin activates AMPK.",
             "metadata": {"journal": "Test J", "pub_year": 2020, "pmc_id": "PMC1"},
             "final_score": 0.9, "final_rank": 1},
            {"chunk_id": "PMC2_body_0", "text": "GWAS studies identify risk loci.",
             "metadata": {"journal": "Test J2", "pub_year": 2019, "pmc_id": "PMC2"},
             "final_score": 0.7, "final_rank": 2},
        ]
        self._raise_error = raise_error

    def retrieve(self, query, top_k=8, fusion_strategy="rrf"):
        if self._raise_error is not None:
            raise self._raise_error
        return {"query_info": None, "fused_candidates": len(self._results), "results": self._results}


class MockLLM:
    def __init__(self, response_text="Mock answer citing [来源 1] and [来源 2]."):
        self.response_text = response_text
        self.calls = []

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=900):
        self.calls.append(prompt)
        return {"text": self.response_text, "elapsed_seconds": 0.001, "model": "mock", "done": True, "cached": False}


def _build_test_graph(retrieval_pipeline=None, llm=None, session_context_fn=None):
    from agent.retrieval_tool import register_retrieval_tool

    registry = ToolRegistry()
    register_retrieval_tool(registry, retrieval_pipeline or MockRetrievalPipeline())
    dispatcher = ToolDispatcherEngine(registry)
    return build_agent_graph(
        retrieval_dispatcher=dispatcher,
        llm=llm or MockLLM(),
        session_context_fn=session_context_fn,
    )


class TestGraphNodeTransitions(unittest.TestCase):
    def test_full_graph_run_happy_path(self):
        graph = _build_test_graph()
        result = graph.invoke(new_agent_state(query="What is metformin's mechanism?", top_k=5))

        self.assertEqual(result["execution_status"].value, "done")
        self.assertIn("Mock answer", result["final_answer"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["chunk_id"], "PMC1_body_0")
        self.assertEqual(len(result["tool_call_history"]), 1)
        self.assertTrue(result["tool_call_history"][0].success)

        trace_steps = [t["step"] for t in result["execution_trace"]]
        self.assertEqual(trace_steps, ["entry", "tool_execution", "answer_generation", "termination"])

    def test_entry_node_detects_language_and_loads_session_context(self):
        captured = {}

        def fake_session_context(session_id):
            captured["session_id"] = session_id
            return "(fake history)\n\n"

        graph = _build_test_graph(session_context_fn=fake_session_context)
        result = graph.invoke(new_agent_state(query="What is GWAS?", session_id="sess-123"))

        self.assertEqual(captured["session_id"], "sess-123")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["conversation_context"], "(fake history)\n\n")

    def test_chinese_query_detected_as_zh(self):
        graph = _build_test_graph()
        result = graph.invoke(new_agent_state(query="二甲双胍的作用机制是什么？"))
        self.assertEqual(result["language"], "zh")

    def test_tool_execution_failure_propagates_without_crashing(self):
        graph = _build_test_graph(retrieval_pipeline=MockRetrievalPipeline(raise_error=ValueError("boom")))
        result = graph.invoke(new_agent_state(query="test query"))

        # 检索失败：应记录失败的 ToolCall，但图仍应跑完全程而不是崩溃
        self.assertFalse(result["tool_call_history"][0].success)
        self.assertEqual(result["retrieval_results"], [])
        self.assertEqual(result["execution_status"].value, "done")  # LLM 仍基于空上下文生成了兜底回答
        trace_steps = [t["step"] for t in result["execution_trace"]]
        self.assertEqual(trace_steps, ["entry", "tool_execution", "answer_generation", "termination"])

    def test_answer_generation_receives_assembled_context(self):
        llm = MockLLM()
        graph = _build_test_graph(llm=llm)
        graph.invoke(new_agent_state(query="What is metformin's mechanism?"))

        self.assertEqual(len(llm.calls), 1)
        prompt = llm.calls[0]
        self.assertIn("What is metformin's mechanism?", prompt)
        self.assertIn("Metformin activates AMPK", prompt)  # 检索到的文本应出现在拼装后的 prompt 里

    def test_retrieval_results_deduplicated_across_repeated_tool_calls(self):
        """
        直接调用 tool_execution 节点两次，模拟未来补充检索场景，验证去重逻辑。

        注意：这里手动模拟 LangGraph 的 reducer 合并语义（tool_call_history
        用 operator.add 累加，其它字段整体覆盖）——直接调用节点函数拿到的是
        "部分更新" dict，节点本身不做累加，累加是 LangGraph 图执行时才发生的
        行为，plain dict.update() 会对 tool_call_history 做覆盖而非累加。
        """
        from agent.nodes import make_tool_execution_node
        from agent.retrieval_tool import register_retrieval_tool

        registry = ToolRegistry()
        register_retrieval_tool(registry, MockRetrievalPipeline())
        dispatcher = ToolDispatcherEngine(registry)
        node = make_tool_execution_node(dispatcher)

        def merge(state, update):
            state["tool_call_history"] = state.get("tool_call_history", []) + update.pop("tool_call_history", [])
            state.update(update)
            return state

        state = new_agent_state(query="test")
        state = merge(state, node(state))
        self.assertEqual(len(state["retrieval_results"]), 2)

        # 第二次调用（同一个 mock 会返回同样两条结果）——去重后总数不应翻倍
        state = merge(state, node(state))
        self.assertEqual(len(state["retrieval_results"]), 2)
        self.assertEqual(len(state["tool_call_history"]), 2)  # 调用历史应累加，不去重


# ══════════════════════════════════════════════════════════════════
# 会话 Agent 执行轨迹查询测试
# ══════════════════════════════════════════════════════════════════

class TestSessionAgentTrace(unittest.TestCase):
    def setUp(self):
        self.sm = SessionManager()
        self.session_id, _ = self.sm.create_session()

    def test_rag_mode_turn_keeps_agent_trace_none(self):
        self.sm.append_turn(self.session_id, "query", "answer")
        info = self.sm.get_session_info(self.session_id)
        self.assertIsNone(info["turns"][0].agent_trace)

    def test_agent_mode_turn_stores_trace(self):
        trace = [
            {"step": "entry", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.01, "status": "success", "error": None, "timestamp": 1.0},
            {"step": "tool_execution", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.5, "status": "success", "error": None, "timestamp": 2.0},
        ]
        self.sm.append_turn(self.session_id, "q", "a", agent_trace=trace)
        info = self.sm.get_session_info(self.session_id)
        self.assertEqual([t["step"] for t in info["turns"][0].agent_trace], ["entry", "tool_execution"])

    def test_get_agent_trace_flattens_across_turns(self):
        trace1 = [{"step": "entry", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.01, "status": "success", "error": None, "timestamp": 1.0}]
        trace2 = [{"step": "tool_execution", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.5, "status": "success", "error": None, "timestamp": 2.0}]
        self.sm.append_turn(self.session_id, "q1", "a1", agent_trace=trace1)
        self.sm.append_turn(self.session_id, "q2", "a2", agent_trace=trace2)

        all_steps = [e["step"] for e in self.sm.get_agent_trace(self.session_id)]
        self.assertEqual(all_steps, ["entry", "tool_execution"])

    def test_get_agent_trace_filters_by_step(self):
        trace = [
            {"step": "entry", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.01, "status": "success", "error": None, "timestamp": 1.0},
            {"step": "tool_execution", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.5, "status": "success", "error": None, "timestamp": 2.0},
            {"step": "tool_execution", "inputs": {}, "outputs": {}, "elapsed_seconds": 0.3, "status": "success", "error": None, "timestamp": 3.0},
        ]
        self.sm.append_turn(self.session_id, "q", "a", agent_trace=trace)
        filtered = self.sm.get_agent_trace(self.session_id, step="tool_execution")
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(e["step"] == "tool_execution" for e in filtered))

    def test_get_agent_trace_empty_for_rag_only_session(self):
        self.sm.append_turn(self.session_id, "q", "a")  # 纯 RAG 模式，无 agent_trace
        self.assertEqual(self.sm.get_agent_trace(self.session_id), [])

    def test_get_agent_trace_nonexistent_session_returns_empty(self):
        self.assertEqual(self.sm.get_agent_trace("does-not-exist"), [])


if __name__ == "__main__":
    unittest.main()
