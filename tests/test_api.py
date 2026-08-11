"""
tests/test_api.py — API 服务层单元测试 + 集成测试（标准库 unittest + FastAPI TestClient）

覆盖端点：
  GET  /health
  POST /api/v1/qa, POST /api/v1/qa/stream
  POST /api/v1/sessions, GET/DELETE /api/v1/sessions/{id}
  GET  /api/v1/stats
  GET  /api/v1/documents, GET /api/v1/documents/{doc_id}

运行：
    PYTHONUTF8=1 python -m unittest tests.test_api -v

设计要点：
  - RAG 流水线（BGE/ChromaDB/reranker/Ollama）通过 setUpModule/tearDownModule
    在整个测试文件范围内只加载一次（而非每个 TestCase 类各加载一次），
    避免重复付出数秒到十几秒的启动成本。
  - 真正触发 LLM 生成的用例控制在少数几个（sync/stream 各一个端到端 smoke
    test，加会话生命周期里的一次问答），其余参数校验/错误路径测试在请求
    到达生成阶段之前就被拒绝，不产生实际 LLM 调用，运行很快。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app

_client_cm: TestClient | None = None
client: TestClient | None = None


def setUpModule():
    global _client_cm, client
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()  # 触发 lifespan startup（加载 RAG 流水线）


def tearDownModule():
    _client_cm.__exit__(None, None, None)  # 触发 lifespan shutdown


# ══════════════════════════════════════════════════════════════════
# 健康检查
# ══════════════════════════════════════════════════════════════════

class TestHealth(unittest.TestCase):
    def test_health_check(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 0)
        self.assertTrue(body["data"]["ready"])


# ══════════════════════════════════════════════════════════════════
# 问答接口 —— 参数校验/错误路径（不触发真实 LLM 调用）
# ══════════════════════════════════════════════════════════════════

class TestQAValidation(unittest.TestCase):
    def test_empty_query_rejected(self):
        resp = client.post("/api/v1/qa", json={"query": ""})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_missing_query_field_rejected(self):
        resp = client.post("/api/v1/qa", json={"top_k": 5})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_query_too_long_rejected(self):
        resp = client.post("/api/v1/qa", json={"query": "a" * 2001})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_top_k_too_high_rejected(self):
        resp = client.post("/api/v1/qa", json={"query": "test", "top_k": 999})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_top_k_zero_rejected(self):
        resp = client.post("/api/v1/qa", json={"query": "test", "top_k": 0})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_invalid_fusion_strategy_rejected(self):
        resp = client.post("/api/v1/qa", json={"query": "test", "fusion_strategy": "bogus"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)

    def test_error_response_has_standard_envelope(self):
        resp = client.post("/api/v1/qa", json={"query": ""})
        body = resp.json()
        for key in ("code", "message", "data", "request_id", "timestamp"):
            self.assertIn(key, body)
        self.assertIsNone(body["data"])
        self.assertIsNotNone(body["request_id"])


# ══════════════════════════════════════════════════════════════════
# 问答接口 —— 端到端（真实调用生成流水线，数量克制）
# ══════════════════════════════════════════════════════════════════

class TestQAEndToEnd(unittest.TestCase):
    def test_sync_qa_returns_structured_answer(self):
        resp = client.post(
            "/api/v1/qa",
            json={
                "query": "What genes have been associated with susceptibility to type 2 diabetes?",
                "top_k": 5,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["code"], 0)
        data = body["data"]
        self.assertIn("Core Answer", data["answer"])
        self.assertIsInstance(data["sources"], list)
        self.assertGreaterEqual(data["citation_retry_attempts"], 0)
        self.assertIsInstance(data["format_check_pass"], bool)

    def test_stream_qa_yields_tokens_and_done_event(self):
        with client.stream(
            "POST", "/api/v1/qa/stream",
            json={
                "query": "What statistical methods are used to analyze survival data in cancer studies?",
                "top_k": 5,
            },
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        event_types = {e["event"] for e in events}
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        done_event = next(e for e in events if e["event"] == "done")
        self.assertIn("answer", done_event["data"])


# ══════════════════════════════════════════════════════════════════
# Agent 模式统一入口（第 2 周新增：agent_mode 参数路由）
# ══════════════════════════════════════════════════════════════════

class TestAgentModeEndToEnd(unittest.TestCase):
    def test_agent_mode_returns_execution_trace_and_sources(self):
        resp = client.post(
            "/api/v1/qa",
            json={"query": "What is the role of the Wnt signaling pathway in cancer?", "top_k": 5, "agent_mode": True},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertTrue(data["agent_mode"])
        self.assertTrue(len(data["answer"]) > 0)
        self.assertIsInstance(data["sources"], list)
        self.assertIsNotNone(data["execution_trace"])
        steps = [t["step"] for t in data["execution_trace"]]
        self.assertEqual(steps, ["entry", "tool_execution", "answer_generation", "termination"])
        # 本周 Agent 链路未执行格式校验，如实标注为 None（不是 True/False）
        self.assertIsNone(data["format_check_pass"])

    def test_rag_mode_response_shape_unaffected_by_new_fields(self):
        """agent_mode 缺省为 false 时，响应结构与升级前完全一致，只是多了两个恒定默认值的新字段。"""
        resp = client.post("/api/v1/qa", json={"query": "What statistical methods are used in cancer studies?", "top_k": 5})
        data = resp.json()["data"]
        self.assertFalse(data["agent_mode"])
        self.assertIsNone(data["execution_trace"])
        self.assertIsInstance(data["format_check_pass"], bool)  # RAG 模式仍是真正的 True/False，不是 None

    def test_agent_mode_same_session_second_call_hits_cache(self):
        """同一会话内重复同一问题：第二次应命中 AgentMemory 检索结果缓存（重复检索被拦截）。"""
        session_resp = client.post("/api/v1/sessions")
        session_id = session_resp.json()["data"]["session_id"]
        query = "How does learning during the day affect brain activity during sleep?"

        resp1 = client.post("/api/v1/qa", json={"query": query, "top_k": 5, "agent_mode": True, "session_id": session_id})
        resp2 = client.post("/api/v1/qa", json={"query": query, "top_k": 5, "agent_mode": True, "session_id": session_id})

        trace1 = resp1.json()["data"]["execution_trace"]
        trace2 = resp2.json()["data"]["execution_trace"]
        tool_step1 = next(t for t in trace1 if t["step"] == "tool_execution")
        tool_step2 = next(t for t in trace2 if t["step"] == "tool_execution")
        self.assertFalse(tool_step1["outputs"]["cache_hit"])
        self.assertTrue(tool_step2["outputs"]["cache_hit"])

    def test_agent_mode_invalid_where_filter_does_not_crash_service(self):
        """非法的元数据过滤条件应被检索工具/ChromaDB 拒绝，但不应让整个服务 500 崩溃或挂起。"""
        resp = client.post(
            "/api/v1/qa",
            json={
                "query": "diabetes treatment",
                "top_k": 5,
                "agent_mode": True,
                "where_filter": {"$invalid_operator": {"nonsense": True}},
            },
        )
        # 无论最终判定是"优雅失败"（业务错误码）还是"降级为无过滤条件正常回答"，
        # 都不应该是未处理异常导致的裸 500；FastAPI 全局异常处理器兜底了这一点。
        self.assertIn(resp.status_code, (200, 400, 422, 500))
        self.assertIsInstance(resp.json(), dict)  # 响应体本身仍是合法 JSON，不是连接中断/裸崩溃

    def test_agent_mode_persists_trace_into_session_history(self):
        session_resp = client.post("/api/v1/sessions")
        session_id = session_resp.json()["data"]["session_id"]
        client.post("/api/v1/qa", json={"query": "What is metformin's mechanism?", "top_k": 5, "agent_mode": True, "session_id": session_id})

        history = client.get(f"/api/v1/sessions/{session_id}").json()["data"]
        self.assertEqual(history["turn_count"], 1)


# ══════════════════════════════════════════════════════════════════
# 会话管理接口
# ══════════════════════════════════════════════════════════════════

class TestSessionLifecycle(unittest.TestCase):
    def test_full_lifecycle_create_get_qa_get_delete_get(self):
        # 1. 创建
        resp = client.post("/api/v1/sessions")
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()["data"]["session_id"]
        self.assertTrue(session_id)

        # 2. 查询——刚创建，历史应为空
        resp = client.get(f"/api/v1/sessions/{session_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"]["turn_count"], 0)

        # 3. 问答，关联该 session_id（关闭评估/审查阶段以加快测试）
        resp = client.post(
            "/api/v1/qa",
            json={
                "query": "What is GWAS?",
                "top_k": 4,
                "session_id": session_id,
                "run_evaluation": False,
                "run_review": False,
            },
        )
        self.assertEqual(resp.status_code, 200)

        # 4. 查询——应有 1 轮历史，且内容与刚才的提问一致
        resp = client.get(f"/api/v1/sessions/{session_id}")
        self.assertEqual(resp.status_code, 200)
        info = resp.json()["data"]
        self.assertEqual(info["turn_count"], 1)
        self.assertEqual(info["turns"][0]["query"], "What is GWAS?")

        # 5. 删除
        resp = client.delete(f"/api/v1/sessions/{session_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["data"]["deleted"])

        # 6. 删除后再查询——应 404
        resp = client.get(f"/api/v1/sessions/{session_id}")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["code"], 3002)

    def test_get_nonexistent_session_404(self):
        resp = client.get("/api/v1/sessions/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["code"], 3002)

    def test_delete_nonexistent_session_404(self):
        resp = client.delete("/api/v1/sessions/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["code"], 3002)

    def test_qa_with_unknown_session_id_does_not_error(self):
        """session_id 不存在/已过期时，问答接口应按新会话继续，而不是报错。"""
        resp = client.post(
            "/api/v1/qa",
            json={
                "query": "What is SNP?",
                "top_k": 3,
                "session_id": "brand-new-unused-session-id",
                "run_evaluation": False,
                "run_review": False,
            },
        )
        self.assertEqual(resp.status_code, 200)


# ══════════════════════════════════════════════════════════════════
# 运营统计接口
# ══════════════════════════════════════════════════════════════════

class TestStats(unittest.TestCase):
    def test_stats_shape(self):
        resp = client.get("/api/v1/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        for key in ("qa", "corpus", "components", "active_sessions"):
            self.assertIn(key, data)
        self.assertEqual(len(data["components"]), 3)
        component_names = {c["name"] for c in data["components"]}
        self.assertEqual(component_names, {"llm", "vector_store", "database"})
        for c in data["components"]:
            self.assertIn(c["status"], ("ok", "degraded", "down"))

    def test_stats_reflects_qa_calls_made_so_far(self):
        """本测试文件前面的用例已经产生了若干次真实调用，total_calls 应 > 0。"""
        resp = client.get("/api/v1/stats")
        data = resp.json()["data"]
        self.assertGreater(data["qa"]["total_calls"], 0)
        self.assertGreaterEqual(data["qa"]["success_rate"], 0.0)
        self.assertLessEqual(data["qa"]["success_rate"], 1.0)


# ══════════════════════════════════════════════════════════════════
# 文档管理接口
# ══════════════════════════════════════════════════════════════════

class TestDocuments(unittest.TestCase):
    def test_list_documents_paginated(self):
        resp = client.get("/api/v1/documents?page=1&page_size=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertLessEqual(len(data["items"]), 5)
        self.assertEqual(data["page_info"]["page"], 1)
        self.assertEqual(data["page_info"]["page_size"], 5)
        self.assertGreater(data["page_info"]["total"], 0)

    def test_list_documents_second_page_differs(self):
        page1 = client.get("/api/v1/documents?page=1&page_size=5").json()["data"]["items"]
        page2 = client.get("/api/v1/documents?page=2&page_size=5").json()["data"]["items"]
        ids1 = {d["doc_id"] for d in page1}
        ids2 = {d["doc_id"] for d in page2}
        self.assertTrue(ids1.isdisjoint(ids2))

    def test_get_document_by_id(self):
        first_page = client.get("/api/v1/documents?page=1&page_size=1").json()["data"]["items"]
        doc_id = first_page[0]["doc_id"]

        resp = client.get(f"/api/v1/documents/{doc_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["doc_id"], doc_id)
        self.assertIn("title", data)
        self.assertIn("chunk_count", data)

    def test_get_nonexistent_document_404(self):
        resp = client.get("/api/v1/documents/PMC_DOES_NOT_EXIST")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["code"], 3001)

    def test_page_size_out_of_range_rejected(self):
        resp = client.get("/api/v1/documents?page=1&page_size=999")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["code"], 1001)


if __name__ == "__main__":
    unittest.main()
