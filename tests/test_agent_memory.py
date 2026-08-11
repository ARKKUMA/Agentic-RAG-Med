"""
tests/test_agent_memory.py — AgentMemory 单元测试（连真实本机 Redis）

与 tests/test_agent.py 里的 FakeMemory 编排逻辑测试互补：这里专门覆盖
agent/memory.py 真实的 Redis 读写行为——键规则、TTL 刷新、分数合并、
手动清除、Redis 不可达时的降级。用独立的 Redis DB 15（约定俗成的测试库
编号，避免与开发用的 DB 0 数据混在一起）连接真实本机 Redis，setUp/tearDown
各清空一次，不使用 mock —— 缓存模块的价值主要就在于真实的键/TTL/分数合并
行为是否正确，mock 掉 Redis 测不出这些。

需要本机 6379 端口有 Redis 在跑（开发环境是 WSL2 里的 Redis 8.10 容器，
localhost 可达，见 reports/AGENT_WEEK2_REPORT.md §2.1）；如果连不上，测试会
在类级 skipUnless 里整体跳过而不是报错失败，避免在没有 Redis 的机器上误报
"功能坏了"。

运行：
    PYTHONUTF8=1 python -m unittest tests.test_agent_memory -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.memory import AgentMemory

TEST_REDIS_URL = "redis://localhost:6379/15"   # 开发环境 Redis 在 WSL2 容器（IPv6 [::1]），localhost 可达；见 agent/memory.py 注释


class FakeSessionManager:
    """只需要 AgentMemory 用到的两个成员：ttl_seconds + build_context_prefix。"""

    ttl_seconds = 3600

    def build_context_prefix(self, session_id: str) -> str:
        return f"(history for {session_id})\n\n"


def _redis_reachable() -> bool:
    try:
        import redis

        redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except Exception:
        return False


@unittest.skipUnless(_redis_reachable(), "本机 6379 端口没有可用的 Redis，跳过真实缓存测试")
class TestAgentMemoryRedisBacked(unittest.TestCase):
    def setUp(self):
        self.memory = AgentMemory(FakeSessionManager(), redis_url=TEST_REDIS_URL, namespace="agent:test")
        self.memory.clear_all()

    def tearDown(self):
        self.memory.clear_all()

    # ── 会话记忆层（直接复用 SessionManager）──────────────────────
    def test_get_session_context_delegates_to_session_manager(self):
        self.assertEqual(self.memory.get_session_context("s1"), "(history for s1)\n\n")

    # ── 检索结果缓存层 ────────────────────────────────────────────
    def test_cache_miss_returns_none(self):
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", {"query": "x"}, session_id="s1"))

    def test_cache_set_then_hit_returns_same_value(self):
        args = {"query": "diabetes genes", "top_k": 6, "fusion_strategy": "rrf"}
        result = {"results": [{"chunk_id": "PMC1", "final_score": 0.9}], "fused_candidates": 10}
        self.memory.cache_tool_result("retrieval", args, result, session_id="s1")
        cached = self.memory.get_cached_tool_result("retrieval", args, session_id="s1")
        self.assertEqual(cached, result)

    def test_different_arguments_produce_different_cache_entries(self):
        base_args = {"query": "diabetes genes", "top_k": 6, "fusion_strategy": "rrf"}
        self.memory.cache_tool_result("retrieval", base_args, {"results": ["a"]}, session_id="s1")

        other_top_k = {**base_args, "top_k": 8}
        other_filter = {**base_args, "where_filter": {"pub_year": {"$gte": 2020}}}
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", other_top_k, session_id="s1"))
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", other_filter, session_id="s1"))

    def test_cache_is_isolated_between_sessions(self):
        args = {"query": "same query"}
        self.memory.cache_tool_result("retrieval", args, {"results": ["a"]}, session_id="session-A")
        # 同样的查询/参数，但换一个 session —— 会话生命周期绑定，不应跨会话共享命中
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", args, session_id="session-B"))

    def test_cache_survives_query_info_dataclass_result(self):
        """检索管线真实返回值里 query_info 是 dataclass（ProcessedQuery），需要能正常序列化/反序列化。"""
        import dataclasses

        @dataclasses.dataclass
        class FakeProcessedQuery:
            original: str
            cleaned: str

        args = {"query": "x"}
        result = {"query_info": FakeProcessedQuery(original="x", cleaned="x"), "results": []}
        self.memory.cache_tool_result("retrieval", args, result, session_id="s1")
        cached = self.memory.get_cached_tool_result("retrieval", args, session_id="s1")
        self.assertEqual(cached["query_info"], {"original": "x", "cleaned": "x"})

    def test_hit_rate_stats_tracked(self):
        args = {"query": "x"}
        self.memory.get_cached_tool_result("retrieval", args, session_id="s1")  # miss
        self.memory.cache_tool_result("retrieval", args, {"r": 1}, session_id="s1")
        self.memory.get_cached_tool_result("retrieval", args, session_id="s1")  # hit
        stats = self.memory.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)

    # ── 文献块去重存储层 ──────────────────────────────────────────
    def test_register_chunks_first_time_all_new(self):
        stats = self.memory.register_chunks("s1", [
            {"chunk_id": "c1", "final_score": 0.9}, {"chunk_id": "c2", "final_score": 0.5},
        ])
        self.assertEqual(stats, {"n_new": 2, "n_repeat": 0, "tracked": True})
        self.assertEqual(self.memory.get_seen_chunk_ids("s1"), {"c1", "c2"})

    def test_register_chunks_detects_repeats(self):
        self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.5}])
        stats = self.memory.register_chunks("s1", [
            {"chunk_id": "c1", "final_score": 0.3}, {"chunk_id": "c2", "final_score": 0.7},
        ])
        self.assertEqual(stats, {"n_new": 1, "n_repeat": 1, "tracked": True})

    def test_register_chunks_merges_to_max_score(self):
        import redis

        r = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
        self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.5}])
        self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.2}])  # 更低分，不应覆盖
        self.assertEqual(r.zscore(self.memory._dedup_key("s1"), "c1"), 0.5)
        self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.9}])  # 更高分，应覆盖
        self.assertEqual(r.zscore(self.memory._dedup_key("s1"), "c1"), 0.9)

    def test_register_chunks_isolated_between_sessions(self):
        self.memory.register_chunks("session-A", [{"chunk_id": "c1", "final_score": 0.5}])
        self.assertEqual(self.memory.get_seen_chunk_ids("session-B"), set())

    def test_get_seen_chunk_ids_empty_for_unknown_session(self):
        self.assertEqual(self.memory.get_seen_chunk_ids("never-seen"), set())

    # ── 失效策略：手动清除 ───────────────────────────────────────
    def test_clear_session_removes_both_cache_and_dedup_keys(self):
        self.memory.cache_tool_result("retrieval", {"query": "x"}, {"r": 1}, session_id="s1")
        self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.5}])

        deleted = self.memory.clear_session("s1")
        self.assertGreaterEqual(deleted, 2)
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", {"query": "x"}, session_id="s1"))
        self.assertEqual(self.memory.get_seen_chunk_ids("s1"), set())

    def test_clear_session_does_not_affect_other_sessions(self):
        self.memory.cache_tool_result("retrieval", {"query": "x"}, {"r": 1}, session_id="s1")
        self.memory.cache_tool_result("retrieval", {"query": "x"}, {"r": 2}, session_id="s2")

        self.memory.clear_session("s1")
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", {"query": "x"}, session_id="s1"))
        self.assertEqual(self.memory.get_cached_tool_result("retrieval", {"query": "x"}, session_id="s2"), {"r": 2})

    # ── TTL 与会话生命周期绑定 ────────────────────────────────────
    def test_cache_entry_ttl_matches_session_ttl(self):
        import redis

        r = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
        args = {"query": "x"}
        self.memory.cache_tool_result("retrieval", args, {"r": 1}, session_id="s1")
        key = self.memory._tool_result_key("retrieval", args, "s1")
        ttl = r.ttl(key)
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, FakeSessionManager.ttl_seconds)


class TestAgentMemoryRedisUnavailable(unittest.TestCase):
    """指向一个没有 Redis 监听的端口，验证优雅降级（不抛异常，功能直通）。"""

    def setUp(self):
        self.memory = AgentMemory(
            FakeSessionManager(), redis_url="redis://127.0.0.1:1/0",
        )

    def test_reports_unavailable(self):
        self.assertFalse(self.memory._redis_available)

    def test_all_operations_degrade_silently(self):
        self.assertIsNone(self.memory.get_cached_tool_result("retrieval", {"q": 1}, session_id="s1"))
        self.memory.cache_tool_result("retrieval", {"q": 1}, {"r": 1}, session_id="s1")  # 不应抛异常
        stats = self.memory.register_chunks("s1", [{"chunk_id": "c1", "final_score": 0.5}])
        self.assertFalse(stats["tracked"])
        self.assertEqual(self.memory.get_seen_chunk_ids("s1"), set())
        self.assertEqual(self.memory.clear_session("s1"), 0)

    def test_session_context_still_works_without_redis(self):
        """会话记忆层不依赖 Redis，Redis 挂了也不该受影响。"""
        self.assertEqual(self.memory.get_session_context("s1"), "(history for s1)\n\n")


if __name__ == "__main__":
    unittest.main()
