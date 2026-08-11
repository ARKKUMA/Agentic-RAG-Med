"""
agent/memory.py — 分层记忆模块（AgentMemory），落地 interfaces.LayeredMemory

三层能力，对应任务书"上下文缓存记忆实现"的三个子项：

  1. 会话记忆层：不重新实现，直接包一层 api.session.SessionManager
     （get_session_context -> build_context_prefix）。

  2. 检索结果缓存层（Redis）：
     键 = sha256(session_id + tool_name + 规范化参数)，规范化参数里显式包含
     "查询文本 + 过滤条件 + 检索参数（top_k/fusion_strategy 等）"三部分——
     字面对应任务书"基于查询文本哈希 + 过滤条件哈希 + 检索参数生成唯一键"。
     命中同一 session 内的重复调用（同一查询 + 同一过滤条件 + 同一参数）时
     直接返回缓存结果，不再真正调用工具 —— 这就是"重复检索被拦截"的落点。
     命名空间固定前缀 `agent:cache:`，与将来这台 Redis 实例上可能出现的其它
     业务用途（如未来的分布式 SessionManager 后端）隔离，见 NAMESPACE。

  3. 文献块去重存储层（Redis Sorted Set，每 session 一个 key）：
     member=chunk_id，score=相关性分数。同一 chunk 多次被检索到时只在新分数
     更高时更新分数（"合并重复文献的相关性评分"）。实现用的是"先 ZSCORE 读
     旧值、比较后再决定是否 ZADD"——刻意不用 Redis 6.2+ 的 `ZADD ... GT`
     原子标志，以便同时兼容较老的 Redis（比如临时用 tporadowski Windows 移植版
     5.0.14 做兜底时）。开发环境实际后端是跑在 WSL2 容器里的 Redis 8.10，
     本身支持 GT，若确定不再需要兼容老版本，可改回单条原子 `ZADD GT`；
     单机单进程场景下当前的非原子读-比较-写不构成实际风险（不存在并发写
     同一 session 同一 chunk）。

     设计取舍（写在这里而不是默默按字面实现）：任务书原文是"多次检索返回的
     重复文献不再重复处理、重复写入上下文"。这句话如果按"跨轮次也从当前
     回答的上下文里剔除已见过的文献"来实现，会有实际的回答质量风险——
     追问场景下（如"它的治疗方法呢"）很可能合理地需要再次引用同一篇文献，
     直接剔除会让模型答不全。因此这里区分两种"重复"：
       (a) 同一次 Agent 运行内、同一工具被调用多次（未来多跳检索周次会出现）
           产生的重复 chunk —— 这属于"同一个问题的同一次回答"，
           agent/nodes.py 里已有的按 chunk_id 去重逻辑（Week 1 就有）继续
           负责，真正从当次上下文里剔除，不受本模块影响。
       (b) 跨轮次（本轮 vs. 会话历史更早的轮次）的重复 —— 只做"识别 + 分数
           合并 + 计数上报"，不从当前轮次的检索结果里剔除，避免影响追问
           质量；重复计数写入 execution_trace，可用于观测缓存/去重是否
           真的生效。

  缓存失效：所有 key 的 TTL 与 SessionManager.ttl_seconds 对齐，每次读/写
  都 EXPIRE 刷新（滑动过期，语义与 SessionManager 用 _last_active 判断
  会话过期完全一致）；额外提供 clear_session() 支持手动清空（适配"数据
  更新场景"——语料库更新后旧的检索结果缓存应该失效）。

  容错：Redis 不可达时不让整个 Agent 崩溃——初始化时探测一次连接，探测
  失败只记 warning 并把所有缓存操作静默降级为"永远未命中/写入被忽略"，
  行为退化为"每次都真实调用工具"，功能仍然正确，只是没有缓存加速。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import Any

NAMESPACE = "agent:cache"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"   # 兼容 IPv4/IPv6：开发环境的 Redis 跑在 WSL2 容器里，只监听 IPv6 [::1]，"localhost" 会解析到它；连接超时已在下方设 2s 兜底
DEFAULT_TTL_SECONDS = 3600  # 与 api.session.DEFAULT_SESSION_TTL_SECONDS 保持一致的兜底值


def _json_default(obj: Any) -> Any:
    """json.dumps 的 default 钩子：兼容 dataclass 结果（如 ProcessedQuery）。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _canonical_params(arguments: dict) -> str:
    """把任意参数字典规范化为确定性字符串（key 排序），用于哈希。"""
    normalized = {k: v for k, v in sorted(arguments.items())}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=_json_default)


class AgentMemory:
    """
    LayeredMemory 的具体实现。构造一次，跨请求复用（与 SessionManager 同生命周期，
    典型用法是在 api/main.py 的 lifespan 里和 SessionManager 一起构建一次）。
    """

    def __init__(
        self,
        session_manager,
        redis_url: str = DEFAULT_REDIS_URL,
        redis_client=None,
        namespace: str = NAMESPACE,
        default_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        log: logging.Logger | None = None,
    ):
        self.session_manager = session_manager
        self.namespace = namespace
        self.default_ttl_seconds = default_ttl_seconds
        self.log = log or logging.getLogger("agent.memory")

        self.hits = 0
        self.misses = 0

        if redis_client is not None:
            self._redis = redis_client
        else:
            import redis as redis_lib

            # 短连接/读写超时：Redis 不可达时应快速判定并降级，而不是让调用方
            # （包括这里的健康探测本身）长时间阻塞——"系统可正常兜底"包含
            # "兜底要快"，不能让缓存后端故障拖慢整个 Agent 请求的响应时间。
            self._redis = redis_lib.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
            )

        self._redis_available = self._check_connection()

    def _check_connection(self) -> bool:
        try:
            self._redis.ping()
            self.log.info(f"Redis 缓存后端连接成功（命名空间：{self.namespace}）")
            return True
        except Exception as e:
            self.log.warning(f"Redis 缓存后端不可达，缓存功能自动降级为直通（不影响正确性）：{e}")
            return False

    # ══════════════════════════════════════════════════════════════
    # 1. 会话记忆层 —— 直接复用 SessionManager
    # ══════════════════════════════════════════════════════════════

    def get_session_context(self, session_id: str) -> str:
        return self.session_manager.build_context_prefix(session_id)

    def _ttl_for(self, session_id: str | None) -> float:
        if session_id is not None:
            return getattr(self.session_manager, "ttl_seconds", self.default_ttl_seconds)
        return self.default_ttl_seconds

    # ══════════════════════════════════════════════════════════════
    # 2. 检索/工具结果缓存层
    # ══════════════════════════════════════════════════════════════

    def _tool_result_key(self, tool_name: str, arguments: dict, session_id: str | None) -> str:
        """
        键 = hash(查询文本 + 过滤条件 + 检索参数)。arguments 本身就是工具调用的
        全部入参（对检索工具而言即 query/where_filter/top_k/fusion_strategy），
        规范化拼接后整体取一次 sha256——等价于"分别哈希三部分再拼接"，但更简单，
        碰撞概率同样可忽略（sha256 + 规范化 JSON，不存在有意义的字段边界歧义）。
        """
        canonical = f"{tool_name}\x1f{_canonical_params(arguments)}"
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        scope = f"session:{session_id}" if session_id else "global"
        return f"{self.namespace}:{scope}:tool_result:{content_hash}"

    def get_cached_tool_result(
        self, tool_name: str, arguments: dict, session_id: str | None = None,
    ) -> Any | None:
        if not self._redis_available:
            return None
        key = self._tool_result_key(tool_name, arguments, session_id)
        try:
            raw = self._redis.get(key)
        except Exception as e:
            self.log.warning(f"读取缓存失败，按未命中处理：{e}")
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            self._redis.expire(key, int(self._ttl_for(session_id)))  # 滑动过期：命中后刷新 TTL
        except Exception:
            pass
        return json.loads(raw)

    def cache_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        ttl_seconds: float | None = None,
        session_id: str | None = None,
    ) -> None:
        if not self._redis_available:
            return
        key = self._tool_result_key(tool_name, arguments, session_id)
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_for(session_id)
        try:
            payload = json.dumps(result, ensure_ascii=False, default=_json_default)
            self._redis.set(key, payload, ex=int(ttl))
        except Exception as e:
            self.log.warning(f"写入缓存失败，跳过（不影响本次调用结果）：{e}")

    # ══════════════════════════════════════════════════════════════
    # 3. 文献块去重存储层
    # ══════════════════════════════════════════════════════════════

    def _dedup_key(self, session_id: str) -> str:
        return f"{self.namespace}:session:{session_id}:seen_chunks"

    def register_chunks(self, session_id: str, chunks: list[dict]) -> dict:
        """
        把本轮检索到的 chunk 登记进该 session 的去重集合。
        对每个 chunk：按 chunk_id 用 ZADD GT 合并分数（只在新分数更高时更新）；
        返回统计信息（本轮新出现 / 之前已见过的数量），供 execution_trace 展示，
        不修改/剔除传入的 chunks 本身（见模块顶部"设计取舍"说明）。
        """
        if not self._redis_available or not chunks:
            return {"n_new": len(chunks), "n_repeat": 0, "tracked": False}

        key = self._dedup_key(session_id)
        n_new = 0
        n_repeat = 0
        try:
            existing_scores = dict(self._redis.zrange(key, 0, -1, withscores=True))
            for c in chunks:
                cid = c.get("chunk_id")
                if not cid:
                    continue
                score = float(c.get("final_score", c.get("relevance_score", 0.0)) or 0.0)
                if cid in existing_scores:
                    n_repeat += 1
                    if score <= existing_scores[cid]:
                        continue  # 新分数不比旧的高，不更新（等价于 ZADD GT 的语义）
                else:
                    n_new += 1
                self._redis.zadd(key, {cid: score})
            self._redis.expire(key, int(self._ttl_for(session_id)))
            return {"n_new": n_new, "n_repeat": n_repeat, "tracked": True}
        except Exception as e:
            self.log.warning(f"去重集合更新失败，跳过（不影响检索结果本身）：{e}")
            return {"n_new": len(chunks), "n_repeat": 0, "tracked": False}

    def get_seen_chunk_ids(self, session_id: str) -> set[str]:
        if not self._redis_available:
            return set()
        try:
            return set(self._redis.zrange(self._dedup_key(session_id), 0, -1))
        except Exception as e:
            self.log.warning(f"读取去重集合失败：{e}")
            return set()

    # ══════════════════════════════════════════════════════════════
    # 失效策略：会话过期自动清理（TTL 已保证）+ 手动清除
    # ══════════════════════════════════════════════════════════════

    def clear_session(self, session_id: str) -> int:
        """
        手动清除某个 session 名下的全部缓存（工具结果缓存 + 去重集合）。
        适配"语料库/索引更新后旧缓存应失效"的场景；自动失效已由 TTL 保证，
        这里是显式兜底入口。返回删除的 key 数量。
        """
        if not self._redis_available:
            return 0
        try:
            pattern = f"{self.namespace}:session:{session_id}:*"
            keys = list(self._redis.scan_iter(match=pattern, count=200))
            if not keys:
                return 0
            return self._redis.delete(*keys)
        except Exception as e:
            self.log.warning(f"手动清除会话缓存失败：{e}")
            return 0

    def clear_all(self) -> int:
        """清空本命名空间下的全部缓存（调试/压测用，正常运行不需要调用）。"""
        if not self._redis_available:
            return 0
        try:
            keys = list(self._redis.scan_iter(match=f"{self.namespace}:*", count=500))
            if not keys:
                return 0
            return self._redis.delete(*keys)
        except Exception as e:
            self.log.warning(f"清空缓存失败：{e}")
            return 0

    # ══════════════════════════════════════════════════════════════
    # 统计（供 execution_trace / 运营统计接口使用）
    # ══════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "redis_available": self._redis_available,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
