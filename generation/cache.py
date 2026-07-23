"""
cache.py — 生成结果缓存（GenerationCache）

缓存键设计：hash(model + system_prompt + prompt + temperature + json模式)。
prompt 由 PromptStage.user_prompt_template.format(query=.., context=.., ...) 渲染而来，
已经把 query/context 内联在其中，因此这本质上就是"查询+上下文的哈希"；额外纳入
model/system_prompt/temperature 是为了避免不同阶段、不同模型、不同确定性要求之间的缓存串扰。

设计要点：
  - LRU 淘汰：max_size 限制内存占用，防止无界增长导致 OOM
  - TTL 过期：医学知识有时效性，不能永久缓存（默认 24 小时）
  - 温度门控：只缓存低温（确定性）生成结果；高温（创造性）结果每次都应重新生成，
    读/写缓存都会跳过
  - 线程安全：内部加锁，配合 BatchGenerationProcessor 的并发调用
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_SIZE = 500
DEFAULT_TTL_SECONDS = 24 * 3600   # 24 小时——医学文献/检索结果有时效性，不宜永久缓存
DEFAULT_MAX_TEMPERATURE = 0.2     # 高于此温度的生成结果不缓存（视为非确定性）


@dataclass
class _CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)


class GenerationCache:
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_temperature: float = DEFAULT_MAX_TEMPERATURE,
        log: logging.Logger | None = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_temperature = max_temperature
        self.log = log or logging.getLogger("generation_cache")

        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ── 键生成 ────────────────────────────────────────────────────
    @staticmethod
    def make_key(**parts: str) -> str:
        """
        对任意数量的具名字符串片段规范化拼接后取 SHA-256。
        相同输入 -> 相同 key，是"相同查询+上下文 -> 命中同一缓存"的基础。
        """
        canonical = "\x1f".join(f"{k}={v}" for k, v in sorted(parts.items()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_cacheable_temperature(self, temperature: float) -> bool:
        return temperature <= self.max_temperature

    # ── 读取 ──────────────────────────────────────────────────────
    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)  # LRU：命中后移到最近使用端
            self.hits += 1
            return entry.value

    # ── 写入 ──────────────────────────────────────────────────────
    def set(self, key: str, value: Any, temperature: float = 0.0) -> None:
        if not self.is_cacheable_temperature(temperature):
            return
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value=value)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # 淘汰最久未使用的条目

    # ── 统计 / 清空 ───────────────────────────────────────────────
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
