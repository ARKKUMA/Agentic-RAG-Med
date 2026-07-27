"""
session.py — 轻量会话管理（SessionManager）
按 session_id 存储最近若干轮对话（query+answer），支持 TTL 过期与线程安全并发访问。

当前实现为进程内存储（服务重启即丢失）。若需要跨进程/多实例共享或持久化，
可替换为 Redis/SQLite 后端，对外接口（get_history/append_turn/build_context_prefix）
保持不变，调用方无需改动。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

DEFAULT_SESSION_TTL_SECONDS = 3600   # 会话 1 小时无新请求则视为过期
DEFAULT_MAX_TURNS_PER_SESSION = 20   # 单会话最多保留的对话轮数，超出淘汰最旧的
DEFAULT_CONTEXT_TURNS = 3            # 拼接进 prompt 的历史轮数上限


@dataclass
class ConversationTurn:
    query: str
    answer: str
    timestamp: float = field(default_factory=time.time)


class SessionManager:
    def __init__(
        self,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        max_turns: int = DEFAULT_MAX_TURNS_PER_SESSION,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: dict[str, list[ConversationTurn]] = {}
        self._last_active: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def _is_expired(self, session_id: str) -> bool:
        last = self._last_active.get(session_id)
        return last is None or (time.time() - last) > self.ttl_seconds

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions and not self._is_expired(session_id)

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """返回该会话的历史轮次；不存在或已过期时返回空列表并清理残留数据。"""
        with self._lock:
            if session_id not in self._sessions or self._is_expired(session_id):
                self._sessions.pop(session_id, None)
                self._last_active.pop(session_id, None)
                return []
            return list(self._sessions[session_id])

    def append_turn(self, session_id: str, query: str, answer: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(ConversationTurn(query=query, answer=answer))
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            self._last_active[session_id] = time.time()

    def build_context_prefix(self, session_id: str, max_turns: int = DEFAULT_CONTEXT_TURNS) -> str:
        """
        把最近 max_turns 轮对话渲染成简短前缀文本，供多轮追问时作为额外上下文
        拼接进生成阶段的 prompt（不用于检索，避免历史文本稀释向量检索的语义焦点）。
        """
        history = self.get_history(session_id)[-max_turns:]
        if not history:
            return ""
        lines = ["(Previous conversation turns, for context only:)"]
        for turn in history:
            lines.append(f"Q: {turn.query}\nA: {turn.answer[:300]}")
        return "\n".join(lines) + "\n\n"
