"""
session.py — 轻量会话管理（SessionManager）
按 session_id 存储最近若干轮对话（query+answer），支持 TTL 过期与线程安全并发访问。

当前实现为进程内存储（服务重启即丢失）。若需要跨进程/多实例共享或持久化，
可替换为 Redis/SQLite 后端，对外接口（create_session/get_history/append_turn/
delete_session/build_context_prefix）保持不变，调用方无需改动。
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
        self._created_at: dict[str, float] = {}
        self._last_active: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def _is_expired(self, session_id: str) -> bool:
        last = self._last_active.get(session_id)
        return last is None or (time.time() - last) > self.ttl_seconds

    def _purge(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._created_at.pop(session_id, None)
        self._last_active.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions and not self._is_expired(session_id)

    # ── 创建 ──────────────────────────────────────────────────────
    def create_session(self) -> tuple[str, float]:
        """显式创建一个新会话（对应"创建会话"接口），返回 (session_id, created_at)。"""
        session_id = self.new_session_id()
        now = time.time()
        with self._lock:
            self._sessions[session_id] = []
            self._created_at[session_id] = now
            self._last_active[session_id] = now
        return session_id, now

    # ── 查询 ──────────────────────────────────────────────────────
    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """返回该会话的历史轮次；不存在或已过期时返回空列表并清理残留数据。"""
        with self._lock:
            if session_id not in self._sessions or self._is_expired(session_id):
                self._purge(session_id)
                return []
            return list(self._sessions[session_id])

    def get_session_info(self, session_id: str) -> dict | None:
        """返回会话完整信息（创建时间/最近活跃时间/历史轮次）；不存在或已过期返回 None。"""
        history = self.get_history(session_id)  # 内部已处理过期清理
        with self._lock:
            if session_id not in self._sessions:
                return None
            return {
                "session_id": session_id,
                "created_at": self._created_at.get(session_id),
                "last_active": self._last_active.get(session_id),
                "turn_count": len(history),
                "turns": history,
            }

    # ── 写入（问答接口自动调用，也支持外部显式调用）──────────────
    def append_turn(self, session_id: str, query: str, answer: str) -> None:
        now = time.time()
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            self._created_at.setdefault(session_id, now)
            turns.append(ConversationTurn(query=query, answer=answer))
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]
            self._last_active[session_id] = now

    # ── 删除 ──────────────────────────────────────────────────────
    def delete_session(self, session_id: str) -> bool:
        """删除会话，返回删除前是否存在（供接口层判断返回 200 还是 404）。"""
        with self._lock:
            existed = session_id in self._sessions
            self._purge(session_id)
            return existed

    # ── 统计（供运营统计接口使用）──────────────────────────────────
    def active_session_count(self) -> int:
        with self._lock:
            return sum(1 for sid in self._sessions if not self._is_expired(sid))

    # ── 对话上下文渲染 ────────────────────────────────────────────
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
