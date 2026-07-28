"""
stats.py — 运营统计（StatsTracker）
线程安全的问答调用计数器：调用次数、平均耗时、成功率。
每次 QA 调用（同步或流式）无论成功失败都应调用 record()，供 /api/v1/stats 使用。
"""

from __future__ import annotations

import threading


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_calls = 0
        self.success_count = 0
        self.failure_count = 0
        self._total_latency = 0.0

    def record(self, elapsed_seconds: float, success: bool) -> None:
        with self._lock:
            self.total_calls += 1
            self._total_latency += elapsed_seconds
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            total = self.total_calls
            avg_latency = round(self._total_latency / total, 4) if total else 0.0
            success_rate = round(self.success_count / total, 4) if total else 0.0
            return {
                "total_calls": total,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": success_rate,
                "avg_latency_seconds": avg_latency,
            }
