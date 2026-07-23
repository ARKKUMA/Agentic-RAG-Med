"""
batch_processor.py — 批量并行生成（BatchGenerationProcessor）
使用 ThreadPoolExecutor 并行处理多条查询：
  - max_workers 根据 CPU 核心数调整，避免过度竞争
  - 单个任务失败不影响其他任务（捕获异常，返回带 error 字段的占位结果）
  - 结果列表与输入 queries 顺序严格一致（按下标预分配，而非按完成顺序 append）
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 本地单实例 Ollama 模型 + 单 GPU 重排序器是共享瓶颈资源，线程数过多不会带来
# 线性加速，反而会造成显存/内存竞争，因此额外设一个硬上限
DEFAULT_MAX_WORKERS_CAP = 4


class BatchGenerationProcessor:
    """对 MedicalGenerationPipeline 的批量并行封装。"""

    def __init__(
        self,
        pipeline,
        max_workers: int | None = None,
        log: logging.Logger | None = None,
    ):
        self.pipeline = pipeline
        self.log = log or logging.getLogger("batch_processor")
        cpu_count = os.cpu_count() or 4
        self._default_max_workers = max(1, min(cpu_count // 2, DEFAULT_MAX_WORKERS_CAP))
        self._explicit_max_workers = max_workers

    def _resolve_max_workers(self, n_tasks: int) -> int:
        cap = self._explicit_max_workers or self._default_max_workers
        return max(1, min(cap, n_tasks))

    def _safe_generate(self, query: str, **kwargs) -> dict:
        """包装单条生成调用：任何异常都被捕获并转换为占位结果，绝不向上抛出。"""
        try:
            return self.pipeline.generate(query, **kwargs)
        except Exception as e:
            self.log.error(f"批量生成任务失败 query={query!r}: {e}")
            return {
                "query": query,
                "answer": None,
                "error": str(e),
                "sources": [],
                "generation_metrics": {},
                "intermediate_results": {},
                "context_metadata": {},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def run_batch(self, queries: list[str], **generate_kwargs) -> list[dict]:
        """
        并行执行多条查询，返回与 queries 顺序严格一致的结果列表。

        Args:
            queries: 查询文本列表
            **generate_kwargs: 透传给 MedicalGenerationPipeline.generate() 的参数
                （top_k / fusion_strategy / run_evaluation / run_review 等）

        Returns:
            与 queries 等长、顺序一致的结果列表；失败的任务对应位置是带
            "error" 字段的占位 dict，不影响其它任务的结果。
        """
        n = len(queries)
        if n == 0:
            return []

        max_workers = self._resolve_max_workers(n)
        self.log.info(f"批量生成开始：{n} 条查询，max_workers={max_workers}")

        results: list[dict | None] = [None] * n
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._safe_generate, q, **generate_kwargs): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # _safe_generate 内部已兜底，不会再抛异常

        elapsed = time.time() - t0
        n_errors = sum(1 for r in results if r and r.get("error"))
        self.log.info(
            f"批量生成完成：耗时={elapsed:.1f}s  成功={n - n_errors}/{n}  失败={n_errors}"
        )
        return results
