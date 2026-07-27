"""
middleware.py — 请求日志中间件
为每个请求生成 request_id，记录路径/方法/状态码/耗时，写入 JSONL 日志，
供后续统计分析。业务异常已由 exceptions.py 的全局处理器转换为标准 JSON 响应，
因此这里无需再 try/except —— call_next() 返回的就是最终响应（含错误响应）。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_path: str | Path | None = None, logger: logging.Logger | None = None):
        super().__init__(app)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("api.requests")

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        t0 = time.time()

        response = await call_next(request)

        elapsed = round(time.time() - t0, 3)
        record = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.logger.info(json.dumps(record, ensure_ascii=False))
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Elapsed-Seconds"] = str(elapsed)
        return response
