"""
agent/tool_dispatcher.py — 工具调度引擎（ToolDispatcherEngine）
统一封装工具调用入口：参数自动填充 + 校验 -> 同步执行 -> 结果标准化解析 ->
异常重试（区分可重试/不可重试异常，指数退避，默认最大 3 次）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import ValidationError

from .state import ToolCall
from .tool_registry import ToolRegistry


class NonRetryableError(Exception):
    """参数非法、工具不存在等——重试没有意义，立即标记失败。"""


class RetryableError(Exception):
    """网络超时、模型过载、向量库连接失败等瞬时性异常——值得退避重试。"""


DEFAULT_BACKOFF_BASE_SECONDS = 0.5   # 指数退避基数：第 n 次重试等待 base * 2**(n-1) 秒

# validate_params 实际返回某个 pydantic BaseModel 子类的实例，具体类型由各
# 工具的 param_schema 决定；用 Any 别名让方法签名可读，不强行引入更复杂的泛型。
BaseModelInstance = Any


class ToolDispatcherEngine:
    """
    工具调度引擎：结合 ToolRegistry 完成"参数自动填充 -> 校验 -> 执行 -> 重试"
    全流程，返回标准化的 ToolCall 记录（success/result/error 三段式），
    单个工具调用失败不会向上抛出异常中断整个 Agent 循环。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        log: logging.Logger | None = None,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    ):
        self.registry = registry
        self.log = log or logging.getLogger("agent.tool_dispatcher")
        self.backoff_base_seconds = backoff_base_seconds

    # ── 参数自动填充 ─────────────────────────────────────────────
    def auto_fill_params(self, tool_name: str, state: dict) -> dict:
        """
        结合当前 AgentState 与工具的 param_schema，自动生成调用参数：
        按 schema 字段名从 state 里直接取值（字段名一致即可自动对上，
        如 query/top_k/fusion_strategy）。

        预留扩展位：未来复杂任务规划场景下，可替换为调用 LLM 根据
        state.query + 工具描述动态生成参数，见 _llm_generate_params。
        """
        spec = self.registry.get(tool_name)
        if spec is None:
            raise NonRetryableError(f"工具 {tool_name!r} 未注册")

        args = {}
        for field_name in spec.param_schema.model_fields:
            if field_name in state and state[field_name] is not None:
                args[field_name] = state[field_name]
        return args

    def _llm_generate_params(self, tool_name: str, state: dict) -> dict:
        """预留扩展位：本周未实现，明确报错而非静默返回空结果，避免调用方误以为已支持。"""
        raise NotImplementedError("LLM 动态参数生成是预留扩展位，本周未实现")

    # ── 参数校验 ─────────────────────────────────────────────────
    def validate_params(self, tool_name: str, arguments: dict) -> BaseModelInstance:
        """
        用工具注册的 pydantic 模型校验参数。校验失败视为不可重试异常——
        参数不合法不会因为重试而变合法。
        """
        spec = self.registry.get(tool_name)
        if spec is None:
            raise NonRetryableError(f"工具 {tool_name!r} 未注册")
        try:
            return spec.param_schema(**arguments)
        except ValidationError as e:
            raise NonRetryableError(f"工具 {tool_name!r} 参数校验失败：{e}") from e

    # ── 调用执行（含重试）────────────────────────────────────────
    def dispatch(self, tool_name: str, arguments: dict, subtask_id: str | None = None) -> ToolCall:
        """
        校验参数 -> 执行 -> 标准化结果。异常按可重试/不可重试分流：
          - NonRetryableError（参数非法/工具不存在）：立即标记失败，不重试
          - RetryableError（网络超时/模型过载/向量库连接失败等）：
            指数退避重试，达到工具注册时配置的 max_retries 后标记失败
          - 其它未预期异常：视为不可重试（对未知错误做无意义重试没有意义，
            且可能掩盖真正的 bug）
        """
        call = ToolCall(tool_name=tool_name, arguments=arguments, subtask_id=subtask_id)
        spec = self.registry.get(tool_name)

        if spec is None:
            call.success = False
            call.error = f"工具 {tool_name!r} 未注册"
            call.retryable = False
            call.elapsed_seconds = 0.0
            self.log.error(call.error)
            return call

        try:
            validated = self.validate_params(tool_name, arguments)
        except NonRetryableError as e:
            call.success = False
            call.error = str(e)
            call.retryable = False
            call.elapsed_seconds = 0.0
            self.log.error(f"[{tool_name}] 参数校验失败，不重试：{e}")
            return call

        max_retries = spec.max_retries
        t_start = time.time()
        last_error: Exception | None = None
        attempt = 0

        for attempt in range(max_retries + 1):
            try:
                result = spec.handler(**validated.model_dump())
                call.result = result
                call.success = True
                call.retry_count = attempt
                call.elapsed_seconds = round(time.time() - t_start, 4)
                if attempt > 0:
                    self.log.info(f"[{tool_name}] 第 {attempt} 次重试后成功")
                return call
            except RetryableError as e:
                last_error = e
                call.retryable = True
                if attempt < max_retries:
                    wait = self.backoff_base_seconds * (2 ** attempt)
                    self.log.warning(
                        f"[{tool_name}] 可重试异常（第 {attempt + 1}/{max_retries} 次）："
                        f"{e}，{wait:.2f}s 后重试"
                    )
                    time.sleep(wait)
            except Exception as e:
                last_error = e
                call.retryable = False
                self.log.error(f"[{tool_name}] 不可重试异常：{e}")
                break

        call.success = False
        call.error = str(last_error) if last_error else "未知错误"
        call.retry_count = attempt
        call.elapsed_seconds = round(time.time() - t_start, 4)
        return call
