"""
agent/tool_registry.py — 工具注册表（ToolRegistry）
统一维护工具描述与参数 Schema，支持动态注册/注销。兼容 LangChain Tool
规范——已注册工具可导出为 langchain_core.tools.StructuredTool，供未来接入
LangGraph 预置的 ToolNode 或其它 LangChain 生态组件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """工具注册表里的一条记录。"""

    name: str
    handler: Callable[..., Any]      # 实际执行函数；关键字参数须与 param_schema 字段名一致
    param_schema: type[BaseModel]    # pydantic 模型：定义 + 校验入参
    description: str = ""
    max_retries: int = 3


class ToolRegistry:
    """
    工具注册表：支持动态注册/注销检索、过滤、评估、合规校验等工具，
    统一维护工具描述与参数 Schema（本周只接入检索工具，其余在设计文档中
    预留，接入方式完全一致）。
    """

    def __init__(self, log: logging.Logger | None = None):
        self.log = log or logging.getLogger("agent.tool_registry")
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            self.log.warning(f"工具 {spec.name!r} 已存在，将被覆盖注册")
        self._tools[spec.name] = spec
        self.log.info(f"工具已注册：{spec.name}")

    def unregister(self, name: str) -> bool:
        existed = name in self._tools
        self._tools.pop(name, None)
        if existed:
            self.log.info(f"工具已注销：{name}")
        return existed

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe_all(self) -> list[dict]:
        """返回所有已注册工具的描述 + 参数 JSON Schema，供健康检查/调试展示。"""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "param_schema": spec.param_schema.model_json_schema(),
                "max_retries": spec.max_retries,
            }
            for spec in self._tools.values()
        ]

    # ── LangChain Tool 规范兼容 ──────────────────────────────────
    def as_langchain_tool(self, name: str):
        """
        把已注册工具导出为 langchain_core.tools.StructuredTool。
        延迟导入：只有真正调用这个方法时才需要 langchain_core 的
        tools 子模块，避免给不需要 LangChain 生态互操作的调用方增加负担。
        """
        from langchain_core.tools import StructuredTool

        spec = self.get(name)
        if spec is None:
            raise KeyError(f"工具 {name!r} 未注册")

        return StructuredTool.from_function(
            func=spec.handler,
            name=spec.name,
            description=spec.description,
            args_schema=spec.param_schema,
        )
