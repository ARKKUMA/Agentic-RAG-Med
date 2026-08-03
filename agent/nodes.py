"""
agent/nodes.py — LangGraph 节点函数
本周落地的四个基础执行节点：entry -> tool_execution -> answer_generation ->
termination，对应链路"接收问题→调用检索工具→生成答案→返回执行轨迹"。

每个节点用工厂函数构造（闭包捕获依赖，如 ToolDispatcherEngine/LLMGenerator/
会话上下文读取函数），节点函数本身保持"输入 AgentState，返回 dict 部分更新"
的纯签名，符合 LangGraph 约定，也避免用模块级全局变量传递依赖。

复用关系：
  - tool_execution 节点通过 ToolDispatcherEngine 调用已注册的 "retrieval" 工具
    （其 handler 就是对 retrieval.pipeline.RetrievalPipeline.retrieve() 的
    轻量封装，见 agent/retrieval_tool.py）。
  - answer_generation 节点复用 generation.context_assembler.ContextAssembler
    做上下文组装、复用 generation.prompt_templates 里 answer_generator 阶段
    的提示词模板 + 语言指令，调用 generation.llm_generator.LLMGenerator。
    本周只调用这一个生成阶段（产出"初始答案"），不接入证据评估/批判性审查/
    最终组装——那些是 compliance_check/reflections 预留字段对应的未来工作。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from generation.context_assembler import ContextAssembler
from generation.llm_generator import LLMGenerator
from generation.prompt_templates import LANGUAGE_DIRECTIVES, MEDICAL_PROMPT_STAGES, detect_query_language

from .state import AgentState, AgentStatus, make_trace_entry
from .tool_dispatcher import ToolDispatcherEngine

log = logging.getLogger("agent.nodes")


# ══════════════════════════════════════════════════════════════════
# 节点 1：入口节点
# ══════════════════════════════════════════════════════════════════

def make_entry_node(
    session_context_fn: Callable[[str], str] | None = None,
) -> Callable[[AgentState], dict]:
    """
    session_context_fn: 可选，接受 session_id 返回历史对话上下文前缀的函数。
    典型实现是 api.session.SessionManager.build_context_prefix，通过依赖注入
    传入而不是在这里直接 import api 模块——agent 包不应反向依赖 api 包。
    """

    def entry_node(state: AgentState) -> dict:
        t0 = time.time()
        query = (state.get("query") or "").strip()
        session_id = state.get("session_id")

        conversation_context = ""
        if session_id and session_context_fn is not None:
            try:
                conversation_context = session_context_fn(session_id)
            except Exception as e:
                log.warning(f"加载会话上下文失败（忽略，按无历史继续）：{e}")

        language = detect_query_language(query)

        trace = make_trace_entry(
            step="entry",
            inputs={"query": query, "session_id": session_id},
            outputs={"language": language, "has_conversation_context": bool(conversation_context)},
            elapsed_seconds=time.time() - t0,
        )
        return {
            "query": query,
            "language": language,
            "conversation_context": conversation_context,
            "current_step": "entry",
            "execution_status": AgentStatus.EXECUTING,
            "execution_trace": [trace],
        }

    return entry_node


# ══════════════════════════════════════════════════════════════════
# 节点 2：工具执行节点
# ══════════════════════════════════════════════════════════════════

def make_tool_execution_node(
    dispatcher: ToolDispatcherEngine,
    tool_name: str = "retrieval",
) -> Callable[[AgentState], dict]:
    def tool_execution_node(state: AgentState) -> dict:
        t0 = time.time()
        args = dispatcher.auto_fill_params(tool_name, state)
        call = dispatcher.dispatch(tool_name, args)

        new_results: list[dict] = []
        error = None
        if call.success and isinstance(call.result, dict):
            new_results = call.result.get("results", [])
        elif not call.success:
            error = call.error

        # 去重：按 chunk_id 合并"已有结果"与"本次新结果"（保留先出现的，即相关性更高的排前）
        existing = state.get("retrieval_results", [])
        seen_ids = {r.get("chunk_id") for r in existing}
        deduped = list(existing)
        for r in new_results:
            cid = r.get("chunk_id")
            if cid not in seen_ids:
                deduped.append(r)
                seen_ids.add(cid)

        trace = make_trace_entry(
            step="tool_execution",
            inputs={"tool_name": tool_name, "arguments": args},
            outputs={
                "n_new_results": len(new_results),
                "n_total_results": len(deduped),
                "success": call.success,
            },
            elapsed_seconds=time.time() - t0,
            status="success" if call.success else "failed",
            error=error,
        )
        return {
            "tool_call_history": [call],
            "retrieval_results": deduped,
            "current_step": "tool_execution",
            "error": error,
            "retry_count": state.get("retry_count", 0) + call.retry_count,
            "execution_trace": [trace],
        }

    return tool_execution_node


# ══════════════════════════════════════════════════════════════════
# 节点 3：答案生成节点
# ══════════════════════════════════════════════════════════════════

def make_answer_generation_node(
    llm: LLMGenerator,
    context_assembler: ContextAssembler | None = None,
) -> Callable[[AgentState], dict]:
    assembler = context_assembler or ContextAssembler()

    def answer_generation_node(state: AgentState) -> dict:
        t0 = time.time()
        query = state.get("query", "")
        language = state.get("language", "en")
        conversation_context = state.get("conversation_context", "")
        retrieval_results = state.get("retrieval_results", [])

        context_result = assembler.assemble(retrieval_results, query=query)
        context_text = context_result["context_text"]

        stage = MEDICAL_PROMPT_STAGES["answer_generator"]
        language_directive = LANGUAGE_DIRECTIVES.get(language, LANGUAGE_DIRECTIVES["en"])
        prompt = stage.user_prompt_template.format(
            query=query,
            context=context_text or "（未检索到相关文献）",
            language_directive=language_directive,
            retry_note="",
            conversation_context=conversation_context,
        )

        error = None
        answer = ""
        try:
            out = llm.generate(
                prompt=prompt,
                system_prompt=stage.system_prompt,
                temperature=stage.temperature,
                max_tokens=stage.max_tokens,
            )
            answer = out["text"].strip()
        except Exception as e:
            error = f"答案生成失败：{e}"
            log.error(error)

        trace = make_trace_entry(
            step="answer_generation",
            inputs={"query": query, "context_tokens": context_result["metadata"]["estimated_tokens"]},
            outputs={"answer_length": len(answer)},
            elapsed_seconds=time.time() - t0,
            status="success" if not error else "failed",
            error=error,
        )
        return {
            "intermediate_answer": answer,
            "current_step": "answer_generation",
            "error": error,
            "execution_trace": [trace],
        }

    return answer_generation_node


# ══════════════════════════════════════════════════════════════════
# 节点 4：终止节点
# ══════════════════════════════════════════════════════════════════

def make_termination_node() -> Callable[[AgentState], dict]:
    def termination_node(state: AgentState) -> dict:
        t0 = time.time()
        answer = state.get("intermediate_answer") or ""
        error = state.get("error")

        sources = [
            {
                "rank": r.get("final_rank", i),
                "chunk_id": r.get("chunk_id"),
                "journal": (r.get("metadata") or {}).get("journal"),
                "pub_year": (r.get("metadata") or {}).get("pub_year"),
                "pmc_id": (r.get("metadata") or {}).get("pmc_id"),
                "relevance_score": r.get("final_score", 0.0),
            }
            for i, r in enumerate(state.get("retrieval_results", []), start=1)
        ]

        status = AgentStatus.FAILED if (error and not answer) else AgentStatus.DONE

        trace = make_trace_entry(
            step="termination",
            inputs={"has_answer": bool(answer)},
            outputs={"final_status": status.value, "n_sources": len(sources)},
            elapsed_seconds=time.time() - t0,
        )
        return {
            "final_answer": answer,
            "sources": sources,
            "current_step": "termination",
            "execution_status": status,
            "execution_trace": [trace],
        }

    return termination_node
