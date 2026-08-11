"""
qa.py — 问答接口
POST /api/v1/qa            同步问答（统一入口：agent_mode 参数路由到 RAG / Agent 两条链路）
POST /api/v1/qa/stream     流式问答（Server-Sent Events；仍是纯 RAG 流水线，Agent 链路
                            本周未接入流式——LangGraph 图本身的流式执行是更大的改动，
                            明确列为未完成项，见 reports/AGENT_WEEK2_REPORT.md）

会话历史查询/创建/删除见 sessions.py；这里只负责在生成完成后自动调用
SessionManager.append_turn()"添加消息"，以及向 StatsTracker 上报每次调用的
耗时与成功/失败状态（供 GET /api/v1/stats 使用）。

统一入口层（第 2 周新增）：
  agent_mode=false（默认）：完全直通原有单轮 RAG 流水线，请求/响应结构与升级前
    100% 一致——这是"确保原有功能无退化"在接口层的落点。
  agent_mode=true：改为调用 agent.build_agent_graph() 编译好的 LangGraph 状态机，
    响应外层结构不变，只多出 execution_trace 字段（其余原有字段里，agent 模式本周
    未执行的校验阶段用 None 如实标注"未评估"，不伪造 True，见 models.QAResponseData）。
  两条链路共用同一个 SessionManager 实例，session_id 语义完全一致，可以在同一个
  会话里自由切换 agent_mode（切换不会破坏历史对话，因为两条链路读写的是同一份
  ConversationTurn 结构，agent_trace 字段仅 Agent 模式的轮次会填充）。
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from agent import new_agent_state
from generation import MedicalGenerationPipeline

from ..dependencies import get_agent_graph, get_generation_pipeline, get_session_manager, get_stats_tracker
from ..exceptions import ModelCallError
from ..models import QARequest, QAResponseData, ResponseModel, SourceItem
from ..session import SessionManager
from ..stats import StatsTracker

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])
log = logging.getLogger("api.qa")


def _build_rag_response_data(result: dict, session_id: str | None) -> QAResponseData:
    return QAResponseData(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        session_id=session_id,
        total_time_seconds=result["generation_metrics"]["total_time_seconds"],
        citation_retry_attempts=result["generation_metrics"]["citation_retry_attempts"],
        format_check_pass=result["format_check"]["overall_pass"],
        agent_mode=False,
        execution_trace=None,
    )


def _build_agent_response_data(agent_result: dict, session_id: str | None, elapsed_seconds: float) -> QAResponseData:
    return QAResponseData(
        answer=agent_result.get("final_answer") or "",
        sources=[SourceItem(**s) for s in agent_result.get("sources", [])],
        session_id=session_id,
        total_time_seconds=round(elapsed_seconds, 2),
        citation_retry_attempts=0,   # 本周 Agent 链路没有引用重试循环（尚未接入 generation 的证据评估/审查阶段）
        format_check_pass=None,      # 未执行格式校验——如实标注"未评估"，不伪造 True
        agent_mode=True,
        execution_trace=agent_result.get("execution_trace", []),
    )


# ── 同步问答（统一入口）───────────────────────────────────────────
@router.post("", response_model=ResponseModel[QAResponseData])
async def ask(
    req: QARequest,
    request: Request,
    pipeline: MedicalGenerationPipeline = Depends(get_generation_pipeline),
    sessions: SessionManager = Depends(get_session_manager),
    stats: StatsTracker = Depends(get_stats_tracker),
    agent_graph: CompiledStateGraph = Depends(get_agent_graph),
) -> ResponseModel[QAResponseData]:
    request_id = getattr(request.state, "request_id", None)

    session_id = req.session_id
    if session_id and not sessions.exists(session_id):
        # 会话不存在/已过期时不报错——按新会话处理，同时把传入的 session_id 继续使用
        log.info(f"[{request_id}] session_id={session_id} 不存在或已过期，作为新会话继续")

    t0 = time.time()

    if req.agent_mode:
        try:
            initial_state = new_agent_state(
                query=req.query,
                session_id=session_id,
                top_k=req.top_k,
                fusion_strategy=req.fusion_strategy,
                where_filter=req.where_filter,
            )
            agent_result = agent_graph.invoke(initial_state, config={"recursion_limit": 25})
        except Exception as e:
            stats.record(elapsed_seconds=time.time() - t0, success=False)
            log.error(f"[{request_id}] Agent 问答失败: {e}", exc_info=e)
            raise ModelCallError(f"Agent 执行失败: {e}") from e

        elapsed = time.time() - t0
        success = agent_result.get("execution_status") is not None and agent_result["execution_status"].value != "failed"
        stats.record(elapsed_seconds=elapsed, success=success)

        if session_id:
            sessions.append_turn(
                session_id, req.query, agent_result.get("final_answer") or "",
                agent_trace=agent_result.get("execution_trace"),
            )

        data = _build_agent_response_data(agent_result, session_id, elapsed)
        return ResponseModel.ok(data=data, request_id=request_id)

    # ── 原有单轮 RAG 直通路径（agent_mode=false，行为与升级前完全一致）──
    conversation_context = sessions.build_context_prefix(session_id) if session_id else ""
    try:
        result = pipeline.generate(
            req.query,
            top_k=req.top_k,
            fusion_strategy=req.fusion_strategy,
            run_evaluation=req.run_evaluation,
            run_review=req.run_review,
            conversation_context=conversation_context,
        )
    except Exception as e:
        stats.record(elapsed_seconds=time.time() - t0, success=False)
        log.error(f"[{request_id}] 问答生成失败: {e}", exc_info=e)
        raise ModelCallError(f"生成失败: {e}") from e

    stats.record(elapsed_seconds=time.time() - t0, success=True)

    if session_id:
        sessions.append_turn(session_id, req.query, result["answer"])

    data = _build_rag_response_data(result, session_id)
    return ResponseModel.ok(data=data, request_id=request_id)


# ── 流式问答（SSE）────────────────────────────────────────────────
@router.post("/stream")
async def ask_stream(
    req: QARequest,
    request: Request,
    pipeline: MedicalGenerationPipeline = Depends(get_generation_pipeline),
    sessions: SessionManager = Depends(get_session_manager),
    stats: StatsTracker = Depends(get_stats_tracker),
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", None)

    session_id = req.session_id
    conversation_context = ""
    if session_id:
        conversation_context = sessions.build_context_prefix(session_id)

    def event_stream():
        final_answer = ""
        t0 = time.time()
        success = False
        try:
            for event in pipeline.generate_stream(
                req.query,
                top_k=req.top_k,
                fusion_strategy=req.fusion_strategy,
                run_evaluation=req.run_evaluation,
                run_review=req.run_review,
                conversation_context=conversation_context,
            ):
                if event["event"] == "done":
                    final_answer = event["result"]["answer"]
                    success = True
                    payload = {
                        "event": "done",
                        "request_id": request_id,
                        "data": _build_rag_response_data(event["result"], session_id).model_dump(),
                    }
                elif event["event"] == "error":
                    payload = {"event": "error", "request_id": request_id, "message": event["message"]}
                else:
                    payload = {"event": event["event"], "request_id": request_id, **{
                        k: v for k, v in event.items() if k != "event"
                    }}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            stats.record(elapsed_seconds=time.time() - t0, success=success)
            if session_id and final_answer:
                sessions.append_turn(session_id, req.query, final_answer)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
