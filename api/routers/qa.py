"""
qa.py — 问答接口
POST /api/v1/qa            同步问答
POST /api/v1/qa/stream     流式问答（Server-Sent Events）
GET  /api/v1/qa/sessions/{session_id}   查看会话历史（调试/前端展示用）
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from generation import MedicalGenerationPipeline

from ..dependencies import get_generation_pipeline, get_session_manager
from ..exceptions import ModelCallError, SessionNotFoundError
from ..models import (
    QARequest,
    QAResponseData,
    ResponseModel,
    SessionHistoryData,
    SessionHistoryTurn,
    SourceItem,
)
from ..session import SessionManager

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])
log = logging.getLogger("api.qa")


def _build_response_data(result: dict, session_id: str | None) -> QAResponseData:
    return QAResponseData(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        session_id=session_id,
        total_time_seconds=result["generation_metrics"]["total_time_seconds"],
        citation_retry_attempts=result["generation_metrics"]["citation_retry_attempts"],
        format_check_pass=result["format_check"]["overall_pass"],
    )


# ── 同步问答 ──────────────────────────────────────────────────────
@router.post("", response_model=ResponseModel[QAResponseData])
async def ask(
    req: QARequest,
    request: Request,
    pipeline: MedicalGenerationPipeline = Depends(get_generation_pipeline),
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[QAResponseData]:
    request_id = getattr(request.state, "request_id", None)

    session_id = req.session_id
    conversation_context = ""
    if session_id:
        if not sessions.exists(session_id):
            # 会话不存在/已过期时不报错——按新会话处理，同时把传入的 session_id 继续使用
            log.info(f"[{request_id}] session_id={session_id} 不存在或已过期，作为新会话继续")
        conversation_context = sessions.build_context_prefix(session_id)

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
        log.error(f"[{request_id}] 问答生成失败: {e}", exc_info=e)
        raise ModelCallError(f"生成失败: {e}") from e

    if session_id:
        sessions.append_turn(session_id, req.query, result["answer"])

    data = _build_response_data(result, session_id)
    return ResponseModel.ok(data=data, request_id=request_id)


# ── 流式问答（SSE）────────────────────────────────────────────────
@router.post("/stream")
async def ask_stream(
    req: QARequest,
    request: Request,
    pipeline: MedicalGenerationPipeline = Depends(get_generation_pipeline),
    sessions: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", None)

    session_id = req.session_id
    conversation_context = ""
    if session_id:
        conversation_context = sessions.build_context_prefix(session_id)

    def event_stream():
        final_answer = ""
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
                payload = {
                    "event": "done",
                    "request_id": request_id,
                    "data": _build_response_data(event["result"], session_id).model_dump(),
                }
            else:
                payload = {"event": event["event"], "request_id": request_id, **{
                    k: v for k, v in event.items() if k != "event"
                }}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        if session_id and final_answer:
            sessions.append_turn(session_id, req.query, final_answer)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── 会话历史查询（调试/前端展示）──────────────────────────────────
@router.get("/sessions/{session_id}", response_model=ResponseModel[SessionHistoryData])
async def get_session_history(
    session_id: str,
    request: Request,
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[SessionHistoryData]:
    history = sessions.get_history(session_id)
    if not history:
        raise SessionNotFoundError(f"会话 {session_id} 不存在或已过期")
    data = SessionHistoryData(
        session_id=session_id,
        turns=[SessionHistoryTurn(query=t.query, answer=t.answer, timestamp=t.timestamp) for t in history],
    )
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))
