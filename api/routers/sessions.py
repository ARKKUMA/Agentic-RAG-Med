"""
sessions.py — 会话管理接口
POST   /api/v1/sessions              创建会话
GET    /api/v1/sessions/{session_id} 获取会话信息（历史消息列表）
DELETE /api/v1/sessions/{session_id} 删除会话

"添加消息" 没有独立接口——由 POST /api/v1/qa 在生成完成后自动调用
SessionManager.append_turn()（见 qa.py），符合任务书"由问答接口自动调用"的要求。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_session_manager
from ..exceptions import SessionNotFoundError
from ..models import ResponseModel, SessionCreateData, SessionDeleteData, SessionHistoryData, SessionHistoryTurn
from ..session import SessionManager

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", response_model=ResponseModel[SessionCreateData])
async def create_session(
    request: Request,
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[SessionCreateData]:
    session_id, created_at = sessions.create_session()
    data = SessionCreateData(session_id=session_id, created_at=created_at)
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))


@router.get("/{session_id}", response_model=ResponseModel[SessionHistoryData])
async def get_session(
    session_id: str,
    request: Request,
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[SessionHistoryData]:
    info = sessions.get_session_info(session_id)
    if info is None:
        raise SessionNotFoundError(f"会话 {session_id} 不存在或已过期")

    data = SessionHistoryData(
        session_id=info["session_id"],
        created_at=info["created_at"],
        last_active=info["last_active"],
        turn_count=info["turn_count"],
        turns=[
            SessionHistoryTurn(query=t.query, answer=t.answer, timestamp=t.timestamp)
            for t in info["turns"]
        ],
    )
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))


@router.delete("/{session_id}", response_model=ResponseModel[SessionDeleteData])
async def delete_session(
    session_id: str,
    request: Request,
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[SessionDeleteData]:
    existed = sessions.delete_session(session_id)
    if not existed:
        raise SessionNotFoundError(f"会话 {session_id} 不存在或已过期")
    data = SessionDeleteData(session_id=session_id, deleted=True)
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))
