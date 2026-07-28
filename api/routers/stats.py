"""
stats.py (router) — 运营统计接口
GET /api/v1/stats  问答调用次数/平均耗时/成功率、语料规模、各组件健康状态
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_document_index, get_health_checker, get_session_manager, get_stats_tracker
from ..document_index import DocumentIndex
from ..health_checker import HealthChecker
from ..models import ComponentHealthItem, CorpusStatsData, OperationalStatsData, QAStatsData, ResponseModel
from ..session import SessionManager
from ..stats import StatsTracker

router = APIRouter(prefix="/api/v1/stats", tags=["stats"])


@router.get("", response_model=ResponseModel[OperationalStatsData])
async def get_stats(
    request: Request,
    tracker: StatsTracker = Depends(get_stats_tracker),
    health_checker: HealthChecker = Depends(get_health_checker),
    doc_index: DocumentIndex = Depends(get_document_index),
    sessions: SessionManager = Depends(get_session_manager),
) -> ResponseModel[OperationalStatsData]:
    qa_snapshot = tracker.snapshot()
    components = health_checker.check_all()

    data = OperationalStatsData(
        qa=QAStatsData(**qa_snapshot),
        corpus=CorpusStatsData(
            total_documents=len(doc_index),
            total_chunks=doc_index.total_chunk_count(),
            index_size_bytes=None,
            incremental_update_count=doc_index.incremental_update_count,
        ),
        components=[ComponentHealthItem(**c) for c in components],
        active_sessions=sessions.active_session_count(),
    )
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))
