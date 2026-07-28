"""
documents.py — 文档管理接口（只读）
GET /api/v1/documents          文档列表查询（分页）
GET /api/v1/documents/{doc_id} 按 ID 查询单篇文档
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Query, Request

from ..dependencies import get_document_index
from ..document_index import DocumentIndex
from ..exceptions import DocNotFoundError
from ..models import DocumentOut, PageInfo, PaginatedData, ResponseModel

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("", response_model=ResponseModel[PaginatedData[DocumentOut]])
async def list_documents(
    request: Request,
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量，1-100"),
    doc_index: DocumentIndex = Depends(get_document_index),
) -> ResponseModel[PaginatedData[DocumentOut]]:
    items, total = doc_index.list_documents(page, page_size)
    total_pages = math.ceil(total / page_size) if total else 0
    data = PaginatedData[DocumentOut](
        items=[DocumentOut(**item) for item in items],
        page_info=PageInfo(page=page, page_size=page_size, total=total, total_pages=total_pages),
    )
    return ResponseModel.ok(data=data, request_id=getattr(request.state, "request_id", None))


@router.get("/{doc_id}", response_model=ResponseModel[DocumentOut])
async def get_document(
    doc_id: str,
    request: Request,
    doc_index: DocumentIndex = Depends(get_document_index),
) -> ResponseModel[DocumentOut]:
    doc = doc_index.get(doc_id)
    if doc is None:
        raise DocNotFoundError(f"文档 {doc_id} 不存在")
    return ResponseModel.ok(data=DocumentOut(**doc), request_id=getattr(request.state, "request_id", None))
