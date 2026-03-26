"""
app/routers/ask.py — QA endpoints with hybrid search, caching, streaming, and history.
"""

import logging
import time
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import CurrentUser, get_db
from app.models.schemas import (
    QueryHistoryItem,
    QueryHistoryResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.services import cache_service
from app.services.db_service import log_query
from app.services.rag_service import async_query
from app.utils.exceptions import VectorStoreNotReadyError
from app.services.rag_service import get_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ask", tags=["QA"])


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
    description=(
        "Submit a question and receive a grounded answer with source documents.\n\n"
        "**Features:**\n"
        "- Hybrid BM25 + vector retrieval\n"
        "- Cross-encoder re-ranking\n"
        "- Redis response caching\n"
        "- Optional `metadata_filter` for scoped search\n"
        "- Optional `stream=true` for Server-Sent Events\n"
    ),
)
async def ask_post(
    request: QueryRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> QueryResponse | StreamingResponse:
    if not get_rag_pipeline().is_ready():
        raise VectorStoreNotReadyError()

    if request.stream:
        return StreamingResponse(
            _stream_answer(request, db, current_user),
            media_type="text/event-stream",
        )

    return await _handle_query(request, db)


# ── GET /ask ──────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=QueryResponse,
    summary="Query via GET parameter (convenience)",
)
async def ask_get(
    query: str = Query(..., min_length=1, max_length=2000),
    top_k: int = Query(4, ge=1, le=20),
    current_user: str = Depends(lambda: None),  # auth optional on GET
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    if not get_rag_pipeline().is_ready():
        raise VectorStoreNotReadyError()
    return await _handle_query(QueryRequest(query=query, top_k=top_k), db)


# ── GET /ask/history ──────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=QueryHistoryResponse,
    summary="Retrieve query history (paginated)",
)
async def query_history(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> QueryHistoryResponse:
    from sqlalchemy import select, func
    from app.models.db_models import QueryLog

    total_q = await db.execute(select(func.count()).select_from(QueryLog))
    total = total_q.scalar_one()

    items_q = await db.execute(
        select(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [QueryHistoryItem.model_validate(row) for row in items_q.scalars()]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return QueryHistoryResponse(
        items=items, count=len(items), page=page,
        page_size=page_size, total_pages=total_pages,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _handle_query(request: QueryRequest, db: AsyncSession) -> QueryResponse:
    t0 = time.perf_counter()

    # Cache lookup
    cached = await cache_service.get_cached(
        request.query, request.top_k, request.metadata_filter
    )
    if cached:
        logger.info("Cache HIT | query=%r", request.query[:50])
        return QueryResponse(**cached, cached=True)

    # RAG query
    answer, sources = await async_query(request.query, request.top_k, request.metadata_filter)
    latency_ms = (time.perf_counter() - t0) * 1000

    source_docs = [SourceDocument(**s) for s in sources]
    response = QueryResponse(
        query=request.query,
        answer=answer,
        source_documents=source_docs,
        cached=False,
        latency_ms=round(latency_ms, 1),
    )

    # Persist to DB
    try:
        await log_query(
            db,
            request_id=None,
            query=request.query,
            answer=answer,
            source_count=len(sources),
            latency_ms=latency_ms,
            cached=False,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.warning("Failed to log query to DB: %s", exc)

    # Cache result
    await cache_service.set_cached(
        request.query, response.model_dump(), request.top_k, request.metadata_filter
    )

    logger.info(
        "Query answered | query=%r | sources=%d | latency=%.0fms",
        request.query[:50], len(sources), latency_ms,
    )
    return response


async def _stream_answer(
    request: QueryRequest, db: AsyncSession, user: str
) -> AsyncGenerator[str, None]:
    """Server-Sent Events stream for progressive answer delivery."""
    answer, sources = await async_query(request.query, request.top_k, request.metadata_filter)
    # Stream answer word by word
    for word in answer.split():
        yield f"data: {word} \n\n"
    yield f"data: [DONE]\n\n"
