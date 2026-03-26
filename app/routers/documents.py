"""
app/routers/documents.py — Document upload and listing endpoints.
"""

import logging
import os
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import CurrentUser, get_db
from app.models.schemas import (
    DocumentListResponse,
    S3FileInfo,
    S3FilesResponse,
    UploadResponse,
)
from app.services import s3_service
from app.services.db_service import log_document
from app.services.rag_service import async_ingest, get_rag_pipeline
from app.utils.exceptions import UnsupportedFileTypeError
from app.utils.file_utils import safe_remove, save_upload_file_tmp, validate_file_extension

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


# ── POST /documents/upload ────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    description=(
        "Upload a **PDF**, **TXT**, or **DOCX** file.\n\n"
        "The pipeline:\n"
        "1. Validates the file type\n"
        "2. Asynchronously uploads to S3 (if configured)\n"
        "3. Chunks, embeds, and indexes into FAISS (background task)\n"
        "4. Logs ingestion event to the database\n"
    ),
)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="PDF, TXT, or DOCX file ≤ 50 MB"),
) -> UploadResponse:
    # Validate extension
    try:
        validate_file_extension(file.filename or "")
    except ValueError as exc:
        raise UnsupportedFileTypeError(str(exc))

    # Stream to temp file (async)
    tmp_path = await save_upload_file_tmp(file)
    file_size = os.path.getsize(tmp_path)

    # Async S3 upload (non-blocking — failure is logged, not fatal)
    s3_key: str | None = None
    try:
        s3_key = await s3_service.upload_file_to_s3(tmp_path, file.filename) or None
    except Exception as exc:
        logger.warning("S3 upload skipped: %s", exc)

    # Log to DB
    try:
        await log_document(
            db,
            filename=file.filename,
            s3_key=s3_key,
            file_size_bytes=file_size,
            status="processing",
        )
    except Exception as exc:
        logger.warning("DB doc log failed: %s", exc)

    # Background indexing
    background_tasks.add_task(_index_in_background, tmp_path, file.filename)
    logger.info("Accepted '%s' for indexing (size=%d bytes).", file.filename, file_size)

    return UploadResponse(
        message=f"'{file.filename}' accepted for indexing.",
        filename=file.filename,
        status="processing",
        s3_key=s3_key,
    )


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List locally indexed documents (paginated)",
)
async def list_documents(
    current_user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> DocumentListResponse:
    pipeline = get_rag_pipeline()
    all_docs = pipeline.list_ingested_docs()
    total = len(all_docs)
    start = (page - 1) * page_size
    page_docs = all_docs[start: start + page_size]
    total_pages = max(1, (total + page_size - 1) // page_size)
    return DocumentListResponse(
        documents=page_docs, count=total,
        page=page, page_size=page_size, total_pages=total_pages,
    )


# ── GET /documents/files ──────────────────────────────────────────────────────

@router.get(
    "/files",
    response_model=S3FilesResponse,
    summary="List documents stored in S3 (paginated)",
)
async def list_s3_files(
    current_user: CurrentUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> S3FilesResponse:
    files, total = await s3_service.list_files(page=page, page_size=page_size)
    return S3FilesResponse(
        files=[S3FileInfo(**f) for f in files],
        count=total, page=page, page_size=page_size,
    )


# ── DELETE /documents/cache ───────────────────────────────────────────────────

@router.delete(
    "/cache",
    summary="Invalidate the response cache",
    description="Purges all cached RAG answers from Redis.",
)
async def invalidate_cache(current_user: CurrentUser) -> dict:
    from app.services import cache_service
    deleted = await cache_service.invalidate_all()
    return {"message": f"Cache invalidated. {deleted} keys removed."}


# ── Background task ───────────────────────────────────────────────────────────

async def _index_in_background(tmp_path: str, filename: str):
    try:
        chunk_count = await async_ingest(tmp_path, filename)
        logger.info("Indexed '%s': %d chunks.", filename, chunk_count)
    except Exception as exc:
        logger.exception("Failed to index '%s': %s", filename, exc)
    finally:
        safe_remove(tmp_path)
