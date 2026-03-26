"""
app/services/db_service.py — async SQLAlchemy session factory and repository helpers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models.db_models import Base, DocumentLog, QueryLog

logger = logging.getLogger(__name__)
_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        cfg = get_settings()
        _engine = create_async_engine(
            cfg.database_url,
            echo=cfg.debug,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def init_db() -> None:
    """Create all tables (idempotent). Call once at startup."""
    async with _get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")


async def close_db() -> None:
    """Dispose connection pool on shutdown."""
    if _engine:
        await _engine.dispose()
        logger.info("Database connection pool closed.")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a DB session and handles commit/rollback."""
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an AsyncSession."""
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Repository helpers ────────────────────────────────────────────────────────

async def log_query(
    session: AsyncSession,
    *,
    request_id: str | None,
    query: str,
    answer: str,
    source_count: int,
    latency_ms: float | None,
    cached: bool,
    top_k: int,
) -> QueryLog:
    record = QueryLog(
        request_id=request_id,
        query=query,
        answer=answer,
        source_count=source_count,
        latency_ms=latency_ms,
        cached=cached,
        top_k=top_k,
    )
    session.add(record)
    await session.flush()
    return record


async def log_document(
    session: AsyncSession,
    *,
    filename: str,
    s3_key: str | None = None,
    file_size_bytes: int | None = None,
    chunk_count: int = 0,
    status: str = "processing",
) -> DocumentLog:
    record = DocumentLog(
        filename=filename,
        s3_key=s3_key,
        file_size_bytes=file_size_bytes,
        chunk_count=chunk_count,
        status=status,
    )
    session.add(record)
    await session.flush()
    return record
