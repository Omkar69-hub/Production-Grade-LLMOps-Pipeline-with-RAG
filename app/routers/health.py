"""
app/routers/health.py — Health and readiness check endpoint.
"""

import time
from fastapi import APIRouter
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.rag_service import get_rag_pipeline

router = APIRouter(tags=["System"])
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns service readiness. Use this for load-balancer health probes.",
)
async def health_check() -> HealthResponse:
    cfg = get_settings()
    pipeline = get_rag_pipeline()

    # Quick DB connectivity check
    db_ok = False
    try:
        from app.services.db_service import _get_engine
        async with _get_engine().connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        vectorstore_loaded=pipeline.is_ready(),
        s3_enabled=cfg.s3_enabled,
        redis_enabled=cfg.redis_enabled,
        db_connected=db_ok,
        version=cfg.app_version,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
