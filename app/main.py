"""
app/main.py — FastAPI application factory (v2 enterprise edition).

On first run the lifespan handler creates an admin user automatically
using credentials from ADMIN_USERNAME / ADMIN_PASSWORD env vars.
Change the password immediately after startup via:
  POST /api/v1/users/me/password
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.routers import ask, auth, documents, health, users
from app.services.db_service import close_db, get_db_session, init_db
from app.services.rag_service import get_rag_pipeline
from app.services.cache_service import close_cache
from app.services.user_service import seed_admin_if_empty
from app.utils.exceptions import (
    RAGBaseException,
    generic_exception_handler,
    rag_exception_handler,
)
from app.utils.logging import setup_logging

# ─── Bootstrap logging ────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ─── Rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("=== RAG LLMOps API v%s starting ===", cfg.app_version)

    # ── Database: create tables ───────────────────────────────────────────────
    await init_db()

    # ── Seed admin user on first run ──────────────────────────────────────────
    admin_username = cfg.admin_username
    admin_password = cfg.admin_password
    async with get_db_session() as db:
        created = await seed_admin_if_empty(
            db, username=admin_username, password=admin_password
        )
        if created:
            logger.warning(
                "FIRST RUN: Admin user '%s' created. "
                "Change the password NOW via POST /api/v1/users/me/password",
                admin_username,
            )

    # ── RAG pipeline: load persisted FAISS index if present ───────────────────
    pipeline = get_rag_pipeline()
    pipeline.load_existing_vectorstore()
    logger.info("RAG pipeline ready | vectorstore_loaded=%s", pipeline.is_ready())

    yield

    # ── Teardown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await close_cache()
    await close_db()


# ─── App factory ──────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "## Production-Grade RAG LLMOps API\n\n"
            "Enterprise-level Retrieval-Augmented Generation system.\n\n"
            "### Quick Start\n"
            "1. `POST /api/v1/auth/token` — Obtain a JWT (username/password)\n"
            "2. Click **Authorize** above and paste the `access_token`\n"
            "3. `POST /api/v1/documents/upload` — Upload a PDF/TXT/DOCX\n"
            "4. `POST /api/v1/ask` — Ask a question\n\n"
            "### First-Run Credentials\n"
            "Username: `admin` | Password: value of `ADMIN_PASSWORD` env var\n\n"
            "> **Security:** Change the admin password immediately via "
            "`POST /api/v1/users/me/password`"
        ),
        openapi_tags=[
            {"name": "System",         "description": "Health and monitoring"},
            {"name": "Authentication", "description": "JWT token management"},
            {"name": "Users",          "description": "User management (admin-only for most ops)"},
            {"name": "QA",             "description": "Retrieval-Augmented Generation queries"},
            {"name": "Documents",      "description": "Document ingestion and management"},
        ],
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request-ID + latency logging ──────────────────────────────────────────
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency_ms:.1f}ms"
        logger.info(
            "HTTP %s %s → %d | %.0fms | id=%s",
            request.method, request.url.path,
            response.status_code, latency_ms, request_id,
        )
        return response

    # ── Rate limiting ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Global exception handlers ──────────────────────────────────────────────
    app.add_exception_handler(RAGBaseException, rag_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = cfg.api_prefix
    app.include_router(health.router)                        # GET  /health
    app.include_router(auth.router,      prefix=prefix)      # POST /api/v1/auth/token
    app.include_router(users.router,     prefix=prefix)      # /api/v1/users/**
    app.include_router(ask.router,       prefix=prefix)      # /api/v1/ask/**
    app.include_router(documents.router, prefix=prefix)      # /api/v1/documents/**

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({
            "message": "RAG LLMOps API",
            "docs": "/docs",
            "health": "/health",
            "version": cfg.app_version,
        })

    return app


app = create_app()
