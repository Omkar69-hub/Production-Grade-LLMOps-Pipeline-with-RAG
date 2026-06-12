"""
tests/conftest.py — Session-level test setup.

Sets required environment variables BEFORE any app module is imported,
clears the pydantic-settings lru_cache so fresh values are picked up,
removes any stale test database, and seeds the admin user directly —
independent of the FastAPI lifespan (which TestClient does NOT trigger
when used without a context manager in Starlette 0.20+).
"""

import os
import pathlib

# ── 1. Set env vars BEFORE any app import ──────────────────────────────────────
# These override .env file values for the entire test session.
os.environ.update(
    {
        "OPENAI_API_KEY": "",
        "S3_BUCKET_NAME": "",
        "REDIS_ENABLED": "false",
        "DATABASE_URL": "sqlite+aiosqlite:///./test_rag.db",
        "SECRET_KEY": "test-secret-key-minimum-32-chars-long!!",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "secret123",  # min 8 chars (enforced by user_service)
    }
)

# ── 2. Clear the lru_cache so pydantic-settings re-reads the env vars above ────
try:
    from app.config import get_settings

    get_settings.cache_clear()
except Exception:
    pass

# ── 3. Remove any stale test DB from a previous run ────────────────────────────
_db_path = pathlib.Path("test_rag.db")
if _db_path.exists():
    _db_path.unlink()

# ── 4. Session fixture: create tables + seed admin ─────────────────────────────
# NOTE: TestClient without a context manager does NOT fire FastAPI lifespan events
# in Starlette 0.20+. We therefore seed the admin here rather than relying on the
# lifespan's seed_admin_if_empty() call.
import pytest_asyncio

from app.services.db_service import get_db_session, init_db
from app.services.user_service import seed_admin_if_empty


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    """Create DB tables and seed the test admin user."""
    from app.config import get_settings

    cfg = get_settings()

    await init_db()

    async with get_db_session() as db:
        await seed_admin_if_empty(
            db,
            username=cfg.admin_username,  # "admin"
            password=cfg.admin_password,  # "secret123"
        )
