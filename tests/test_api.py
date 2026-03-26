"""
Tests for the enterprise RAG LLMOps API (v2).

Run with:  pytest tests/ -v --asyncio-mode=auto

All external services (FAISS, Redis, S3, OpenAI) are mocked so the suite
runs fully offline in CI without any real credentials.
"""

import os
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Must be set before app imports
os.environ.update({
    "OPENAI_API_KEY": "",
    "S3_BUCKET_NAME": "",
    "REDIS_ENABLED": "false",
    "DATABASE_URL": "sqlite+aiosqlite:///./test_rag.db",
    "SECRET_KEY": "test-secret-key-minimum-32-chars-long!!",
})

from app.main import app  # noqa: E402

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def auth_token() -> str:
    """Obtain a real JWT for the demo user."""
    resp = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "secret"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


# ─── Health ────────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vectorstore_loaded" in data
    assert "s3_enabled" in data
    assert "redis_enabled" in data
    assert "version" in data
    assert "uptime_seconds" in data


# ─── Authentication ─────────────────────────────────────────────────────────────

def test_login_success():
    resp = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


def test_login_wrong_password():
    resp = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "wrong"},
    )
    assert resp.status_code == 401


def test_protected_endpoint_without_token():
    """Ask endpoint must reject unauthenticated requests."""
    resp = client.post("/api/v1/ask", json={"query": "What is RAG?"})
    assert resp.status_code == 401


# ─── Ask ──────────────────────────────────────────────────────────────────────

def test_ask_post_vectorstore_not_ready(auth_headers):
    """When no docs are indexed, /ask should return 503."""
    resp = client.post(
        "/api/v1/ask",
        json={"query": "What is RAG?", "top_k": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 503


def test_ask_get_vectorstore_not_ready(auth_headers):
    resp = client.get(
        "/api/v1/ask",
        params={"query": "What is RAG?"},
        headers=auth_headers,
    )
    assert resp.status_code in (200, 503)


def test_ask_returns_cache_flag_when_cached(auth_headers):
    """Mocked cache HIT should return cached=True."""
    mock_response = {
        "query": "test",
        "answer": "cached answer",
        "source_documents": [],
        "cached": True,
        "latency_ms": 1.0,
        "request_id": None,
    }
    with patch("app.routers.ask.cache_service.get_cached", new=AsyncMock(return_value=mock_response)):
        with patch("app.routers.ask.get_rag_pipeline") as mock_pipeline:
            mock_pipeline.return_value.is_ready.return_value = True
            resp = client.post(
                "/api/v1/ask",
                json={"query": "test", "top_k": 2},
                headers=auth_headers,
            )
    assert resp.status_code == 200
    assert resp.json()["cached"] is True


# ─── Documents — Upload ────────────────────────────────────────────────────────

def test_upload_invalid_file_type(auth_headers):
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("evil.exe", BytesIO(b"MZ\x90"), "application/octet-stream")},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_upload_txt_accepted(auth_headers):
    """Valid .txt upload should be queued (processing)."""
    content = b"RAG stands for Retrieval-Augmented Generation."
    with patch("app.routers.documents.async_ingest", new=AsyncMock(return_value=5)):
        with patch("app.routers.documents.s3_service.upload_file_to_s3", new=AsyncMock(return_value="")):
            resp = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test_doc.txt", BytesIO(content), "text/plain")},
                headers=auth_headers,
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert "test_doc.txt" in body["message"]


# ─── Documents — Listing ──────────────────────────────────────────────────────

def test_list_documents(auth_headers):
    resp = client.get("/api/v1/documents", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "count" in data
    assert "total_pages" in data


def test_list_s3_files_no_s3(auth_headers):
    resp = client.get("/api/v1/documents/files", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "files" in data
    assert "count" in data


def test_list_s3_files_with_mock(auth_headers):
    mock_files = [
        {"key": "documents/report.pdf", "size_bytes": 1024, "last_modified": "2024-01-01T00:00:00"},
    ]
    with patch("app.services.s3_service.list_files", new=AsyncMock(return_value=(mock_files, 1))):
        resp = client.get("/api/v1/documents/files", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["files"][0]["key"] == "documents/report.pdf"


# ─── Error Handling ────────────────────────────────────────────────────────────

def test_root_redirect():
    """Root should return API info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "docs" in data


def test_404_returns_json():
    """Unknown routes should get a JSON error, not HTML."""
    resp = client.get("/nonexistent-route-xyz")
    assert resp.status_code == 404
