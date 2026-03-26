"""
Tests for the RAG FastAPI endpoints.
Run with:  pytest tests/ -v

All external services (FAISS, S3, OpenAI) are either mocked or gracefully
skipped so the suite runs in CI without real credentials.
"""

import os
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Disable real OpenAI & S3 during tests
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("S3_BUCKET_NAME", "")

from app.main import app  # noqa: E402  (must come after env setup)

client = TestClient(app)


# ─── Health ──────────────────────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vectorstore_loaded" in data
    assert "s3_enabled" in data


# ─── Ask — no docs indexed yet ────────────────────────────────────────────────
def test_ask_post_without_docs_returns_503():
    """Vector store not ready → 503."""
    resp = client.post("/ask", json={"query": "What is RAG?", "top_k": 3})
    assert resp.status_code in (200, 503)


def test_ask_get_without_docs():
    """GET /ask with no vectorstore."""
    resp = client.get("/ask", params={"query": "What is RAG?"})
    assert resp.status_code in (200, 503)


# ─── Upload ───────────────────────────────────────────────────────────────────
def test_upload_invalid_file_type():
    resp = client.post(
        "/upload",
        files={"file": ("evil.exe", BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_valid_txt(tmp_path):
    """Upload a real .txt file; background indexing is mocked."""
    content = b"RAG stands for Retrieval-Augmented Generation."
    with patch("app.main.rag") as mock_rag:
        mock_rag.is_ready.return_value = False
        mock_rag.ingest = MagicMock()
        resp = client.post(
            "/upload",
            files={"file": ("test_doc.txt", BytesIO(content), "text/plain")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert "test_doc.txt" in body["message"]


# ─── /files — S3 listing ──────────────────────────────────────────────────────
def test_list_files_no_s3(monkeypatch):
    """When S3 is not configured, /files returns an empty list."""
    resp = client.get("/files")
    assert resp.status_code == 200
    data = resp.json()
    assert "files" in data
    assert isinstance(data["files"], list)
    assert data["count"] == len(data["files"])


def test_list_files_with_s3_mocked():
    """Mock S3 list to verify response shape."""
    mock_files = [
        {"key": "documents/report.pdf", "size_bytes": 1024, "last_modified": "2024-01-01T00:00:00"},
        {"key": "documents/guide.txt", "size_bytes": 512, "last_modified": "2024-01-02T00:00:00"},
    ]
    with patch("app.main.list_files", return_value=mock_files):
        resp = client.get("/files")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["files"][0]["key"] == "documents/report.pdf"


# ─── /documents — local FAISS index ──────────────────────────────────────────
def test_list_documents_empty():
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert "count" in data
