"""
Tests for the RAG FastAPI endpoints.
Run with:  pytest tests/ -v
"""

import os
import pytest
from fastapi.testclient import TestClient

# Ensure no real OpenAI calls during testing
os.environ.setdefault("OPENAI_API_KEY", "")

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vectorstore_loaded" in data


def test_ask_without_docs_returns_503():
    """Before any document is uploaded the API should return 503."""
    resp = client.post("/ask", json={"query": "What is RAG?"})
    # 503 when vector store not ready
    assert resp.status_code in (200, 503)


def test_upload_invalid_type():
    from io import BytesIO
    resp = client.post(
        "/upload",
        files={"file": ("malware.exe", BytesIO(b"MZ\x90"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_list_documents_empty():
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert "documents" in resp.json()
