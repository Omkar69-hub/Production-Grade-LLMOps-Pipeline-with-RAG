"""
app/models/schemas.py — Pydantic v2 request/response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    """Credentials for obtaining a JWT."""
    username: str = Field(..., min_length=1, examples=["admin"])
    password: str = Field(..., min_length=1, examples=["changeme123"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")


# ── Users ─────────────────────────────────────────────────────────────────────

class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, examples=["alice"])
    password: str = Field(..., min_length=8, examples=["str0ng-p@ssword!"])
    email: str | None = Field(None, examples=["alice@example.com"])
    role: Literal["admin", "viewer"] = "viewer"

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, hyphens, and underscores.")
        return v.lower()


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, examples=["n3w-str0ng-p@ss!"])


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: datetime
    last_login: datetime | None

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserProfileResponse]
    count: int
    page: int
    page_size: int
    total_pages: int


# ── QA / Ask ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["What is RAG?"])
    top_k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    metadata_filter: dict[str, Any] | None = Field(
        None, description="Optional metadata key-value filters"
    )
    stream: bool = Field(False, description="Enable streaming response")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SourceDocument(BaseModel):
    content: str
    source: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: list[SourceDocument] = []
    cached: bool = False
    latency_ms: float | None = None
    request_id: str | None = None


# ── Documents ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    filename: str
    status: str
    s3_key: str | None = None
    request_id: str | None = None


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class DocumentListResponse(BaseModel):
    documents: list[str]
    count: int
    page: int
    page_size: int
    total_pages: int


class S3FileInfo(BaseModel):
    key: str
    size_bytes: int
    last_modified: str


class S3FilesResponse(BaseModel):
    files: list[S3FileInfo]
    count: int
    page: int
    page_size: int


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    s3_enabled: bool
    redis_enabled: bool
    db_connected: bool
    version: str
    uptime_seconds: float


# ── Query History ─────────────────────────────────────────────────────────────

class QueryHistoryItem(BaseModel):
    id: int
    query: str
    answer: str
    latency_ms: float | None
    cached: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryHistoryResponse(BaseModel):
    items: list[QueryHistoryItem]
    count: int
    page: int
    page_size: int
    total_pages: int


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None
