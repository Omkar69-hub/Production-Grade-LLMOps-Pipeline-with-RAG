"""
config.py — Centralised application settings using pydantic-settings.
All configuration is loaded from environment variables (or .env file).
No secrets are ever hard-coded here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Production-Grade RAG API"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: str = "*"
    api_prefix: str = "/api/v1"

    # ── JWT Authentication ─────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production-use-a-long-random-string"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ── First-run admin seed ──────────────────────────────────────────────────
    # Used ONLY when the users table is empty (first start).
    # Set a strong ADMIN_PASSWORD in .env — never use the default in production.
    admin_username: str = "admin"
    admin_password: str = "changeme123"

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-3.5-turbo"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    vectorstore_path: str = "vectorstore"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Hybrid Search ─────────────────────────────────────────────────────────
    bm25_weight: float = 0.3  # weight for BM25 in hybrid search
    vector_weight: float = 0.7  # weight for vector similarity
    reranker_top_k: int = 10  # candidates fed to re-ranker
    final_top_k: int = 4  # results returned after re-ranking

    # ── AWS / S3 ──────────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # ── Database (PostgreSQL or SQLite fallback) ───────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./rag_llmops.db"

    # ── Redis (caching) ───────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 300  # 5 minutes default TTL
    redis_enabled: bool = False  # set True when Redis is available

    # ── MLflow Tracking ───────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "rag-llmops"
    mlflow_enabled: bool = False

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def s3_enabled(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key and self.s3_bucket_name)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
