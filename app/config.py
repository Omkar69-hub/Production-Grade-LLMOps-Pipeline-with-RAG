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
        extra="ignore",          # silently ignore unknown env vars
        case_sensitive=False,
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-3.5-turbo"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    vectorstore_path: str = "vectorstore"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── AWS / S3 ──────────────────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""

    # ── Application ───────────────────────────────────────────────────────────
    log_level: str = "INFO"
    cors_origins: str = "*"           # comma-separated list for production

    @property
    def s3_enabled(self) -> bool:
        """True only when all S3 credentials are provided."""
        return bool(
            self.aws_access_key_id
            and self.aws_secret_access_key
            and self.s3_bucket_name
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
