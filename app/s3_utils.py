"""
s3_utils.py — AWS S3 integration helpers.

Provides:
  - upload_file_to_s3(local_path, filename)  → str  (S3 key or "")
  - download_file_from_s3(key, dest_path)    → bool
  - list_files()                             → list[dict]

All credentials come from environment variables via config.py — no secrets
are ever hard-coded in this module.
"""

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

# S3 key prefix used for all document objects
S3_PREFIX = "documents/"

# ── Singleton boto3 S3 client ─────────────────────────────────────────────────
_client: Optional["boto3.client"] = None  # type: ignore[type-arg]


def _get_s3_client():
    """Lazily create and cache a boto3 S3 client."""
    global _client
    if _client is None:
        cfg = get_settings()
        _client = boto3.client(
            "s3",
            aws_access_key_id=cfg.aws_access_key_id or None,
            aws_secret_access_key=cfg.aws_secret_access_key or None,
            region_name=cfg.aws_region,
        )
    return _client


# ── Public API ────────────────────────────────────────────────────────────────


def upload_file_to_s3(local_path: str, filename: str) -> str:
    """
    Upload a local file to S3 under the 'documents/' prefix.

    Parameters
    ----------
    local_path : str
        Absolute path of the file on disk.
    filename : str
        Desired object name inside S3 (without the prefix).

    Returns
    -------
    str
        The full S3 object key on success, or an empty string on failure /
        when S3 is not configured.
    """
    cfg = get_settings()
    if not cfg.s3_enabled:
        logger.warning("S3 not configured — skipping upload of '%s'.", filename)
        return ""

    key = f"{S3_PREFIX}{filename}"
    try:
        _get_s3_client().upload_file(local_path, cfg.s3_bucket_name, key)
        logger.info("Uploaded '%s' → s3://%s/%s", filename, cfg.s3_bucket_name, key)
        return key
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 upload failed for '%s': %s", filename, exc)
        return ""


def download_file_from_s3(key: str, dest_path: str) -> bool:
    """
    Download an S3 object to a local destination path.

    Parameters
    ----------
    key : str
        Full S3 object key (e.g. 'documents/report.pdf').
    dest_path : str
        Local file path to write the object to.

    Returns
    -------
    bool
        True on success, False on failure or when S3 is not configured.
    """
    cfg = get_settings()
    if not cfg.s3_enabled:
        logger.warning("S3 not configured — skipping download of '%s'.", key)
        return False

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        _get_s3_client().download_file(cfg.s3_bucket_name, key, dest_path)
        logger.info("Downloaded s3://%s/%s → %s", cfg.s3_bucket_name, key, dest_path)
        return True
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 download failed for key '%s': %s", key, exc)
        return False


def list_files() -> list[dict]:
    """
    List all document objects stored in S3 under the 'documents/' prefix.

    Returns
    -------
    list[dict]
        Each item contains: key, size (bytes), last_modified (ISO string).
        Returns an empty list when S3 is not configured or on error.
    """
    cfg = get_settings()
    if not cfg.s3_enabled:
        logger.warning("S3 not configured — returning empty file list.")
        return []

    try:
        response = _get_s3_client().list_objects_v2(Bucket=cfg.s3_bucket_name, Prefix=S3_PREFIX)
        objects = response.get("Contents", [])
        return [
            {
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            }
            for obj in objects
        ]
    except (BotoCoreError, ClientError) as exc:
        logger.error("Failed to list S3 objects: %s", exc)
        return []
