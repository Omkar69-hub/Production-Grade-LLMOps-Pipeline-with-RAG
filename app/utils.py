"""
Utility helpers: logging setup, S3 integration, temp-file saving.
"""

import os
import logging
import tempfile
import shutil

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import UploadFile


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(level: str = None):
    """Configure structured logging for the application."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy third-party loggers
    for lib in ("httpx", "httpcore", "faiss", "sentence_transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── File helpers ───────────────────────────────────────────────────────────────

async def save_upload_file_tmp(upload: UploadFile) -> str:
    """
    Stream an UploadFile to a named temp file and return its path.
    Caller is responsible for deleting the file when done.
    """
    suffix = os.path.splitext(upload.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


# ── AWS S3 helpers ─────────────────────────────────────────────────────────────

_s3_client = None


def _get_s3():
    """Lazily create a singleton boto3 S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _s3_client


def upload_to_s3(local_path: str, filename: str) -> str:
    """
    Upload a file to S3 and return its object key.
    Returns empty string if S3 is not configured.
    """
    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        logging.getLogger(__name__).warning(
            "S3_BUCKET_NAME not set — skipping S3 upload."
        )
        return ""
    try:
        key = f"documents/{filename}"
        _get_s3().upload_file(local_path, bucket, key)
        logging.getLogger(__name__).info("Uploaded '%s' → s3://%s/%s", filename, bucket, key)
        return key
    except (BotoCoreError, ClientError) as exc:
        logging.getLogger(__name__).error("S3 upload failed: %s", exc)
        return ""


def download_from_s3(key: str, dest_path: str) -> bool:
    """
    Download an S3 object to a local file.
    Returns True on success, False otherwise.
    """
    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        return False
    try:
        _get_s3().download_file(bucket, key, dest_path)
        logging.getLogger(__name__).info("Downloaded s3://%s/%s → %s", bucket, key, dest_path)
        return True
    except (BotoCoreError, ClientError) as exc:
        logging.getLogger(__name__).error("S3 download failed: %s", exc)
        return False


def list_s3_documents() -> list[str]:
    """Return a list of document keys stored in S3 under the 'documents/' prefix."""
    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        return []
    try:
        resp = _get_s3().list_objects_v2(Bucket=bucket, Prefix="documents/")
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except (BotoCoreError, ClientError) as exc:
        logging.getLogger(__name__).error("Failed to list S3 objects: %s", exc)
        return []
