"""
app/services/s3_service.py — Async-friendly AWS S3 integration.
Uses run_in_executor to keep blocking boto3 calls off the event loop.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.utils.exceptions import S3OperationError

logger = logging.getLogger(__name__)
S3_PREFIX = "documents/"
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="s3-worker")
_client: Optional[object] = None


def _get_s3_client():
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


def _sync_upload(local_path: str, bucket: str, key: str) -> str:
    _get_s3_client().upload_file(local_path, bucket, key)
    return key


def _sync_download(bucket: str, key: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _get_s3_client().download_file(bucket, key, dest)
    return True


def _sync_list(bucket: str, prefix: str) -> list[dict]:
    resp = _get_s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [
        {
            "key": obj["Key"],
            "size_bytes": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
        }
        for obj in resp.get("Contents", [])
    ]


async def upload_file_to_s3(local_path: str, filename: str) -> str:
    """Upload a file to S3 asynchronously. Returns S3 key or ''."""
    cfg = get_settings()
    if not cfg.s3_enabled:
        logger.warning(
            "S3 not configured — skipping upload of '%s'.", filename)
        return ""
    key = f"{S3_PREFIX}{filename}"
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _executor, partial(_sync_upload, local_path,
                               cfg.s3_bucket_name, key)
        )
        logger.info("Uploaded '%s' → s3://%s/%s",
                    filename, cfg.s3_bucket_name, key)
        return key
    except (BotoCoreError, ClientError) as exc:
        raise S3OperationError(
            f"S3 upload failed for '{filename}': {exc}") from exc


async def download_file_from_s3(key: str, dest_path: str) -> bool:
    """Download an S3 object asynchronously. Returns True on success."""
    cfg = get_settings()
    if not cfg.s3_enabled:
        return False
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _executor, partial(
                _sync_download, cfg.s3_bucket_name, key, dest_path)
        )
        logger.info("Downloaded s3://%s/%s → %s",
                    cfg.s3_bucket_name, key, dest_path)
        return True
    except (BotoCoreError, ClientError) as exc:
        raise S3OperationError(
            f"S3 download failed for key '{key}': {exc}") from exc


async def list_files(page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
    """
    List S3 documents. Returns (page_items, total_count).
    Returns ([], 0) when S3 is not configured.
    """
    cfg = get_settings()
    if not cfg.s3_enabled:
        return [], 0
    loop = asyncio.get_event_loop()
    try:
        all_files = await loop.run_in_executor(
            _executor, partial(_sync_list, cfg.s3_bucket_name, S3_PREFIX)
        )
        total = len(all_files)
        start = (page - 1) * page_size
        return all_files[start: start + page_size], total
    except (BotoCoreError, ClientError) as exc:
        raise S3OperationError(f"Failed to list S3 objects: {exc}") from exc
