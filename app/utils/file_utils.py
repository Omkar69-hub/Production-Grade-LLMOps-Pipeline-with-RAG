"""
app/utils/file_utils.py — Async-safe file helpers.
"""

import os
import tempfile

import aiofiles
from fastapi import UploadFile

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 50


async def save_upload_file_tmp(upload: UploadFile) -> str:
    """
    Stream an UploadFile to a named temp file asynchronously.
    Returns the temp file path. Caller must delete it when done.
    """
    suffix = os.path.splitext(upload.filename or "file")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    async with aiofiles.open(tmp_path, "wb") as out_file:
        while chunk := await upload.read(1024 * 64):  # 64 KB chunks
            await out_file.write(chunk)

    return tmp_path


def validate_file_extension(filename: str) -> str:
    """Return lowercase extension or raise ValueError for unsupported types."""
    ext = os.path.splitext(filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return ext


def safe_remove(path: str) -> None:
    """Delete a file without raising if it is missing."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
