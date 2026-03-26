"""
app/utils/exceptions.py — Custom exception hierarchy and global handler.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class RAGBaseException(Exception):
    """Root exception for all application errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "An unexpected error occurred."):
        self.message = message
        super().__init__(message)


class VectorStoreNotReadyError(RAGBaseException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "VECTORSTORE_NOT_READY"


class DocumentIngestionError(RAGBaseException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "INGESTION_FAILED"


class UnsupportedFileTypeError(RAGBaseException):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "UNSUPPORTED_FILE_TYPE"


class S3OperationError(RAGBaseException):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "S3_OPERATION_FAILED"


class CacheError(RAGBaseException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "CACHE_ERROR"


class AuthenticationError(RAGBaseException):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_FAILED"


class AuthorizationError(RAGBaseException):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "AUTHORIZATION_FAILED"


class RateLimitError(RAGBaseException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"


# ── Global Exception Handlers ─────────────────────────────────────────────────

def _error_body(error_code: str, message: str, request_id: str | None = None) -> dict:
    body = {"error": {"code": error_code, "message": message}}
    if request_id:
        body["request_id"] = request_id
    return body


async def rag_exception_handler(request: Request, exc: RAGBaseException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error_code, exc.message, request_id),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import logging
    logger = logging.getLogger(__name__)
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled exception [request_id=%s]", request_id, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            "INTERNAL_ERROR", "An internal server error occurred.", request_id),
    )
