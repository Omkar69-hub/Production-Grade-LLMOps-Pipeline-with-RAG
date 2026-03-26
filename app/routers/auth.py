"""
app/routers/auth.py — JWT authentication endpoints.
Delegates credential verification to user_service (DB-backed + bcrypt).
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.schemas import TokenRequest, TokenResponse
from app.services.auth_service import create_access_token
from app.services.user_service import authenticate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
    description=(
        "Exchange valid credentials for a short-lived JWT Bearer token.\n\n"
        "Use the returned `access_token` in the `Authorization: Bearer <token>` "
        "header on all protected endpoints.\n\n"
        "**First-run default** (change immediately):\n"
        "> username: `admin`  |  password: set via `ADMIN_PASSWORD` env var\n\n"
        "Change your password via `POST /api/v1/users/me/password`."
    ),
)
async def login(
    body: TokenRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user = await authenticate(db, body.username, body.password)
    token, expires_in = create_access_token(user.username, role=user.role)
    logger.info("Token issued for '%s' (role=%s).", user.username, user.role)
    return TokenResponse(access_token=token, expires_in=expires_in)
