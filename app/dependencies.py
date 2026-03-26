"""
app/dependencies.py — FastAPI dependency injection container.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_service import TokenClaims, decode_access_token
from app.services.db_service import get_db
from app.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


# ── Auth — returns full TokenClaims (username + role) ─────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials |
                           None, Depends(_bearer)] = None,
) -> TokenClaims:
    """
    Dependency: validate Bearer JWT and return decoded TokenClaims.
    Use `claims.sub` for the username, `claims.is_admin()` for role checks.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Convenience aliases ────────────────────────────────────────────────────────

CurrentUser = Annotated[TokenClaims, Depends(get_current_user)]
DbSession = Annotated[object, Depends(get_db)]


# ── Admin guard helper ────────────────────────────────────────────────────────

def require_admin(claims: TokenClaims) -> None:
    """Raise HTTP 403 if the caller is not an admin."""
    if not claims.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation.",
        )
