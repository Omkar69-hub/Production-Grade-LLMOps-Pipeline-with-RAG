"""
app/services/auth_service.py — JWT creation and verification.

Authentication flow:
  POST /api/v1/auth/token
    ↓  user_service.authenticate(db, username, password)   [DB lookup + bcrypt verify]
    ↓  create_access_token(username, role)
    →  returns signed JWT

Protected endpoint:
    Authorization: Bearer <token>
    ↓  decode_access_token(token)                          [jose verify + expiry check]
    →  returns TokenClaims(sub=username, role=role)
"""

import logging
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import get_settings
from app.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


# ── Token claims dataclass ─────────────────────────────────────────────────────


class TokenClaims:
    """Decoded JWT payload, available in request handlers via dependency injection."""

    __slots__ = ("sub", "role")

    def __init__(self, sub: str, role: str = "viewer"):
        self.sub = sub  # username
        self.role = role

    def is_admin(self) -> bool:
        return self.role == "admin"


# ── JWT helpers ────────────────────────────────────────────────────────────────


def create_access_token(username: str, role: str = "viewer") -> tuple[str, int]:
    """
    Create and sign a JWT containing username and role.

    Returns
    -------
    tuple[str, int]
        (encoded_jwt, expires_in_seconds)
    """
    cfg = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=cfg.access_token_expire_minutes)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": expire,
    }
    token = jwt.encode(payload, cfg.secret_key, algorithm=cfg.algorithm)
    return token, cfg.access_token_expire_minutes * 60


def decode_access_token(token: str) -> TokenClaims:
    """
    Decode and validate a JWT.

    Returns
    -------
    TokenClaims
        Verified claims from the token payload.

    Raises
    ------
    AuthenticationError
        If the token is invalid, expired, or missing required claims.
    """
    cfg = get_settings()
    try:
        payload = jwt.decode(token, cfg.secret_key, algorithms=[cfg.algorithm])
    except JWTError as exc:
        raise AuthenticationError(f"Token validation failed: {exc}") from exc

    username: str | None = payload.get("sub")
    if not username:
        raise AuthenticationError("Token payload missing 'sub' claim.")

    role: str = payload.get("role", "viewer")
    return TokenClaims(sub=username, role=role)
