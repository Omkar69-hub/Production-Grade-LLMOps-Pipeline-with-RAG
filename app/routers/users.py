"""
app/routers/users.py — User management endpoints.

Role-based access:
  POST   /users              → admin only  (create a user)
  GET    /users              → admin only  (list all users)
  GET    /users/me           → any authenticated user (own profile)
  POST   /users/me/password  → any authenticated user (change own password)
  DELETE /users/{id}         → admin only  (deactivate user)
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.schemas import (
    ChangePasswordRequest,
    UserCreateRequest,
    UserListResponse,
    UserProfileResponse,
)
from app.services import user_service
from app.services.auth_service import TokenClaims, decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["Users"])

_bearer = HTTPBearer(auto_error=False)


# ── Shared dependency: decode JWT and return claims ────────────────────────────


async def get_claims(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> TokenClaims:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(credentials.credentials)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(claims: TokenClaims) -> None:
    if not claims.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )


# ── POST /users — create a new user (admin only) ──────────────────────────────


@router.post(
    "",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user [admin]",
)
async def create_user(
    body: UserCreateRequest,
    claims: Annotated[TokenClaims, Depends(get_claims)],
    db: AsyncSession = Depends(get_db),
) -> UserProfileResponse:
    require_admin(claims)
    try:
        user = await user_service.create_user(
            db,
            username=body.username,
            password=body.password,
            email=body.email,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return UserProfileResponse.model_validate(user)


# ── GET /users — list all users (admin only) ──────────────────────────────────


@router.get(
    "",
    response_model=UserListResponse,
    summary="List all users [admin]",
)
async def list_users(
    claims: Annotated[TokenClaims, Depends(get_claims)],
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> UserListResponse:
    require_admin(claims)
    users, total = await user_service.list_users(db, page=page, page_size=page_size)
    total_pages = max(1, (total + page_size - 1) // page_size)
    return UserListResponse(
        users=[UserProfileResponse.model_validate(u) for u in users],
        count=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ── GET /users/me — own profile ───────────────────────────────────────────────


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get your own profile",
)
async def get_my_profile(
    claims: Annotated[TokenClaims, Depends(get_claims)],
    db: AsyncSession = Depends(get_db),
) -> UserProfileResponse:
    user = await user_service.get_user_by_username(db, claims.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserProfileResponse.model_validate(user)


# ── POST /users/me/password — change own password ─────────────────────────────


@router.post(
    "/me/password",
    summary="Change your password",
    status_code=status.HTTP_200_OK,
)
async def change_my_password(
    body: ChangePasswordRequest,
    claims: Annotated[TokenClaims, Depends(get_claims)],
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Verify current password before allowing the change
    try:
        await user_service.authenticate(db, claims.sub, body.current_password)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )
    try:
        user = await user_service.get_user_by_username(db, claims.sub)
        await user_service.change_password(db, user.id, body.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return {"message": "Password updated successfully."}


# ── DELETE /users/{user_id} — deactivate (admin only) ────────────────────────


@router.delete(
    "/{user_id}",
    summary="Deactivate a user account [admin]",
    status_code=status.HTTP_200_OK,
)
async def deactivate_user(
    user_id: int,
    claims: Annotated[TokenClaims, Depends(get_claims)],
    db: AsyncSession = Depends(get_db),
) -> dict:
    require_admin(claims)
    try:
        user = await user_service.deactivate_user(db, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return {"message": f"User '{user.username}' deactivated.", "id": user.id}
