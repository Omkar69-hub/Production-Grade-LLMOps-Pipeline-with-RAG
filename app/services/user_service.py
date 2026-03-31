"""
app/services/user_service.py — DB-backed user management.

Handles:
  - Creating users (with bcrypt hashing)
  - Looking up users by username
  - Updating last_login timestamp
  - Listing / deactivating users (admin only)
"""

import logging
from datetime import datetime, timezone

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import User
from app.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Minimum password length enforced at creation time
MIN_PASSWORD_LENGTH = 8


# ── Password helpers ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* password."""
    return pwd_context.verify(plain, hashed)


# ── CRUD ──────────────────────────────────────────────────────────────────────


async def create_user(
    db: AsyncSession,
    *,
    username: str,
    password: str,
    email: str | None = None,
    role: str = "viewer",
) -> User:
    """
    Create and persist a new user.

    Raises
    ------
    ValueError
        If the username already exists or the password is too short.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")

    # Uniqueness check
    existing = await get_user_by_username(db, username)
    if existing:
        raise ValueError(f"Username '{username}' is already taken.")

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.flush()  # get id without full commit
    logger.info("Created user '%s' (role=%s).", username, role)
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    """Return the User row for *username*, or None if not found."""
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    """Return the User row for *user_id*, or None if not found."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession, *, page: int = 1, page_size: int = 20
) -> tuple[list[User], int]:
    """Return a paginated list of all users and the total count."""
    from sqlalchemy import func

    total_result = await db.execute(select(func.count()).select_from(User))
    total = total_result.scalar_one()

    users_result = await db.execute(
        select(User)
        .order_by(User.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(users_result.scalars()), total


async def authenticate(db: AsyncSession, username: str, password: str) -> User:
    """
    Look up the user and verify the password.

    Returns
    -------
    User
        The authenticated user record.

    Raises
    ------
    AuthenticationError
        For any invalid credential or inactive account.
    """
    user = await get_user_by_username(db, username)

    # Use a dummy verify call even on missing user to avoid timing attacks
    dummy = hash_password("dummy-to-prevent-timing-leak")
    password_ok = verify_password(password, user.hashed_password if user else dummy)

    if not user or not password_ok:
        raise AuthenticationError("Invalid username or password.")

    if not user.is_active:
        raise AuthenticationError("Account is disabled. Contact an administrator.")

    # Update last_login
    user.last_login = datetime.now(timezone.utc)
    await db.flush()
    logger.info("User '%s' authenticated successfully.", username)
    return user


async def deactivate_user(db: AsyncSession, user_id: int) -> User:
    """Disable a user account (soft delete)."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise ValueError(f"User id={user_id} not found.")
    user.is_active = False
    await db.flush()
    logger.info("Deactivated user id=%d ('%s').", user_id, user.username)
    return user


async def change_password(db: AsyncSession, user_id: int, new_password: str) -> User:
    """Update a user's password hash."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
    user = await get_user_by_id(db, user_id)
    if not user:
        raise ValueError(f"User id={user_id} not found.")
    user.hashed_password = hash_password(new_password)
    await db.flush()
    logger.info("Password changed for user id=%d.", user_id)
    return user


# ── First-run seed ────────────────────────────────────────────────────────────


async def seed_admin_if_empty(db: AsyncSession, *, username: str, password: str) -> bool:
    """
    Create the initial admin user ONLY if the users table is empty.
    Called once at application startup.

    Returns True if the admin was created, False if users already exist.
    """
    from sqlalchemy import func

    count_result = await db.execute(select(func.count()).select_from(User))
    if count_result.scalar_one() > 0:
        return False

    await create_user(db, username=username, password=password, role="admin")
    logger.info(
        "First-run: admin user '%s' created. "
        "CHANGE THIS PASSWORD immediately via POST /api/v1/users/me/password",
        username,
    )
    return True
