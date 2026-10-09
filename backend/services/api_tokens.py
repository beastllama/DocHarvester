"""Personal API tokens for MCP clients such as Hermes Agent.

A token is "dh_" plus 32 random bytes. Only its SHA-256 hash is stored, so a leaked
database does not reveal usable tokens. Every use re-checks that the token is not
revoked and that its user is still active.
"""
import hashlib
import secrets
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import ApiToken, User

TOKEN_PREFIX = "dh_"


def is_api_token(value: Optional[str]) -> bool:
    return bool(value) and value.startswith(TOKEN_PREFIX)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_api_token(db: AsyncSession, user: User, name: str) -> Tuple[ApiToken, str]:
    """Create a token. The plaintext is returned once and never stored."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    row = ApiToken(user_id=user.id, name=name, token_hash=hash_token(token), created_at=datetime.utcnow())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row, token


async def user_for_api_token(db: AsyncSession, token: Optional[str]) -> Optional[User]:
    """The user behind a valid token, or None if the token is unknown, revoked, or its user is inactive."""
    if not is_api_token(token):
        return None
    result = await db.execute(
        select(ApiToken, User)
        .join(User, User.id == ApiToken.user_id)
        .where(ApiToken.token_hash == hash_token(token))
    )
    row = result.first()
    if row is None:
        return None
    api_token, user = row
    if api_token.revoked_at is not None or not user.is_active:
        return None
    api_token.last_used_at = datetime.utcnow()
    await db.commit()
    return user


async def list_api_tokens(db: AsyncSession, user: User) -> List[ApiToken]:
    result = await db.execute(
        select(ApiToken).where(ApiToken.user_id == user.id).order_by(ApiToken.id)
    )
    return list(result.scalars().all())


async def revoke_api_token(db: AsyncSession, user: User, token_id: int) -> bool:
    """Revoke one of the user's own tokens. False if it does not exist or belongs to someone else."""
    result = await db.execute(
        select(ApiToken).where(ApiToken.id == token_id, ApiToken.user_id == user.id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    if row.revoked_at is None:
        row.revoked_at = datetime.utcnow()
        await db.commit()
    return True
