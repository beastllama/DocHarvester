from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database import get_db
from backend.models import ApiToken, User
from backend.services.api_tokens import (
    create_api_token,
    is_api_token,
    list_api_tokens,
    revoke_api_token,
    user_for_api_token,
)


router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/token")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    
    class Config:
        from_attributes = True


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Personal API tokens (dh_...) are checked by hash, against revocation and the user's active flag
    if is_api_token(token):
        user = await user_for_api_token(db, token)
        if user is None:
            raise credentials_exception
        return user

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(
        select(User).where(User.email == token_data.email)
    )
    user = result.scalar_one_or_none()
    
    # Deactivated users lose access on their next request, even with a login token that has not expired
    if user is None or not user.is_active:
        raise credentials_exception
    
    return user


@router.post("/register", response_model=UserResponse)
async def register(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new user. Admins only."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can create accounts")
    # Check if user exists
    result = await db.execute(
        select(User).where(User.email == user_create.email)
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_create.password)
    
    
    user = User(
        email=user_create.email,
        hashed_password=hashed_password,
        full_name=user_create.full_name,
        is_admin=False
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=400,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_user)
):
    """Get current user info"""
    return current_user


class ApiTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiTokenInfo(BaseModel):
    id: int
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class ApiTokenCreated(ApiTokenInfo):
    token: str  # Shown once. It is stored only as a hash, so it cannot be shown again.


def _token_info(row: ApiToken) -> ApiTokenInfo:
    return ApiTokenInfo(
        id=row.id,
        name=row.name,
        created_at=row.created_at,
        last_used_at=row.last_used_at,
        revoked_at=row.revoked_at,
    )


@router.post("/tokens", response_model=ApiTokenCreated)
async def create_token(
    payload: ApiTokenCreate,
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a personal API token for an MCP client. Needs a password login: a token cannot mint tokens."""
    if is_api_token(token):
        raise HTTPException(status_code=403, detail="Log in with your password to create a token")
    row, plaintext = await create_api_token(db, current_user, payload.name)
    return ApiTokenCreated(**_token_info(row).model_dump(), token=plaintext)


@router.get("/tokens", response_model=List[ApiTokenInfo])
async def list_tokens(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List your API tokens. Plaintext values are never returned."""
    return [_token_info(row) for row in await list_api_tokens(db, current_user)]


@router.delete("/tokens/{token_id}")
async def revoke_token(
    token_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke one of your API tokens. It stops working on the next request."""
    if not await revoke_api_token(db, current_user, token_id):
        raise HTTPException(status_code=404, detail="Token not found")
    return {"revoked": token_id} 