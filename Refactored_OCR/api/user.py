from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_db
from db.models import User, ApiKey
from core.deps import get_current_user
from core.security import generate_raw_api_key

router = APIRouter(prefix="/api/user", tags=["User Dashboard"])


@router.get("/token")
async def get_my_token(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves active API token for currently authenticated user or provisions one if none exists."""
    stmt = (
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id, ApiKey.is_active == True)
        .order_by(desc(ApiKey.created_at))
    )
    result = await db.execute(stmt)
    key = result.scalars().first()

    if not key:
        raw_key, key_prefix, key_hash = generate_raw_api_key()
        key = ApiKey(
            user_id=current_user.id,
            name=f"Key for {current_user.username}",
            key_prefix=key_prefix,
            secret_key=raw_key,
            key_hash=key_hash,
            is_active=True
        )
        db.add(key)
        await db.commit()
        await db.refresh(key)

    return {
        "status": "success",
        "token": key.secret_key,
        "prefix": key.key_prefix,
        "name": key.name,
        "created_at": key.created_at.isoformat() if key.created_at else None
    }


@router.post("/token/regenerate")
async def regenerate_my_token(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Revokes all existing keys for the user and issues a fresh random API token."""
    stmt = select(ApiKey).where(ApiKey.user_id == current_user.id, ApiKey.is_active == True)
    result = await db.execute(stmt)
    active_keys = result.scalars().all()
    for k in active_keys:
        k.is_active = False

    raw_key, key_prefix, key_hash = generate_raw_api_key()
    new_key = ApiKey(
        user_id=current_user.id,
        name=f"Key for {current_user.username}",
        key_prefix=key_prefix,
        secret_key=raw_key,
        key_hash=key_hash,
        is_active=True
    )
    db.add(new_key)
    await db.commit()
    await db.refresh(new_key)

    return {
        "status": "success",
        "message": "توکن جدید با موفقیت صادر شد و توکن‌های قبلی باطل گردیدند.",
        "token": new_key.secret_key,
        "prefix": new_key.key_prefix,
        "created_at": new_key.created_at.isoformat() if new_key.created_at else None
    }
