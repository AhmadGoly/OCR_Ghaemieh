from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_db
from db.models import User, ExtractionHistory
from core.security import hash_password
from core.deps import require_admin
import config

router = APIRouter(prefix="/api/admin", tags=["Admin Portal"], dependencies=[Depends(require_admin)])


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=4, max_length=128)
    is_admin: bool = False


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=4, max_length=128)


@router.get("/users")
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all registered user accounts with metadata."""
    stmt = select(User).order_by(desc(User.created_at))
    result = await db.execute(stmt)
    users = result.scalars().all()

    return [
        {
            "id": u.id,
            "username": u.username,
            "is_admin": u.is_admin,
            "is_active": u.is_active,
            "created_version": u.created_version or "unknown",
            "created_at": u.created_at.isoformat() if u.created_at else None
        }
        for u in users
    ]


@router.post("/users")
async def create_user(req: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """Creates a new user account with active application version recording."""
    # Check if username exists
    stmt = select(User).where(User.username == req.username)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"کاربر '{req.username}' از قبل وجود دارد."
        )

    new_user = User(
        username=req.username,
        hashed_password=hash_password(req.password),
        is_admin=req.is_admin,
        is_active=True,
        created_version=config.VERSION
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "status": "success",
        "message": f"کاربر '{new_user.username}' با موفقیت ساخته شد.",
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "is_admin": new_user.is_admin,
            "created_version": new_user.created_version
        }
    }


@router.patch("/users/{user_id}/toggle-status")
async def toggle_user_status(user_id: int, current_user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    """Toggle user active / inactive status."""
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="شما نمی‌توانید حساب کاربری خودتان را غیرفعال کنید."
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد.")

    user.is_active = not user.is_active
    await db.commit()

    status_str = "فعال" if user.is_active else "غیرفعال"
    return {"status": "success", "message": f"وضعیت کاربر به '{status_str}' تغییر یافت.", "is_active": user.is_active}


@router.post("/users/{user_id}/reset-password")
async def reset_password(user_id: int, req: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Resets password for the specified user."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد.")

    user.hashed_password = hash_password(req.new_password)
    await db.commit()

    return {"status": "success", "message": f"رمز عبور کاربر '{user.username}' با موفقیت تغییر یافت."}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, current_user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    """Deletes the specified user account."""
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="شما نمی‌توانید حساب کاربری خودتان را حذف کنید."
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="کاربر یافت نشد.")

    await db.delete(user)
    await db.commit()

    return {"status": "success", "message": f"کاربر '{user.username}' با موفقیت حذف شد."}


@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    """Returns system-wide metrics and engine status."""
    total_users_stmt = select(func.count(User.id))
    total_users = (await db.execute(total_users_stmt)).scalar() or 0

    total_extractions_stmt = select(func.count(ExtractionHistory.id))
    total_extractions = (await db.execute(total_extractions_stmt)).scalar() or 0

    return {
        "version": config.VERSION,
        "total_users": total_users,
        "total_extractions": total_extractions,
        "runtime_config": config.get_config_dict()
    }
