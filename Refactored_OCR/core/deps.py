import hashlib
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_db
from db.models import User, ApiKey
from core.security import decode_access_token

security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Authenticates current user using:
    1. X-API-Key header (API token e.g. sk-gh-...)
    2. Authorization: Bearer <token> (API key or JWT access token)
    3. Cookie: access_token (JWT session)
    Raises 401 Unauthorized if missing, invalid, or expired.
    """
    # 1. Check X-API-Key header
    api_key_header = request.headers.get("x-api-key")
    if api_key_header:
        api_key_str = api_key_header.strip()
        key_hash = hashlib.sha256(api_key_str.encode("utf-8")).hexdigest()
        stmt = (
            select(ApiKey)
            .options(selectinload(ApiKey.user))
            .where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
        )
        result = await db.execute(stmt)
        key_obj = result.scalar_one_or_none()
        if key_obj and key_obj.user:
            if not key_obj.user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="حساب کاربری متصل به این کلید غیرفعال است."
                )
            return key_obj.user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="کلید API نامعتبر است یا منقضی شده است."
            )

    # 2. Check Authorization Bearer header
    raw_bearer = auth.credentials if auth and auth.credentials else None
    if raw_bearer:
        raw_bearer = raw_bearer.strip()
        # If it's formatted as an API key (sk-gh-...)
        if raw_bearer.startswith("sk-gh-"):
            key_hash = hashlib.sha256(raw_bearer.encode("utf-8")).hexdigest()
            stmt = (
                select(ApiKey)
                .options(selectinload(ApiKey.user))
                .where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            )
            result = await db.execute(stmt)
            key_obj = result.scalar_one_or_none()
            if key_obj and key_obj.user:
                if not key_obj.user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="حساب کاربری متصل به این کلید غیرفعال است."
                    )
                return key_obj.user
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="کلید API نامعتبر است یا منقضی شده است."
                )
        else:
            # It's a JWT access token
            payload = decode_access_token(raw_bearer)
            if payload and "sub" in payload:
                stmt = select(User).where(User.username == payload["sub"])
                result = await db.execute(stmt)
                user = result.scalar_one_or_none()
                if user:
                    if not user.is_active:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="این حساب کاربری غیرفعال شده است."
                        )
                    return user

    # 3. Check access_token cookie
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        payload = decode_access_token(cookie_token)
        if payload and "sub" in payload:
            stmt = select(User).where(User.username == payload["sub"])
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            if user:
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="این حساب کاربری غیرفعال شده است."
                    )
                return user

    # 4. Check query parameters (e.g. ?api_key=... or ?token=...)
    query_token = request.query_params.get("api_key") or request.query_params.get("token")
    if query_token:
        query_token = query_token.strip()
        if query_token.startswith("sk-gh-"):
            key_hash = hashlib.sha256(query_token.encode("utf-8")).hexdigest()
            stmt = (
                select(ApiKey)
                .options(selectinload(ApiKey.user))
                .where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            )
            result = await db.execute(stmt)
            key_obj = result.scalar_one_or_none()
            if key_obj and key_obj.user:
                if not key_obj.user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="حساب کاربری متصل به این کلید غیرفعال است."
                    )
                return key_obj.user
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="کلید API نامعتبر است یا منقضی شده است."
                )
        else:
            payload = decode_access_token(query_token)
            if payload and "sub" in payload:
                stmt = select(User).where(User.username == payload["sub"])
                result = await db.execute(stmt)
                user = result.scalar_one_or_none()
                if user:
                    if not user.is_active:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="این حساب کاربری غیرفعال شده است."
                        )
                    return user

    # No valid authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="دسترسی غیرمجاز. لطفاً وارد شوید یا توکن معتبر ارسال کنید."
    )


async def get_current_user_optional(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Returns current user if authenticated, otherwise None without raising."""
    try:
        return await get_current_user(request=request, auth=auth, db=db)
    except HTTPException:
        return None


async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Enforces administrator privileges."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="دسترسی به این بخش نیازمند دسترسی مدیریت (Admin) است."
        )
    return current_user


async def check_docs_access(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Guards API documentation (/docs, /redoc, /openapi.json).
    Requires active authentication, and restricts to administrator accounts if configured.
    For browser navigation without credentials, redirects to the login screen.
    """
    import config
    from fastapi.responses import RedirectResponse

    user = None
    try:
        user = await get_current_user(request=request, db=db)
    except HTTPException:
        pass

    if not user:
        accept_header = request.headers.get("accept", "")
        # If browser navigated to HTML docs without login, redirect to login page
        if "text/html" in accept_header and not request.url.path.endswith(".json"):
            return RedirectResponse(
                url=f"/login?next={request.url.path}",
                status_code=status.HTTP_307_TEMPORARY_REDIRECT
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="مشاهده مستندات فنی API نیازمند احراز هویت با توکن معتبر است."
        )

    if getattr(config, "DOCS_REQUIRE_ADMIN", True) and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="دسترسی به مستندات و شمای OpenAPI فقط برای مدیران سامانه (Admin) مجاز است."
        )

    return user
