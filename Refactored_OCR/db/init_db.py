import logging
from sqlalchemy import select
from .session import Base, engine, AsyncSessionLocal
from .models import User, ApiKey, ExtractionHistory, OCRBookTask, OCRBookTaskPage
from core.security import hash_password, generate_raw_api_key
import config

logger = logging.getLogger("init_db")

ADMIN_USERNAME = getattr(config, "ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = getattr(config, "ADMIN_PASSWORD", "admin123")
DEMO_USERNAME = getattr(config, "DEMO_USERNAME", "demo")
DEMO_PASSWORD = getattr(config, "DEMO_PASSWORD", "demo123")


async def init_database() -> None:
    """Creates database tables and provisions initial admin and demo users if not present."""
    logger.info("Initializing database schema...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

    async with AsyncSessionLocal() as session:
        try:
            # 1. Provision / Verify Admin User
            stmt = select(User).where(User.username == ADMIN_USERNAME)
            result = await session.execute(stmt)
            existing_admin = result.scalar_one_or_none()

            if not existing_admin:
                logger.info(f"Provisioning default admin user '{ADMIN_USERNAME}'...")
                admin_user = User(
                    username=ADMIN_USERNAME,
                    hashed_password=hash_password(ADMIN_PASSWORD),
                    is_admin=True,
                    is_active=True,
                    created_version=config.VERSION
                )
                session.add(admin_user)
                await session.flush()

                raw_key, key_prefix, key_hash = generate_raw_api_key()
                default_key = ApiKey(
                    user_id=admin_user.id,
                    name="Default Admin Master Key",
                    key_prefix=key_prefix,
                    secret_key=raw_key,
                    key_hash=key_hash,
                    is_active=True
                )
                session.add(default_key)
                await session.commit()

                print("-" * 68, flush=True)
                print(f" [DB Provisioning] Initial Admin User Created:", flush=True)
                print(f"   Username: {ADMIN_USERNAME}", flush=True)
                print(f"   Password: {ADMIN_PASSWORD}", flush=True)
                print(f"   API Key:  {raw_key}", flush=True)
                print("-" * 68, flush=True)
            else:
                key_stmt = select(ApiKey).where(ApiKey.user_id == existing_admin.id, ApiKey.is_active == True)
                key_result = await session.execute(key_stmt)
                admin_key = key_result.scalar_one_or_none()
                if not admin_key:
                    raw_key, key_prefix, key_hash = generate_raw_api_key()
                    default_key = ApiKey(
                        user_id=existing_admin.id,
                        name="Default Admin Master Key",
                        key_prefix=key_prefix,
                        secret_key=raw_key,
                        key_hash=key_hash,
                        is_active=True
                    )
                    session.add(default_key)
                    await session.commit()
                logger.info(f"Admin user '{ADMIN_USERNAME}' verified with active API key.")

            # 2. Provision / Verify Demo (Standard) User
            demo_stmt = select(User).where(User.username == DEMO_USERNAME)
            demo_result = await session.execute(demo_stmt)
            existing_demo = demo_result.scalar_one_or_none()

            if not existing_demo:
                logger.info(f"Provisioning default standard user '{DEMO_USERNAME}'...")
                demo_user = User(
                    username=DEMO_USERNAME,
                    hashed_password=hash_password(DEMO_PASSWORD),
                    is_admin=False,
                    is_active=True,
                    created_version=config.VERSION
                )
                session.add(demo_user)
                await session.flush()

                raw_key, key_prefix, key_hash = generate_raw_api_key()
                demo_key = ApiKey(
                    user_id=demo_user.id,
                    name=f"Key for {DEMO_USERNAME}",
                    key_prefix=key_prefix,
                    secret_key=raw_key,
                    key_hash=key_hash,
                    is_active=True
                )
                session.add(demo_key)
                await session.commit()

                print("-" * 68, flush=True)
                print(f" [DB Provisioning] Initial Demo User Created:", flush=True)
                print(f"   Username: {DEMO_USERNAME}", flush=True)
                print(f"   Password: {DEMO_PASSWORD}", flush=True)
                print(f"   API Key:  {raw_key}", flush=True)
                print("-" * 68, flush=True)
            else:
                demo_key_stmt = select(ApiKey).where(ApiKey.user_id == existing_demo.id, ApiKey.is_active == True)
                demo_key_res = await session.execute(demo_key_stmt)
                if not demo_key_res.scalar_one_or_none():
                    raw_key, key_prefix, key_hash = generate_raw_api_key()
                    d_key = ApiKey(
                        user_id=existing_demo.id,
                        name=f"Key for {DEMO_USERNAME}",
                        key_prefix=key_prefix,
                        secret_key=raw_key,
                        key_hash=key_hash,
                        is_active=True
                    )
                    session.add(d_key)
                    await session.commit()
                logger.info(f"Demo user '{DEMO_USERNAME}' verified with active API key.")

        except Exception as e:
            logger.error(f"Database seeding error: {e}")
            await session.rollback()
            raise
