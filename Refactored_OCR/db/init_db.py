import logging
from sqlalchemy import select
from .session import Base, engine, AsyncSessionLocal
from .models import User, ApiKey
from core.security import hash_password, generate_raw_api_key
import config

logger = logging.getLogger("init_db")

ADMIN_USERNAME = getattr(config, "ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = getattr(config, "ADMIN_PASSWORD", "admin123")


async def init_database() -> None:
    """Creates database tables and provisions initial admin user if not present."""
    logger.info("Initializing database schema...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

    async with AsyncSessionLocal() as session:
        try:
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

                # Optional initial key for admin
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
                logger.info(f"Admin user '{ADMIN_USERNAME}' is already provisioned.")
        except Exception as e:
            logger.error(f"Database seeding error: {e}")
            await session.rollback()
            raise
