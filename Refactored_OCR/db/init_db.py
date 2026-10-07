import logging
from sqlalchemy import select, text
from .session import Base, engine, AsyncSessionLocal
from .models import User, ApiKey, ExtractionHistory, OCRBookTask, OCRBookTaskPage
from core.security import hash_password, generate_raw_api_key
import config

logger = logging.getLogger("init_db")

ADMIN_USERNAME = getattr(config, "ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = getattr(config, "ADMIN_PASSWORD", "admin123")
DEMO_USERNAME = getattr(config, "DEMO_USERNAME", "demo")
DEMO_PASSWORD = getattr(config, "DEMO_PASSWORD", "demo123")

# List of incremental, backward-compatible column migrations:
# (table_name, column_name, postgres_sql, generic_or_sqlite_sql)
SCHEMA_MIGRATIONS = [
    (
        "ocr_book_tasks",
        "prompt_mode",
        "ALTER TABLE ocr_book_tasks ADD COLUMN IF NOT EXISTS prompt_mode VARCHAR(32) DEFAULT 'classical';",
        "ALTER TABLE ocr_book_tasks ADD COLUMN prompt_mode VARCHAR(32) DEFAULT 'classical'"
    ),
    (
        "ocr_book_tasks",
        "cooldown_seconds",
        "ALTER TABLE ocr_book_tasks ADD COLUMN IF NOT EXISTS cooldown_seconds FLOAT DEFAULT 1.0;",
        "ALTER TABLE ocr_book_tasks ADD COLUMN cooldown_seconds FLOAT DEFAULT 1.0"
    ),
    (
        "users",
        "created_version",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_version VARCHAR(32);",
        "ALTER TABLE users ADD COLUMN created_version VARCHAR(32)"
    ),
]


async def apply_schema_migrations(conn) -> None:
    """Applies incremental schema updates and column additions for existing databases.

    Ensures that newer application versions run smoothly against legacy database volumes
    without requiring manual DB intervention or data loss.
    """
    dialect = conn.dialect.name
    logger.info(f"Verifying database schema migrations (dialect: {dialect})...")

    for table_name, column_name, pg_sql, generic_sql in SCHEMA_MIGRATIONS:
        try:
            if dialect == "postgresql":
                await conn.execute(text(pg_sql))
            else:
                def get_cols(sync_conn):
                    from sqlalchemy import inspect
                    inspector = inspect(sync_conn)
                    if table_name in inspector.get_table_names():
                        return [c["name"] for c in inspector.get_columns(table_name)]
                    return []

                existing_cols = await conn.run_sync(get_cols)
                if existing_cols and column_name not in existing_cols:
                    await conn.execute(text(generic_sql))
                    logger.info(f"Added missing column '{column_name}' to table '{table_name}'.")
        except Exception as exc:
            logger.warning(
                f"Schema migration skipped or failed for table '{table_name}', column '{column_name}': {exc}"
            )


async def init_database() -> None:
    """Creates database tables, applies auto-migrations, and provisions initial admin and demo users."""
    logger.info("Initializing database schema...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await apply_schema_migrations(conn)
    except Exception as e:
        logger.error(f"Error creating tables or running migrations: {e}")
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
