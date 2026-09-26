from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from .session import Base

class User(Base):
    """User account entity for authentication and RBAC."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_version = Column(String(32), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    extractions = relationship("ExtractionHistory", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    """Programmatic API keys for microservice integrations."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False, default="Default Key")
    key_prefix = Column(String(16), nullable=False)
    key_hash = Column(String(255), nullable=False)
    secret_key = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    user = relationship("User", back_populates="api_keys")


class ExtractionHistory(Base):
    """Audit log of document OCR extractions."""
    __tablename__ = "extraction_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(64), nullable=False)
    pages_count = Column(Integer, default=1, nullable=False)
    primary_model = Column(String(64), nullable=False)
    secondary_model = Column(String(64), nullable=True)
    use_llm = Column(Boolean, default=False, nullable=False)
    ocr_duration = Column(Float, default=0.0)
    llm_duration = Column(Float, default=-1.0)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    user = relationship("User", back_populates="extractions")
