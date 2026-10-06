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
    book_tasks = relationship("OCRBookTask", back_populates="user", cascade="all, delete-orphan")


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


class OCRBookTask(Base):
    """Persistent background task for multi-page book / document OCR processing."""
    __tablename__ = "ocr_book_tasks"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)
    file_type = Column(String(32), default="pdf", nullable=False)
    file_size = Column(Integer, default=0, nullable=False)

    # Task execution state: queued | processing | completed | completed_with_errors | failed | cancelled
    status = Column(String(32), default="queued", nullable=False, index=True)

    # OCR & Preprocessing Configuration
    primary_model = Column(String(64), nullable=False)
    secondary_model = Column(String(64), nullable=True)
    lang = Column(String(64), default="eng+ara+fas", nullable=False)
    preprocess = Column(Boolean, default=False, nullable=False)
    contrast = Column(Boolean, default=False, nullable=False)
    crop_whitespaces = Column(Boolean, default=False, nullable=False)
    scale = Column(Float, default=1.0, nullable=False)
    use_llm = Column(Boolean, default=False, nullable=False)
    prompt_mode = Column(String(32), default="classical", nullable=True)
    cooldown_seconds = Column(Float, default=1.0, nullable=False)

    # Page Tracking & Progress
    start_page = Column(Integer, default=1, nullable=False)
    end_page = Column(Integer, nullable=True)
    total_pages = Column(Integer, default=1, nullable=False)
    completed_pages = Column(Integer, default=0, nullable=False)
    failed_pages = Column(Integer, default=0, nullable=False)
    current_page = Column(Integer, nullable=True)

    # Timing & Diagnostics
    total_ocr_duration = Column(Float, default=0.0, nullable=False)
    total_llm_duration = Column(Float, default=0.0, nullable=False)
    error_message = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="book_tasks")
    pages = relationship(
        "OCRBookTaskPage",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="OCRBookTaskPage.page_number"
    )


class OCRBookTaskPage(Base):
    """Per-page checkpoint state and extracted text for a background book OCR task."""
    __tablename__ = "ocr_book_task_pages"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(64), ForeignKey("ocr_book_tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)

    # Page status: pending | processing | completed | failed
    status = Column(String(32), default="pending", nullable=False, index=True)
    retry_count = Column(Integer, default=0, nullable=False)
    extracted_text = Column(Text, nullable=True)
    ocr_duration = Column(Float, default=0.0, nullable=False)
    llm_duration = Column(Float, default=-1.0, nullable=False)
    last_error = Column(Text, nullable=True)

    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    task = relationship("OCRBookTask", back_populates="pages")

