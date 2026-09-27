import os
import uuid
import urllib.parse
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

import config
from db.session import get_db
from db.models import User, OCRBookTask, OCRBookTaskPage
from core.deps import get_current_user
from services.task_manager import task_manager
from utils.pdf_utils import PDFUtils

router = APIRouter(prefix="/api/tasks", tags=["Background Book OCR Tasks"])


def _is_model_enabled(model_name: str) -> bool:
    """Verify if the requested OCR engine is enabled on this server."""
    mapping = {
        "tesseract": getattr(config, "LOAD_TESSERACT", True),
        "docling": getattr(config, "LOAD_DOCLING", False),
        "qwen": getattr(config, "LOAD_QWEN", False),
        "varco": getattr(config, "LOAD_VARCO", False),
        "olmocr_2b": getattr(config, "LOAD_OLMOCR_2B", True),
        "gemma4": getattr(config, "LOAD_GEMMA4", True),
    }
    return bool(mapping.get(model_name, False))


def _validate_languages(lang_str: Optional[str]) -> str:
    """Validate "+"-separated Tesseract/OCR language codes."""
    if not lang_str:
        return config.DEFAULT_LANG
    parts = [p.strip() for p in lang_str.split("+") if p.strip()]
    for part in parts:
        if part not in config.ACCEPTED_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"زبان '{part}' پشتیبانی نمی‌شود. زبان‌های مجاز: {config.ACCEPTED_LANGUAGES}"
            )
    return "+".join(parts)


@router.post(
    "/book",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a Book or Multi-Page Document for Background OCR"
)
async def create_book_ocr_task(
    file: UploadFile = File(..., description="فایل کتاب (PDF) یا تصویر سند"),
    model: str = Form(config.DEFAULT_MODEL, description="مدل اصلی استخراج متن"),
    secondary_model: Optional[str] = Form(None, description="مدل کمکی دوم در صورت فعال بودن LLM"),
    lang: Optional[str] = Form(config.DEFAULT_LANG, description="زبان‌های سند (مثلاً eng+ara+fas)"),
    start_page: int = Form(1, ge=1, description="صفحه شروع (برای PDF)"),
    end_page: Optional[int] = Form(None, description="صفحه پایان (اختیاری)"),
    preprocess: bool = Form(config.DEFAULT_PREPROCESS, description="پیش‌پردازش خاکستری"),
    contrast: bool = Form(config.DEFAULT_CONTRAST, description="بهبود کنتراست CLAHE"),
    crop_whitespaces: bool = Form(False, description="برش حاشیه‌های سفید"),
    scale: float = Form(config.DEFAULT_SCALE, ge=0.2, le=4.0, description="ضریب مقیاس تصویر"),
    use_llm: bool = Form(config.DEFAULT_USE_LLM, description="تصحیح و ادغام با LLM"),
    cooldown_seconds: float = Form(
        config.TASK_PAGE_COOLDOWN_SECONDS,
        ge=0.0,
        le=60.0,
        description="زمان استراحت بین صفحات (ثانیه) جهت جلوگیری از فشار بیش از حد به سرور"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a persistent, resumable background OCR task for a full book (PDF) or image.

    - Streams pages one-by-one in the background to prevent RAM exhaustion.
    - Checkpoints every completed page to the database.
    - Automatically retries failed pages up to `TASK_MAX_PAGE_RETRIES` times.
    - Applies `cooldown_seconds` rest time between pages so the service stays healthy.
    """
    model = (model or config.DEFAULT_MODEL).strip().lower()
    if model not in config.ACCEPTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"مدل '{model}' نامعتبر است. مدل‌های مجاز: {config.ACCEPTED_MODELS}"
        )
    if not _is_model_enabled(model):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Primary model '{model}' is not active on this server."
        )

    sec_model_clean: Optional[str] = None
    if secondary_model and secondary_model.strip():
        sec_model_clean = secondary_model.strip().lower()
        if sec_model_clean not in config.ACCEPTED_MODELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"مدل دوم '{sec_model_clean}' نامعتبر است."
            )
        if use_llm and not _is_model_enabled(sec_model_clean):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Secondary model '{sec_model_clean}' is not active on this server."
            )

    validated_lang = _validate_languages(lang)

    orig_filename = file.filename or "uploaded_book.pdf"
    ext = os.path.splitext(orig_filename)[1].lower()
    is_pdf = (ext == ".pdf") or (file.content_type == "application/pdf")
    is_img = ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp") or (
        file.content_type and file.content_type.startswith("image/")
    )

    if not is_pdf and not is_img:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="فرمت فایل پشتیبانی نمی‌شود. لطفاً فایل PDF یا تصویر معتبر بارگذاری نمایید."
        )

    task_id = str(uuid.uuid4())
    storage_dir = task_manager.ensure_storage_dir()
    safe_ext = ".pdf" if is_pdf else (ext if ext else ".png")
    stored_path = os.path.join(storage_dir, f"{task_id}{safe_ext}")

    # Stream uploaded file to disk
    file_size = 0
    try:
        with open(stored_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out_f.write(chunk)
                file_size += len(chunk)
    except Exception as e:
        if os.path.exists(stored_path):
            os.unlink(stored_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطا در ذخیره‌سازی فایل کتاب روی سرور: {e}"
        )

    # Determine page numbers to schedule
    try:
        if is_pdf:
            doc_total_pages = PDFUtils.get_page_count(stored_path)
            s_page = max(1, start_page or 1)
            e_page = min(doc_total_pages, end_page) if (end_page and end_page > 0) else doc_total_pages
            if s_page > e_page:
                if os.path.exists(stored_path):
                    os.unlink(stored_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"بازه صفحات نامعتبر است: صفحه شروع ({s_page}) بزرگتر از کل صفحات کتاب ({doc_total_pages}) است."
                )
            page_numbers = list(range(s_page, e_page + 1))
        else:
            s_page = 1
            e_page = 1
            page_numbers = [1]
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(stored_path):
            os.unlink(stored_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"فایل PDF ارسالی معتبر نیست یا قابل خواندن نمی‌باشد: {e}"
        )

    task_obj = OCRBookTask(
        id=task_id,
        user_id=current_user.id,
        filename=orig_filename,
        file_path=stored_path,
        file_type="pdf" if is_pdf else "image",
        file_size=file_size,
        status="queued",
        primary_model=model,
        secondary_model=sec_model_clean if use_llm else None,
        lang=validated_lang,
        preprocess=preprocess,
        contrast=contrast,
        crop_whitespaces=crop_whitespaces,
        scale=scale,
        use_llm=use_llm,
        cooldown_seconds=cooldown_seconds,
        start_page=s_page,
        end_page=e_page,
        total_pages=len(page_numbers),
        completed_pages=0,
        failed_pages=0,
    )
    db.add(task_obj)
    await db.flush()

    for p_num in page_numbers:
        db.add(
            OCRBookTaskPage(
                task_id=task_id,
                page_number=p_num,
                status="pending",
                retry_count=0,
            )
        )

    await db.commit()

    # Load with pages for serialization
    stmt = (
        select(OCRBookTask)
        .options(selectinload(OCRBookTask.pages))
        .where(OCRBookTask.id == task_id)
    )
    res = await db.execute(stmt)
    created_task = res.scalar_one()

    # Enqueue in background worker
    await task_manager.enqueue_task(task_id)

    return {
        "status": "accepted",
        "message": f"کتاب «{orig_filename}» ({len(page_numbers)} صفحه) با موفقیت در صف پردازش پس‌زمینه ثبت شد.",
        "task": task_manager.serialize_task(created_task, include_pages=False, username=current_user.username),
    }


@router.get("", summary="List Status of Last N Background Book Tasks")
async def list_recent_tasks(
    n: Optional[int] = Query(
        None,
        ge=1,
        le=config.TASK_MAX_LIST_LIMIT,
        description="تعداد آخرین وظایف مورد نظر برای دریافت (N)"
    ),
    limit: int = Query(
        config.TASK_DEFAULT_LIST_LIMIT,
        ge=1,
        le=config.TASK_MAX_LIST_LIMIT,
        description="تعداد آخرین وظایف (پیش‌فرض ۱۰)"
    ),
    status_filter: Optional[str] = Query(None, alias="status", description="فیلتر بر اساس وضعیت وظیفه"),
    all_users: bool = Query(False, description="مخصوص مدیر: مشاهده وظایف تمامی کاربران"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve the status and live progress of the last `N` background OCR tasks.
    Pass `?n=10` or `?limit=10`.
    """
    effective_limit = n if n is not None else limit
    effective_limit = min(max(1, effective_limit), config.TASK_MAX_LIST_LIMIT)

    stmt = select(OCRBookTask).options(selectinload(OCRBookTask.user))
    if not (current_user.is_admin and all_users):
        stmt = stmt.where(OCRBookTask.user_id == current_user.id)

    if status_filter:
        stmt = stmt.where(OCRBookTask.status == status_filter.strip().lower())

    stmt = stmt.order_by(desc(OCRBookTask.created_at)).limit(effective_limit)
    result = await db.execute(stmt)
    tasks = result.scalars().all()

    return {
        "count": len(tasks),
        "requested_n": effective_limit,
        "worker_metrics": task_manager.get_worker_metrics(),
        "tasks": [
            task_manager.serialize_task(t, include_pages=False)
            for t in tasks
        ],
    }


@router.get("/{task_id}", summary="Get Detailed Status & Page Progress of a Specific Task")
async def get_task_detail(
    task_id: str,
    include_pages: bool = Query(True, description="شامل جزئیات تک‌تک صفحات"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full status, progress percentage, ETA, and per-page checkpoints for a task."""
    stmt = (
        select(OCRBookTask)
        .options(selectinload(OCRBookTask.pages), selectinload(OCRBookTask.user))
        .where(OCRBookTask.id == task_id)
    )
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="وظیفه مورد نظر یافت نشد.")

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="شما به این وظیفه دسترسی ندارید.")

    return task_manager.serialize_task(task, include_pages=include_pages)


@router.get("/{task_id}/download", summary="Download the OCRed Book Output")
async def download_task_book(
    task_id: str,
    format: str = Query(
        "txt",
        pattern="^(txt|html|md|json|zip)$",
        description="فرمت خروجی کتاب: txt, html, md, json, zip"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Download the extracted text of a completed (or partially completed) book task.
    Supports formats:
    - `txt`: Plain UTF-8 text separated by page numbers.
    - `html`: Printable RTL Persian book layout (can be saved as PDF in browser).
    - `md`: Structured Markdown book.
    - `json`: Complete JSON with page metadata and timings.
    - `zip`: Archive containing full book + individual page text files + metadata.json.
    """
    stmt = (
        select(OCRBookTask)
        .options(selectinload(OCRBookTask.pages))
        .where(OCRBookTask.id == task_id)
    )
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="وظیفه مورد نظر یافت نشد.")

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="شما اجازه دانلود این کتاب را ندارید.")

    if (task.completed_pages or 0) <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="هنوز هیچ صفحه‌ای از این کتاب تکمیل نشده است. لطفاً تا استخراج صفحات شکیبا باشید."
        )

    content_bytes, media_type, download_filename = task_manager.build_export_payload(task, format.lower())
    encoded_filename = urllib.parse.quote(download_filename)

    headers = {
        "Content-Disposition": f"attachment; filename=\"{ encoded_filename }\"; filename*=UTF-8''{ encoded_filename }",
        "Cache-Control": "no-cache",
    }
    return Response(content=content_bytes, media_type=media_type, headers=headers)


@router.post("/{task_id}/retry", summary="Retry Failed or Cancelled Pages of a Book Task")
async def retry_failed_pages(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Re-queue a task to retry ONLY its failed or unfinished pages without re-running already completed pages.
    """
    stmt = (
        select(OCRBookTask)
        .options(selectinload(OCRBookTask.pages), selectinload(OCRBookTask.user))
        .where(OCRBookTask.id == task_id)
    )
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="وظیفه مورد نظر یافت نشد.")

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="شما به این وظیفه دسترسی ندارید.")

    if task.status in ("queued", "processing"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="این وظیفه هم‌اکنون در صف یا در حال پردازش است."
        )

    if not task.file_path or not os.path.exists(task.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="فایل اصلی کتاب روی سرور یافت نشد؛ لطفاً کتاب را مجدداً بارگذاری نمایید."
        )

    pages_reset = 0
    for p in task.pages:
        if p.status != "completed":
            p.status = "pending"
            p.retry_count = 0
            p.last_error = None
            pages_reset += 1

    if pages_reset == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="تمامی صفحات این کتاب قبلاً با موفقیت استخراج شده‌اند."
        )

    task.status = "queued"
    task.failed_pages = 0
    task.error_message = None
    task.completed_at = None
    await db.commit()

    await task_manager.enqueue_task(task.id)

    return {
        "status": "requeued",
        "message": f"{pages_reset} صفحه باقی‌مانده/ناموفق مجدداً در صف پردازش قرار گرفتند.",
        "task": task_manager.serialize_task(task, include_pages=False),
    }


@router.post("/{task_id}/cancel", summary="Cancel a Queued or Running Book Task")
async def cancel_book_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel an active or queued background task while keeping all pages extracted so far."""
    stmt = select(OCRBookTask).where(OCRBookTask.id == task_id)
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="وظیفه مورد نظر یافت نشد.")

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="شما به این وظیفه دسترسی ندارید.")

    await task_manager.cancel_task(task_id)
    await db.refresh(task)

    return {
        "status": "cancelled",
        "message": "پردازش وظیفه متوقف شد. صفحات استخراج‌شده تا این لحظه محفوظ و قابل دانلود هستند.",
    }


@router.delete("/{task_id}", summary="Delete a Book Task and its Stored Files")
async def delete_book_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a book task record and remove its uploaded file from server storage."""
    stmt = select(OCRBookTask).where(OCRBookTask.id == task_id)
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="وظیفه مورد نظر یافت نشد.")

    if task.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="شما اجازه حذف این وظیفه را ندارید.")

    if task.status in ("queued", "processing"):
        await task_manager.cancel_task(task_id)

    file_path = task.file_path
    await db.delete(task)
    await db.commit()

    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception:
            pass

    return {
        "status": "deleted",
        "message": "وظیفه و فایل‌های مرتبط با موفقیت حذف شدند.",
    }
