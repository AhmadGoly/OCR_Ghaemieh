import os
import io
import gc
import json
import time
import uuid
import html
import zipfile
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

import config
from db.session import AsyncSessionLocal
from db.models import OCRBookTask, OCRBookTaskPage, ExtractionHistory
from services.queue_manager import queue_manager
from utils.pdf_utils import PDFUtils

logger = logging.getLogger("task_manager")

LOCAL_GPU_MODELS = {"qwen", "varco"}


class BookTaskManager:
    """
    Asynchronous Background Task Manager for Multi-Page Books & Batch OCR Documents.

    Key Capabilities:
    - Memory-safe single-page streaming (prevents RAM OOM on 500+ page books).
    - Database-backed per-page checkpointing (auto-resumes interrupted tasks on server restart).
    - Smart per-page retry with exponential backoff (isolates corrupt pages without failing the book).
    - Configurable inter-page rest/cooldown period to prevent GPU thermal throttling & API overload.
    - Multi-format book export (TXT, Markdown, Printable RTL HTML, JSON, ZIP).
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._active_tasks: Set[str] = set()
        self._queued_tasks: Set[str] = set()
        self._cancelled_tasks: Set[str] = set()
        self._workers: List[asyncio.Task] = []
        self._ocr_service = None
        self._ocr_executor = None
        self._running: bool = False

    def ensure_storage_dir(self) -> str:
        """Ensure the persistent task file storage directory exists."""
        storage_dir = getattr(
            config,
            "TASK_STORAGE_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "tasks")
        )
        os.makedirs(storage_dir, exist_ok=True)
        return storage_dir

    async def start(self, ocr_service, ocr_executor) -> None:
        """Start background worker loops and resume any unfinished tasks from database."""
        if self._running:
            return
        self._ocr_service = ocr_service
        self._ocr_executor = ocr_executor
        self._running = True
        self.ensure_storage_dir()

        worker_count = max(1, int(getattr(config, "TASK_WORKER_CONCURRENCY", 2)))
        for idx in range(worker_count):
            worker_task = asyncio.create_task(self._worker_loop(idx + 1))
            self._workers.append(worker_task)

        logger.info(f"Started {worker_count} background Book OCR Task worker(s).")
        await self._resume_unfinished_tasks()

    async def stop(self) -> None:
        """Gracefully stop background workers during application shutdown."""
        self._running = False
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _resume_unfinished_tasks(self) -> None:
        """Find any tasks interrupted by a previous server stop/crash and re-enqueue them."""
        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(OCRBookTask)
                    .options(selectinload(OCRBookTask.pages))
                    .where(OCRBookTask.status.in_(["queued", "processing"]))
                    .order_by(OCRBookTask.created_at.asc())
                )
                result = await session.execute(stmt)
                tasks = result.scalars().all()

                for task in tasks:
                    # Reset any page that was mid-processing back to pending
                    for p in task.pages:
                        if p.status == "processing":
                            p.status = "pending"
                    task.status = "queued"
                    task.current_page = None
                await session.commit()

                for task in tasks:
                    await self.enqueue_task(task.id)
                    logger.info(f"Auto-resumed unfinished book task: {task.id} ({task.filename})")
        except Exception as e:
            logger.error(f"Failed to resume unfinished book tasks: {e}")

    async def enqueue_task(self, task_id: str) -> None:
        """Enqueue a task ID for background execution if not already queued or active."""
        self._cancelled_tasks.discard(task_id)
        if task_id in self._queued_tasks or task_id in self._active_tasks:
            return
        self._queued_tasks.add(task_id)
        await self._queue.put(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Signal cancellation for a queued or running task."""
        self._cancelled_tasks.add(task_id)
        async with AsyncSessionLocal() as session:
            task = await session.get(OCRBookTask, task_id)
            if not task:
                return False
            if task.status in ("queued", "processing"):
                task.status = "cancelled"
                task.current_page = None
                await session.commit()
                return True
        return False

    async def _worker_loop(self, worker_id: int) -> None:
        """Background worker loop that pulls book tasks from the queue and processes them."""
        while self._running:
            try:
                task_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            self._queued_tasks.discard(task_id)
            if task_id in self._cancelled_tasks:
                self._queue.task_done()
                continue

            self._active_tasks.add(task_id)
            try:
                await self._process_task(task_id, worker_id)
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] Unhandled error processing task {task_id}: {e}")
            finally:
                self._active_tasks.discard(task_id)
                self._queue.task_done()

    async def _process_task(self, task_id: str, worker_id: int) -> None:
        """Process a book task page-by-page with checkpointing, retry, and cooldown."""
        async with AsyncSessionLocal() as session:
            stmt = (
                select(OCRBookTask)
                .options(selectinload(OCRBookTask.pages))
                .where(OCRBookTask.id == task_id)
            )
            res = await session.execute(stmt)
            task = res.scalar_one_or_none()
            if not task:
                return

            if task_id in self._cancelled_tasks or task.status == "cancelled":
                task.status = "cancelled"
                task.current_page = None
                await session.commit()
                return

            if not task.file_path or not os.path.exists(task.file_path):
                task.status = "failed"
                task.error_message = "فایل مبدا کتاب روی سرور یافت نشد یا حذف شده است."
                task.current_page = None
                await session.commit()
                return

            task.status = "processing"
            if not task.started_at:
                task.started_at = datetime.now(timezone.utc)
            task.error_message = None
            await session.commit()

            # Determine pages needing work (pending or failed)
            pending_pages = [
                p for p in sorted(task.pages, key=lambda x: x.page_number)
                if p.status in ("pending", "failed", "processing")
            ]

            max_retries = max(1, int(getattr(config, "TASK_MAX_PAGE_RETRIES", 3)))
            cooldown_sec = max(
                0.0,
                float(task.cooldown_seconds if task.cooldown_seconds is not None else getattr(config, "TASK_PAGE_COOLDOWN_SECONDS", 1.0))
            )
            is_gpu = (
                task.primary_model in LOCAL_GPU_MODELS
                or (task.use_llm and task.secondary_model in LOCAL_GPU_MODELS)
            )

            total_pending = len(pending_pages)

            for idx, page_obj in enumerate(pending_pages):
                # Check if cancelled before starting page
                if task_id in self._cancelled_tasks:
                    task.status = "cancelled"
                    task.current_page = None
                    await session.commit()
                    return

                # Refresh task status in case cancelled via DB
                await session.refresh(task)
                if task.status == "cancelled":
                    task.current_page = None
                    await session.commit()
                    return

                page_num = page_obj.page_number
                task.current_page = page_num
                page_obj.status = "processing"
                await session.commit()

                page_succeeded = False
                for attempt in range(1, max_retries + 1):
                    if task_id in self._cancelled_tasks:
                        break

                    try:
                        # Acquire concurrency slot so background book tasks coordinate with live users
                        async with queue_manager.acquire_slot(request=None, user=None, is_gpu_model=is_gpu):
                            loop = asyncio.get_running_loop()

                            def _extract_single_page() -> dict:
                                if task.file_type == "pdf":
                                    img = PDFUtils.render_single_page(task.file_path, page_num)
                                    if img is None:
                                        raise RuntimeError(f"امکان رندر تصویر صفحه {page_num} از فایل PDF وجود ندارد.")
                                else:
                                    img = Image.open(task.file_path).convert("RGB")

                                try:
                                    res_dict = self._ocr_service.process_image(
                                        image=img,
                                        primary_model_name=task.primary_model,
                                        secondary_model_name=task.secondary_model if task.use_llm else None,
                                        lang=task.lang,
                                        preprocess=task.preprocess,
                                        contrast=task.contrast,
                                        scale=task.scale,
                                        crop_whitespaces=task.crop_whitespaces,
                                        use_llm=task.use_llm
                                    )
                                    # Drop base64 images to save RAM
                                    res_dict.pop("original_image", None)
                                    res_dict.pop("processed_image", None)
                                    return res_dict
                                finally:
                                    try:
                                        img.close()
                                    except Exception:
                                        pass

                            page_result = await loop.run_in_executor(self._ocr_executor, _extract_single_page)

                        # Save page checkpoint
                        page_obj.status = "completed"
                        page_obj.extracted_text = page_result.get("text", "")
                        page_obj.ocr_duration = float(page_result.get("ocr_duration", 0.0))
                        page_obj.llm_duration = float(page_result.get("llm_duration", -1.0))
                        page_obj.last_error = None
                        page_succeeded = True
                        break

                    except Exception as page_err:
                        err_text = str(page_err)
                        page_obj.retry_count = (page_obj.retry_count or 0) + 1
                        page_obj.last_error = f"تلاش {attempt}/{max_retries}: {err_text[:500]}"
                        await session.commit()
                        logger.warning(
                            f"[Task {task_id}] Page {page_num} failed (attempt {attempt}/{max_retries}): {err_text}"
                        )

                        if attempt < max_retries and task_id not in self._cancelled_tasks:
                            backoff_delay = min(15.0, 1.5 * (2 ** (attempt - 1)))
                            await asyncio.sleep(backoff_delay)

                if not page_succeeded:
                    if task_id in self._cancelled_tasks:
                        page_obj.status = "pending"
                        task.status = "cancelled"
                        task.current_page = None
                        await session.commit()
                        return
                    page_obj.status = "failed"

                # Update aggregate counters on parent task immediately
                self._recalculate_task_aggregates(task)
                await session.commit()

                # Periodic garbage collection to keep memory footprint flat on huge books
                if idx % 5 == 0:
                    gc.collect()

                # Cooldown / Rest period between pages so GPU/CPU/API does not overload
                if idx < total_pending - 1 and cooldown_sec > 0 and task_id not in self._cancelled_tasks:
                    await asyncio.sleep(cooldown_sec)

            # Finalize Task State
            self._recalculate_task_aggregates(task)
            task.current_page = None
            task.completed_at = datetime.now(timezone.utc)

            if task_id in self._cancelled_tasks or task.status == "cancelled":
                task.status = "cancelled"
            elif task.completed_pages == task.total_pages and task.failed_pages == 0:
                task.status = "completed"
                task.error_message = None
            elif task.completed_pages > 0 and task.failed_pages > 0:
                task.status = "completed_with_errors"
                task.error_message = f"{task.failed_pages} صفحه با خطا مواجه شد. می‌توانید صفحات ناموفق را مجدداً تلاش کنید یا نسخه فعلی را دانلود نمایید."
            else:
                task.status = "failed"
                first_err = next((p.last_error for p in task.pages if p.last_error), "خطا در استخراج صفحات کتاب")
                task.error_message = first_err

            # Record in ExtractionHistory for unified admin statistics
            if task.completed_pages > 0:
                hist = ExtractionHistory(
                    user_id=task.user_id,
                    filename=task.filename,
                    file_type=f"book_{task.file_type}",
                    pages_count=task.completed_pages,
                    primary_model=task.primary_model,
                    secondary_model=task.secondary_model,
                    use_llm=task.use_llm,
                    ocr_duration=task.total_ocr_duration,
                    llm_duration=task.total_llm_duration if task.total_llm_duration > 0 else -1.0
                )
                session.add(hist)

            await session.commit()
            logger.info(
                f"[Task {task_id}] Finished with status='{task.status}' "
                f"(completed={task.completed_pages}/{task.total_pages}, failed={task.failed_pages})"
            )

    @staticmethod
    def _recalculate_task_aggregates(task: OCRBookTask) -> None:
        """Recalculate completed/failed page counts and durations from task.pages."""
        completed = 0
        failed = 0
        ocr_sum = 0.0
        llm_sum = 0.0
        for p in task.pages:
            if p.status == "completed":
                completed += 1
                ocr_sum += float(p.ocr_duration or 0.0)
                if p.llm_duration and p.llm_duration > 0:
                    llm_sum += float(p.llm_duration)
            elif p.status == "failed":
                failed += 1
        task.completed_pages = completed
        task.failed_pages = failed
        task.total_ocr_duration = round(ocr_sum, 2)
        task.total_llm_duration = round(llm_sum, 2)

    def get_worker_metrics(self) -> dict:
        """Return live telemetry of background book workers."""
        return {
            "active_book_tasks": len(self._active_tasks),
            "queued_book_tasks": self._queue.qsize(),
            "worker_concurrency": getattr(config, "TASK_WORKER_CONCURRENCY", 2),
            "default_cooldown_seconds": getattr(config, "TASK_PAGE_COOLDOWN_SECONDS", 1.0),
            "max_page_retries": getattr(config, "TASK_MAX_PAGE_RETRIES", 3),
        }

    @staticmethod
    def serialize_task(task: OCRBookTask, include_pages: bool = False, username: Optional[str] = None) -> dict:
        """Convert an OCRBookTask ORM instance into a rich JSON-serializable status dictionary."""
        total = max(1, task.total_pages or 1)
        done_or_failed = (task.completed_pages or 0) + (task.failed_pages or 0)
        progress_pct = round(min(100.0, (done_or_failed / total) * 100.0), 1)
        if task.status == "completed":
            progress_pct = 100.0

        # Estimate remaining time (ETA) in seconds when processing
        eta_seconds = None
        remaining_pages = max(0, total - done_or_failed)
        if task.status == "processing" and remaining_pages > 0:
            if task.completed_pages and task.completed_pages > 0:
                avg_page_time = (
                    (task.total_ocr_duration + max(0.0, task.total_llm_duration)) / task.completed_pages
                ) + (task.cooldown_seconds or 0.0)
            else:
                avg_page_time = 4.0 + (task.cooldown_seconds or 0.0)
            eta_seconds = int(round(remaining_pages * avg_page_time))

        data = {
            "task_id": task.id,
            "user_id": task.user_id,
            "username": username or (task.user.username if "user" in task.__dict__ and task.user else None),
            "filename": task.filename,
            "file_type": task.file_type,
            "file_size": task.file_size,
            "status": task.status,
            "primary_model": task.primary_model,
            "secondary_model": task.secondary_model,
            "lang": task.lang,
            "preprocess": task.preprocess,
            "contrast": task.contrast,
            "crop_whitespaces": task.crop_whitespaces,
            "scale": task.scale,
            "use_llm": task.use_llm,
            "cooldown_seconds": task.cooldown_seconds,
            "start_page": task.start_page,
            "end_page": task.end_page,
            "total_pages": task.total_pages,
            "completed_pages": task.completed_pages,
            "failed_pages": task.failed_pages,
            "pending_pages": max(0, total - done_or_failed),
            "current_page": task.current_page,
            "progress_percent": progress_pct,
            "eta_seconds": eta_seconds,
            "total_ocr_duration": round(task.total_ocr_duration or 0.0, 2),
            "total_llm_duration": round(task.total_llm_duration or 0.0, 2),
            "error_message": task.error_message,
            "can_download": (task.completed_pages or 0) > 0,
            "can_retry": (task.failed_pages or 0) > 0 or task.status in ("failed", "completed_with_errors", "cancelled"),
            "can_cancel": task.status in ("queued", "processing"),
            "download_urls": {
                "txt": f"/api/tasks/{task.id}/download?format=txt",
                "html": f"/api/tasks/{task.id}/download?format=html",
                "md": f"/api/tasks/{task.id}/download?format=md",
                "json": f"/api/tasks/{task.id}/download?format=json",
                "zip": f"/api/tasks/{task.id}/download?format=zip",
            },
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }

        if include_pages and "pages" in task.__dict__ and task.pages is not None:
            data["pages"] = [
                {
                    "page_number": p.page_number,
                    "status": p.status,
                    "retry_count": p.retry_count,
                    "ocr_duration": round(p.ocr_duration or 0.0, 2),
                    "llm_duration": round(p.llm_duration or -1.0, 2),
                    "last_error": p.last_error,
                    "char_count": len(p.extracted_text) if p.extracted_text else 0,
                    "preview": (p.extracted_text[:220] + "...") if p.extracted_text and len(p.extracted_text) > 220 else (p.extracted_text or ""),
                    "text": p.extracted_text or "",
                }
                for p in sorted(task.pages, key=lambda x: x.page_number)
            ]

        return data

    @staticmethod
    def build_export_payload(task: OCRBookTask, export_format: str = "txt") -> Tuple[bytes, str, str]:
        """
        Build downloadable book output in the requested format (txt, md, html, json, zip).
        Returns (content_bytes, media_type, download_filename).
        """
        base_name = os.path.splitext(task.filename or "book")[0]
        safe_base = "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in base_name).strip() or f"book_{task.id[:8]}"
        sorted_pages = sorted(task.pages, key=lambda p: p.page_number)

        if export_format == "json":
            payload = {
                "task_id": task.id,
                "filename": task.filename,
                "status": task.status,
                "primary_model": task.primary_model,
                "secondary_model": task.secondary_model,
                "use_llm": task.use_llm,
                "total_pages": task.total_pages,
                "completed_pages": task.completed_pages,
                "failed_pages": task.failed_pages,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "pages": [
                    {
                        "page": p.page_number,
                        "status": p.status,
                        "text": p.extracted_text or "",
                        "ocr_duration": round(p.ocr_duration or 0.0, 2),
                        "llm_duration": round(p.llm_duration or -1.0, 2),
                        "retry_count": p.retry_count,
                        "error": p.last_error,
                    }
                    for p in sorted_pages
                ],
            }
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            return raw, "application/json; charset=utf-8", f"{safe_base}_ocr.json"

        if export_format == "md":
            lines = [
                f"# {task.filename}",
                "",
                f"- **شناسه وظیفه**: `{task.id}`",
                f"- **مدل استخراج**: `{task.primary_model}`" + (f" + `{task.secondary_model}` (LLM)" if task.use_llm else ""),
                f"- **تعداد صفحات استخراج‌شده**: {task.completed_pages} از {task.total_pages}",
                "",
                "---",
                "",
            ]
            for p in sorted_pages:
                lines.append(f"## صفحه {p.page_number}")
                lines.append("")
                if p.status == "completed":
                    lines.append(p.extracted_text or "")
                elif p.status == "failed":
                    lines.append(f"> **[خطا در استخراج صفحه {p.page_number}]** — {p.last_error or 'ناموفق'}")
                else:
                    lines.append(f"> *[صفحه {p.page_number} هنوز پردازش نشده است]*")
                lines.append("")
                lines.append("---")
                lines.append("")
            raw = "\n".join(lines).encode("utf-8")
            return raw, "text/markdown; charset=utf-8", f"{safe_base}_ocr.md"

        if export_format == "html":
            page_blocks = []
            for p in sorted_pages:
                if p.status == "completed":
                    escaped_txt = html.escape(p.extracted_text or "").replace("\n", "<br>\n")
                    body_html = f'<div class="page-text">{escaped_txt}</div>'
                elif p.status == "failed":
                    body_html = f'<div class="page-error">خطا در استخراج صفحه {p.page_number}: {html.escape(p.last_error or "")}</div>'
                else:
                    body_html = f'<div class="page-pending">صفحه {p.page_number} در انتظار پردازش...</div>'

                page_blocks.append(f"""
                <section class="book-page" id="page-{p.page_number}">
                    <div class="page-header">
                        <span>صفحه {p.page_number}</span>
                        <span class="page-meta">{round(p.ocr_duration or 0, 2)} ثانیه</span>
                    </div>
                    {body_html}
                </section>
                """)

            html_doc = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(task.filename)} - خروجی استخراج متن قائمیه</title>
    <style>
        body {{
            font-family: 'Vazirmatn', 'Tahoma', sans-serif;
            background: #f4f4f5;
            color: #18181b;
            margin: 0;
            padding: 24px;
            line-height: 2;
        }}
        .book-header {{
            max-width: 850px;
            margin: 0 auto 24px auto;
            background: #ffffff;
            border-right: 5px solid #e11d48;
            padding: 20px 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .book-header h1 {{ margin: 0 0 8px 0; font-size: 20px; color: #be123c; }}
        .book-header p {{ margin: 4px 0; font-size: 13px; color: #52525b; }}
        .book-page {{
            max-width: 850px;
            margin: 0 auto 20px auto;
            background: #ffffff;
            padding: 28px 32px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            page-break-after: always;
        }}
        .page-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #e4e4e7;
            padding-bottom: 8px;
            margin-bottom: 16px;
            font-weight: bold;
            font-size: 13px;
            color: #e11d48;
        }}
        .page-meta {{ color: #71717a; font-weight: normal; }}
        .page-text {{ font-size: 15px; text-align: justify; white-space: normal; }}
        .page-error {{ color: #b91c1c; background: #fef2f2; padding: 12px; border-radius: 8px; font-size: 13px; }}
        @media print {{
            body {{ background: #fff; padding: 0; }}
            .book-page {{ box-shadow: none; border: 1px solid #e4e4e7; margin-bottom: 0; }}
        }}
    </style>
</head>
<body>
    <div class="book-header">
        <h1>{html.escape(task.filename)}</h1>
        <p>مدل استخراج: <strong>{html.escape(task.primary_model)}</strong> | صفحات تکمیل‌شده: <strong>{task.completed_pages} از {task.total_pages}</strong></p>
    </div>
    {"".join(page_blocks)}
</body>
</html>"""
            raw = html_doc.encode("utf-8")
            return raw, "text/html; charset=utf-8", f"{safe_base}_ocr.html"

        # Build plain text representation (used for both txt and zip)
        txt_blocks = []
        for p in sorted_pages:
            header = f"--- صفحه {p.page_number} ---"
            if p.status == "completed":
                txt_blocks.append(f"{header}\n{p.extracted_text or ''}\n")
            elif p.status == "failed":
                txt_blocks.append(f"{header}\n[خطا در استخراج این صفحه: {p.last_error or 'ناموفق'}]\n")
            else:
                txt_blocks.append(f"{header}\n[این صفحه هنوز پردازش نشده است]\n")
        full_txt = "\n".join(txt_blocks)

        if export_format == "zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{safe_base}_full_book.txt", full_txt.encode("utf-8"))
                for p in sorted_pages:
                    page_filename = f"pages/page_{p.page_number:04d}.txt"
                    page_content = p.extracted_text if p.status == "completed" else f"[STATUS: {p.status}] {p.last_error or ''}"
                    zf.writestr(page_filename, (page_content or "").encode("utf-8"))
                meta_bytes, _, _ = BookTaskManager.build_export_payload(task, "json")
                zf.writestr("metadata.json", meta_bytes)
            return buf.getvalue(), "application/zip", f"{safe_base}_ocr_package.zip"

        # Default: TXT
        return full_txt.encode("utf-8"), "text/plain; charset=utf-8", f"{safe_base}_ocr.txt"


# Singleton Book Task Manager
task_manager = BookTaskManager()
