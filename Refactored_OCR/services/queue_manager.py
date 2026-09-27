import asyncio
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import Request, HTTPException, status
from db.models import User
import config


class OCRQueueManager:
    """
    High-Performance Asynchronous Queue & Concurrency Orchestrator for OCR tasks.
    
    Features:
    - Dedicated Semaphores for general concurrency and local GPU-heavy models (Qwen, Varco).
    - Client Disconnection Detection (cancels aborted jobs early before burning compute).
    - Queue Timeout and Backpressure (returns friendly 503 instead of indefinite socket hang).
    - Per-User Concurrency Guard (prevents a single user from starving server resources).
    - Real-Time Queue & Worker Telemetry.
    """

    def __init__(self):
        self.general_semaphore = asyncio.Semaphore(getattr(config, "MAX_CONCURRENT_OCR", 20))
        self.gpu_semaphore = asyncio.Semaphore(getattr(config, "LOCAL_GPU_CONCURRENCY_LIMIT", 2))
        self.user_active_jobs: Dict[int, int] = {}
        
        self.active_jobs: int = 0
        self.queued_jobs: int = 0
        self.gpu_active_jobs: int = 0
        self.gpu_queued_jobs: int = 0
        
        self.total_completed_jobs: int = 0
        self.total_failed_jobs: int = 0

    @asynccontextmanager
    async def acquire_slot(
        self,
        request: Request,
        user: User,
        is_gpu_model: bool = False
    ):
        """
        Safely acquires worker slots for an incoming OCR request with timeout and client liveness checks.
        """
        # 1. Early disconnect check
        if await request.is_disconnected():
            raise HTTPException(
                status_code=499,
                detail="ارتباط با کلاینت پیش از شروع پردازش قطع گردید."
            )

        # 2. Per-user active concurrency guard (exempt administrators)
        max_user_slots = getattr(config, "MAX_USER_CONCURRENT_OCR", 4)
        if not user.is_admin and self.user_active_jobs.get(user.id, 0) >= max_user_slots:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"تعداد پردازش‌های هم‌زمان شما به سقف مجاز ({max_user_slots}) رسیده است. لطفاً تا پایان پردازش‌های قبلی شکیبا باشید."
            )

        queue_timeout = getattr(config, "QUEUE_TIMEOUT_SECONDS", 90.0)
        gpu_slot_acquired = False
        general_slot_acquired = False

        self.queued_jobs += 1
        if is_gpu_model:
            self.gpu_queued_jobs += 1

        try:
            # 3. Wait to acquire general semaphore
            try:
                await asyncio.wait_for(self.general_semaphore.acquire(), timeout=queue_timeout)
                general_slot_acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="ظرفیت پردازش هم‌زمان سرور تکمیل است و زمان انتظار در صف پایان یافت. لطفاً لحظاتی دیگر مجدداً تلاش فرمایید."
                )

            # 4. If local GPU model, wait to acquire GPU semaphore
            if is_gpu_model:
                try:
                    await asyncio.wait_for(self.gpu_semaphore.acquire(), timeout=queue_timeout)
                    gpu_slot_acquired = True
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="کارت گرافیک سرور در حال حاضر مشغول پردازش سایر اسناد است. لطفاً لحظاتی بعد مجدداً تلاش کنید."
                    )

            # 5. Check if client disconnected while waiting in queue
            if await request.is_disconnected():
                raise HTTPException(
                    status_code=499,
                    detail="درخواست توسط کاربر لغو گردید."
                )

            # 6. Mark job active
            self.active_jobs += 1
            if is_gpu_model:
                self.gpu_active_jobs += 1
            self.user_active_jobs[user.id] = self.user_active_jobs.get(user.id, 0) + 1

            try:
                yield
                self.total_completed_jobs += 1
            except Exception:
                self.total_failed_jobs += 1
                raise
            finally:
                self.active_jobs = max(0, self.active_jobs - 1)
                if is_gpu_model:
                    self.gpu_active_jobs = max(0, self.gpu_active_jobs - 1)
                self.user_active_jobs[user.id] = max(0, self.user_active_jobs.get(user.id, 1) - 1)

        finally:
            self.queued_jobs = max(0, self.queued_jobs - 1)
            if is_gpu_model:
                self.gpu_queued_jobs = max(0, self.gpu_queued_jobs - 1)

            if gpu_slot_acquired:
                self.gpu_semaphore.release()
            if general_slot_acquired:
                self.general_semaphore.release()

    def get_metrics(self) -> dict:
        """Returns snapshot of real-time queue depth and worker load."""
        return {
            "status": "congested" if self.queued_jobs > 5 else ("busy" if self.queued_jobs > 0 else "optimal"),
            "active_jobs": self.active_jobs,
            "queued_jobs": self.queued_jobs,
            "max_concurrent_ocr": getattr(config, "MAX_CONCURRENT_OCR", 20),
            "gpu_active_jobs": self.gpu_active_jobs,
            "gpu_queued_jobs": self.gpu_queued_jobs,
            "max_gpu_concurrency": getattr(config, "LOCAL_GPU_CONCURRENCY_LIMIT", 2),
            "max_per_user_concurrency": getattr(config, "MAX_USER_CONCURRENT_OCR", 4),
            "total_completed_jobs": self.total_completed_jobs,
            "total_failed_jobs": self.total_failed_jobs,
            "queue_timeout_seconds": getattr(config, "QUEUE_TIMEOUT_SECONDS", 90.0)
        }


# Singleton Queue Manager
queue_manager = OCRQueueManager()
