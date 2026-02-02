"""Celery worker for background transcription tasks."""

import logging
from celery import Celery

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "x-transcript",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_transcription_job(self, job_id: str):
    """
    Process a transcription job in the background.

    Args:
        job_id: UUID of the job to process

    Returns:
        Dict with job status
    """
    import asyncio
    from app.database import async_session_maker
    from app.services.orchestrator import TranscriptionOrchestrator
    from app.models import JobStatus
    import uuid

    try:
        job_uuid = uuid.UUID(job_id)

        async def _process():
            async with async_session_maker() as db:
                orchestrator = TranscriptionOrchestrator(db)
                await orchestrator.process_job(job_uuid)

        asyncio.run(_process())

        return {"status": "completed", "job_id": job_id}

    except Exception as e:
        logger.exception(f"Task failed: {e}")

        # Try to update job status in DB
        try:
            async def _update_failed():
                async with async_session_maker() as db:
                    from sqlalchemy import select
                    from app.models import Job

                    result = await db.execute(
                        select(Job).where(Job.id == job_uuid)
                    )
                    job = result.scalar_one_or_none()
                    if job:
                        job.status = JobStatus.FAILED
                        job.error_message = str(e)
                        await db.commit()

            asyncio.run(_update_failed())
        except Exception:
            pass

        # Retry on certain errors
        if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e)

        return {"status": "failed", "job_id": job_id, "error": str(e)}


@celery_app.task
def send_webhook_notification(webhook_url: str, job_id: str, status: str):
    """Send webhook notification when job completes."""
    import httpx
    import uuid

    try:
        payload = {
            "event": "job_completed",
            "job_id": job_id,
            "status": status,
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        logger.info(f"Webhook sent for job {job_id}")
        return {"status": "sent"}

    except Exception as e:
        logger.error(f"Webhook failed: {e}")
        return {"status": "failed", "error": str(e)}
