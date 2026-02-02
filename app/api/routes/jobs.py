"""API routes for transcription jobs."""

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import Job, JobStatus, Transcript, User
from app.schemas import JobCreate, JobResponse, JobWithTranscript
from app.services.orchestrator import create_job
from app.workers.transcriber_worker import process_transcription_job

router = APIRouter()


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_transcription_job(
    job_data: JobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new transcription job.

    The job will be processed in the background.
    Use GET /jobs/{job_id} to check status.
    """
    # Check if user has credits
    if current_user.credits <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="No credits remaining. Please upgrade your plan.",
        )

    # Create job
    job = await create_job(
        db=db,
        user_id=current_user.id,
        tweet_url=str(job_data.tweet_url),
        output_format=job_data.output_format,
        webhook_url=str(job_data.webhook_url) if job_data.webhook_url else None,
    )

    # Deduct credit (or track for later)
    current_user.credits -= 1
    await db.commit()

    # Queue background task
    process_transcription_job.delay(str(job.id))

    return job


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all jobs for the current user."""
    result = await db.execute(
        select(Job)
        .where(Job.user_id == current_user.id)
        .order_by(Job.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    jobs = result.scalars().all()
    return jobs


@router.get("/jobs/{job_id}", response_model=JobWithTranscript)
async def get_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get job status and details."""
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    response = JobWithTranscript(
        id=job.id,
        tweet_url=job.tweet_url,
        video_url=job.video_url,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        transcript_url=None,
    )

    # Add transcript URL if completed
    if job.status == JobStatus.COMPLETED and job.transcript:
        response.transcript_url = f"/api/v1/transcripts/{job.transcript.id}/download"

    return response


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a job and its transcript."""
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Don't allow deleting running jobs
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a job that is currently processing",
        )

    await db.delete(job)
    await db.commit()
