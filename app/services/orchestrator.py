"""Main transcription service orchestrator."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Job, JobStatus, OutputFormat, Transcript, User
from app.services.downloader import VideoDownloader
from app.services.exporter import TranscriptExporter
from app.services.extractor import extract_video_info
from app.services.transcriber import AudioExtractor, TranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionOrchestrator:
    """Orchestrate the transcription pipeline."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.extractor = extract_video_info
        self.downloader = VideoDownloader()
        self.transcriber = TranscriptionService()
        self.audio_extractor = AudioExtractor()
        self.exporter = TranscriptExporter()

    async def process_job(self, job_id: uuid.UUID) -> None:
        """
        Process a transcription job.

        Args:
            job_id: ID of the job to process
        """
        # Get job from database
        result = await self.db.execute(
            select(Job).where(Job.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        try:
            # Update status to processing
            job.status = JobStatus.PROCESSING
            job.progress = 10
            await self.db.commit()

            # Step 1: Extract video URL from tweet
            job.progress = 20
            logger.info(f"Extracting video URL from: {job.tweet_url}")

            try:
                tweet_info = await self.extractor(job.tweet_url)
                job.video_url = tweet_info.video_url
            except Exception as e:
                logger.error(f"Failed to extract video: {e}")
                job.status = JobStatus.FAILED
                job.error_message = f"Failed to extract video: {e}"
                await self.db.commit()
                return

            # Step 2: Download video
            job.progress = 30

            def progress_hook(d):
                if d["status"] == "downloading":
                    percent = d.get("downloaded_bytes", 0) / d.get("total_bytes", 1) * 100
                    job.progress = 30 + int(percent * 0.3)

            video_path = await self.downloader.download(
                tweet_info.video_url,
                job_id,
                progress_hook,
            )

            # Step 3: Extract audio
            job.progress = 70
            audio_path = self.audio_extractor.extract_audio(video_path, str(job_id))

            # Step 4: Transcribe
            job.progress = 80
            result = self.transcriber.transcribe(audio_path)

            # Step 5: Export transcript
            job.progress = 95

            # Get output format from transcript table
            result = await self.db.execute(
                select(Transcript).where(Transcript.job_id == job_id)
            )
            transcript_record = result.scalar_one_or_none()
            output_format = transcript_record.output_format if transcript_record else OutputFormat.TXT

            output_path = self.exporter.export(
                text=result["text"],
                segments=result["segments"],
                job_id=str(job_id),
                output_format=output_format,
                duration_seconds=tweet_info.duration,
            )

            # Update transcript record
            if transcript_record:
                transcript_record.text_content = result["text"]
                transcript_record.file_path = str(output_path)
                transcript_record.file_size = self.exporter.get_file_size(output_path)
                transcript_record.duration_seconds = tweet_info.duration
                transcript_record.word_count = len(result["text"].split())

            # Cleanup temp files
            self.downloader.cleanup(video_path)
            self.audio_extractor.cleanup(audio_path)

            # Update job status
            job.status = JobStatus.COMPLETED
            job.progress = 100
            await self.db.commit()

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            await self.db.commit()


async def create_job(
    db: AsyncSession,
    user_id: uuid.UUID,
    tweet_url: str,
    output_format: OutputFormat = OutputFormat.TXT,
    webhook_url: Optional[str] = None,
) -> Job:
    """Create a new transcription job."""
    job = Job(
        user_id=user_id,
        tweet_url=tweet_url,
        status=JobStatus.PENDING,
        progress=0,
    )
    db.add(job)

    transcript = Transcript(
        user_id=user_id,
        job=job,
        output_format=output_format,
    )
    db.add(transcript)

    await db.commit()
    await db.refresh(job)

    return job
