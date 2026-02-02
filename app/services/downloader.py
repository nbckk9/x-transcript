"""Video download service."""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import yt_dlp

from app.config import settings

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Download videos from URLs."""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def download(
        self,
        video_url: str,
        job_id: uuid.UUID,
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """
        Download a video file.

        Args:
            video_url: URL of the video to download
            job_id: Job ID for naming the file
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the downloaded video file
        """
        output_path = self.upload_dir / f"{job_id}.mp4"

        # Use yt-dlp for reliable download
        loop = asyncio.get_event_loop()
        download_info = await loop.run_in_executor(
            None,
            lambda: self._download_with_ytdlp(str(output_path), video_url, progress_callback)
        )

        return Path(download_info)

    def _download_with_ytdlp(
        self,
        output_path: str,
        video_url: str,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Download video using yt-dlp."""
        ytdlp_opts = {
            "outtmpl": output_path.replace(".mp4", ""),
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_callback] if progress_callback else [],
        }

        try:
            with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
                ydl.download([video_url])

            # yt-dlp adds extension, find the actual file
            expected_base = output_path.replace(".mp4", "")
            for ext in [".mp4", ".mkv", ".webm"]:
                actual_path = expected_base + ext
                if Path(actual_path).exists():
                    return actual_path

            raise FileNotFoundError(f"Downloaded file not found at {output_path}")

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            raise ValueError(f"Failed to download video: {e}")

    async def stream_download(
        self,
        video_url: str,
        chunk_size: int = 8192,
    ) -> AsyncGenerator[bytes, None]:
        """Stream download a video file."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", video_url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    yield chunk

    def cleanup(self, file_path: Path) -> bool:
        """Remove a downloaded file."""
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except OSError as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
        return False

    def get_file_size(self, file_path: Path) -> int:
        """Get the size of a file in bytes."""
        if file_path.exists():
            return file_path.stat().st_size
        return 0
