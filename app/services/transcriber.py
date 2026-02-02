"""Transcription service using OpenAI Whisper."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import ffmpeg
import whisper

from app.config import settings

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribe audio using OpenAI Whisper."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self._model = None

    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        output_format: str = "txt",
        language: Optional[str] = None,
    ) -> Dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            output_format: Output format (txt, srt, vtt, json)
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            Dictionary with transcription results
        """
        loop = asyncio.get_event_loop()
        result = loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio_path, output_format, language)
        )
        return result

    def _transcribe_sync(
        self,
        audio_path: Path,
        output_format: str = "txt",
        language: Optional[str] = None,
    ) -> Dict:
        """Synchronous transcription."""
        options = {}
        if language:
            options["language"] = language

        result = self.model.transcribe(str(audio_path), **options)

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
        }

    def get_audio_duration(self, video_path: Path) -> float:
        """Get the duration of an audio/video file in seconds."""
        try:
            probe = ffmpeg.probe(str(video_path))
            duration = float(probe["format"]["duration"])
            return duration
        except ffmpeg.Error as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0


class AudioExtractor:
    """Extract audio from video files."""

    def __init__(self):
        self.transcript_dir = Path(settings.TRANSCRIPT_DIR)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(
        self,
        video_path: Path,
        job_id: str,
        format: str = "wav",
    ) -> Path:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the video file
            job_id: Job ID for naming
            format: Audio format (wav, mp3, ogg)

        Returns:
            Path to the extracted audio file
        """
        output_path = self.transcript_dir / f"{job_id}.{format}"

        # Build ffmpeg command
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    vn=None,  # No video
                    acodec="pcm_s16le",  # WAV codec
                    ar=16000,  # 16kHz sample rate
                    ac=1,  # Mono
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            raise ValueError(f"Failed to extract audio: {e}")

        return output_path

    def cleanup(self, file_path: Path) -> bool:
        """Remove an audio file."""
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except OSError as e:
            logger.error(f"Failed to cleanup audio file {file_path}: {e}")
        return False
