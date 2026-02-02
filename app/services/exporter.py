"""Transcript export service."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.models import OutputFormat

logger = logging.getLogger(__name__)


class TranscriptExporter:
    """Export transcripts in various formats."""

    def __init__(self):
        self.output_dir = Path(settings.TRANSCRIPT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        text: str,
        segments: List[Dict],
        job_id: str,
        output_format: OutputFormat,
        duration_seconds: Optional[float] = None,
    ) -> Path:
        """
        Export transcript in the specified format.

        Args:
            text: Full transcript text
            segments: Whisper segments with timestamps
            job_id: Job ID for naming
            output_format: Output format
            duration_seconds: Duration of the video

        Returns:
            Path to the exported file
        """
        output_path = self.output_dir / f"{job_id}.{output_format.value}"

        if output_format == OutputFormat.TXT:
            content = self._export_txt(text)
        elif output_format == OutputFormat.SRT:
            content = self._export_srt(segments)
        elif output_format == OutputFormat.VTT:
            content = self._export_vtt(segments)
        elif output_format == OutputFormat.JSON:
            content = self._export_json(text, segments, duration_seconds)
        elif output_format == OutputFormat.MD:
            content = self._export_md(text, segments, duration_seconds)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        output_path.write_text(content, encoding="utf-8")
        return output_path

    def _export_txt(self, text: str) -> str:
        """Export as plain text."""
        return text.strip()

    def _export_srt(self, segments: List[Dict]) -> str:
        """Export as SRT subtitle format."""
        srt_lines = []
        for i, segment in enumerate(segments):
            start = self._format_srt_timestamp(segment.get("start", 0))
            end = self._format_srt_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            srt_lines.append(f"{i + 1}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(text)
            srt_lines.append("")
        return "\n".join(srt_lines)

    def _export_vtt(self, segments: List[Dict]) -> str:
        """Export as VTT subtitle format."""
        vtt_lines = ["WEBVTT", ""]
        for i, segment in enumerate(segments):
            start = self._format_vtt_timestamp(segment.get("start", 0))
            end = self._format_vtt_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            vtt_lines.append(f"{i + 1}")
            vtt_lines.append(f"{start} --> {end}")
            vtt_lines.append(text)
            vtt_lines.append("")
        return "\n".join(vtt_lines)

    def _export_json(
        self,
        text: str,
        segments: List[Dict],
        duration_seconds: Optional[float] = None,
    ) -> str:
        """Export as JSON."""
        data = {
            "transcript": text.strip(),
            "segments": segments,
            "duration_seconds": duration_seconds,
            "word_count": len(text.split()),
            "exported_at": datetime.utcnow().isoformat(),
            "format": "json",
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _export_md(
        self,
        text: str,
        segments: List[Dict],
        duration_seconds: Optional[float] = None,
    ) -> str:
        """Export as Markdown."""
        md_lines = [
            "# Transcript",
            "",
            f"**Exported:** {datetime.utcnow().isoformat()}",
            f"**Duration:** {self._format_duration(duration_seconds)}",
            f"**Words:** {len(text.split())}",
            "",
            "---",
            "",
            "## Full Text",
            "",
            text.strip(),
            "",
            "---",
            "",
            "## Timestamps",
            "",
        ]

        for segment in segments:
            start = self._format_srt_timestamp(segment.get("start", 0))
            text = segment.get("text", "").strip()
            md_lines.append(f"**{start}** {text}")

        return "\n".join(md_lines)

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Format seconds to VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _format_duration(self, seconds: Optional[float]) -> str:
        """Format duration in human-readable format."""
        if seconds is None:
            return "Unknown"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        if file_path.exists():
            return file_path.stat().st_size
        return 0
