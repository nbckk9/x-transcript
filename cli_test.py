#!/usr/bin/env python3
"""
CLI test script for X video transcription using yt-dlp and OpenAI Whisper.

Usage:
    uv run python cli_test.py <tweet_url>
    uv run xtranscript-cli <tweet_url>
"""

import sys
import os
import uuid
from pathlib import Path


def main():
    """Run the transcription pipeline."""
    url = sys.argv[1] if len(sys.argv) > 1 else "https://x.com/AlexFinn/status/2017991866306977944"
    test_pipeline(url)


def test_pipeline(url: str):
    """Test the full transcription pipeline."""
    print(f"🎬 Testing with: {url}\n")

    import yt_dlp
    import ffmpeg

    job_id = uuid.uuid4()
    base_dir = Path(__file__).parent
    storage_dir = base_dir / "storage"
    transcript_dir = base_dir / "transcripts"
    storage_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download video using yt-dlp
    print("1️⃣ Downloading video...")
    video_template = storage_dir / f"{job_id}.%(ext)s"

    ytdlp_opts = {
        "outtmpl": str(video_template),
        "format": "best",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
            ydl.download([url])

        video_path = None
        for ext in ["mp4", "mkv", "webm", "m4a"]:
            candidate = storage_dir / f"{job_id}.{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path:
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ Downloaded: {video_path.name} ({size_mb:.1f} MB)")
        else:
            print("   ✗ File not found after download")
            return
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
        return

    # Step 2: Extract audio
    print("\n2️⃣ Extracting audio...")
    audio_path = storage_dir / f"{job_id}.wav"

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), vn=None, acodec="pcm_s16le", ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        size_kb = audio_path.stat().st_size / 1024
        print(f"   ✓ Audio: {audio_path.name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"   ✗ Audio extraction failed: {e}")
        return

    # Step 3: Transcribe with Whisper
    print("\n3️⃣ Transcribing with Whisper...")
    try:
        import whisper

        print("   Loading Whisper 'tiny' model...")
        model = whisper.load_model("tiny")
        print("   Transcribing...")
        result = model.transcribe(str(audio_path))
        word_count = len(result["text"].split())
        print(f"   ✓ Transcript: {word_count} words")
        print(f"\n   --- Preview ---")
        preview = result["text"][:500].strip()
        if len(result["text"]) > 500:
            preview += "..."
        print(preview)
    except ImportError:
        print("   ⚠️ Whisper not installed")
        print("   Run: uv add openai-whisper")
        print("\n✅ Pipeline complete (video + audio ready for transcription)")
        return
    except Exception as e:
        print(f"   ✗ Transcription failed: {e}")
        return

    # Step 4: Export
    print("\n4️⃣ Exporting...")
    output_path = transcript_dir / f"{job_id}.txt"
    try:
        output_path.write_text(result["text"], encoding="utf-8")
        print(f"   ✓ Saved: {output_path}")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return

    # Cleanup
    print("\n🧹 Cleanup...")
    video_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    print("   ✓ Done")

    print(f"\n✅ Complete! Transcript: {output_path}")


if __name__ == "__main__":
    main()
