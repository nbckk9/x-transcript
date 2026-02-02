#!/usr/bin/env python3
"""
X-Transcript CLI - Transcribe X (Twitter) videos using OpenAI Whisper.

Usage:
    uv run python cli.py <tweet_url>
    uv run xtranscript-cli <tweet_url>
"""

import sys
import uuid
from pathlib import Path


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe X (Twitter) videos using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python cli.py "https://x.com/user/status/1234567890"
    uv run xtranscript-cli "https://x.com/user/status/1234567890"
        """,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://x.com/AlexFinn/status/2017991866306977944",
        help="URL of the X tweet with video",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: tiny)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: transcripts/<job_id>.txt)",
    )

    args = parser.parse_args()
    run_pipeline(args.url, args.model, args.output)


def run_pipeline(url: str, model: str = "tiny", output_path: str = None):
    """Run the full transcription pipeline."""
    print(f"🎬 X-Transcript CLI")
    print(f"   URL: {url}")
    print(f"   Model: {model}\n")

    import yt_dlp
    import ffmpeg

    job_id = uuid.uuid4()
    base_dir = Path(__file__).parent
    storage_dir = base_dir / "storage"
    transcript_dir = base_dir / "transcripts"
    storage_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
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
            print(f"   ✓ {video_path.name} ({size_mb:.1f} MB)")
        else:
            print("   ✗ Download failed")
            return
    except Exception as e:
        print(f"   ✗ Error: {e}")
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
        print(f"   ✓ {audio_path.name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Step 3: Transcribe
    print(f"\n3️⃣ Transcribing with Whisper ({model})...")
    try:
        import whisper

        print("   Loading model...")
        whisper_model = whisper.load_model(model)
        print("   Transcribing...")
        result = whisper_model.transcribe(str(audio_path))
        word_count = len(result["text"].split())
        print(f"   ✓ {word_count} words")
    except ImportError:
        print("   ⚠️ Whisper not installed")
        print("   Run: uv add openai-whisper")
        print(f"\n✅ Pipeline complete. Audio saved: {audio_path}")
        return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Step 4: Export
    print("\n4️⃣ Exporting...")
    if output_path is None:
        output_path = transcript_dir / f"{job_id}.txt"

    try:
        Path(output_path).write_text(result["text"], encoding="utf-8")
        print(f"   ✓ {output_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Cleanup
    print("\n🧹 Cleanup...")
    video_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    print("   ✓ Done")

    # Preview
    print("\n--- Preview ---")
    preview = result["text"][:300].strip()
    if len(result["text"]) > 300:
        preview += "..."
    print(preview)
    print("---")

    print(f"\n✅ Complete! {output_path}")


if __name__ == "__main__":
    main()
