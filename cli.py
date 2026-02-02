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
    # Basic transcription
    uv run python cli.py "https://x.com/user/status/1234567890"

    # With custom summary prompt
    uv run python cli.py "url" --summarize "5 key takeaways"

    # Custom LLM provider
    uv run python cli.py "url" --summarize --llm-provider anthropic --api-key "sk-..."

    # Custom model
    uv run python cli.py "url" --whisper-model base

    # Save to specific file
    uv run python cli.py "url" -o output.txt
        """,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://x.com/AlexFinn/status/2017991866306977944",
        help="URL of the X tweet with video",
    )
    parser.add_argument(
        "--whisper-model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: tiny)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: transcripts/<job_id>.txt)",
    )
    parser.add_argument(
        "--summarize",
        "-s",
        nargs="?",
        const="5 key takeaways from this video",
        metavar="PROMPT",
        help="Summarize with custom prompt (default: '5 key takeaways from this video')",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic", "groq", "ollama", "local"],
        help="LLM provider for summarization (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        help="LLM model (provider-specific, e.g., gpt-4o, claude-3-5-sonnet)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for LLM provider (or set <PROVIDER>_API_KEY env var)",
    )

    args = parser.parse_args()
    run_pipeline(
        args.url,
        args.whisper_model,
        args.output,
        args.summarize,
        args.llm_provider,
        args.llm_model,
        args.api_key,
    )


def run_pipeline(
    url: str,
    whisper_model: str = "tiny",
    output_path: str = None,
    summarize: str = None,
    llm_provider: str = "openai",
    llm_model: str = None,
    api_key: str = None,
):
    """Run the full transcription pipeline."""
    print(f"🎬 X-Transcript CLI")
    print(f"   URL: {url}")
    print(f"   Whisper: {whisper_model}")
    if summarize:
        print(f"   Summary: {summarize}")
        print(f"   LLM Provider: {llm_provider}")
    print()

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
    print(f"\n3️⃣ Transcribing with Whisper ({whisper_model})...")
    try:
        import whisper

        print("   Loading model...")
        whisper_model_obj = whisper.load_model(whisper_model)
        print("   Transcribing...")
        result = whisper_model_obj.transcribe(str(audio_path))
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

    # Step 4: Summarize (optional)
    summary = None
    if summarize:
        print(f"\n4️⃣ Summarizing with {llm_provider}...")
        summary = summarize_transcript(
            result["text"],
            summarize,
            llm_provider,
            llm_model,
            api_key,
        )
        print(f"\n   --- Summary ---")
        print(summary)
        print(f"   ---")

    # Step 5: Export
    print("\n5️⃣ Exporting...")
    if output_path is None:
        output_path = transcript_dir / f"{job_id}.txt"

    try:
        content = result["text"]
        if summary:
            content = f"TRANSCRIPT:\n{result['text']}\n\nSUMMARY ({summarize}):\n{summary}"
        Path(output_path).write_text(content, encoding="utf-8")
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
    if not summarize:
        print("\n--- Preview ---")
        preview = result["text"][:300].strip()
        if len(result["text"]) > 300:
            preview += "..."
        print(preview)
        print("---")

    print(f"\n✅ Complete! {output_path}")


# ==================== LLM Provider Abstraction ====================

class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model

    def summarize(self, text: str, prompt: str) -> str:
        """Summarize text with given prompt."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key, model or "gpt-4o-mini")

    def summarize(self, text: str, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts. Be concise and structured."},
                {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text[:15000]}"},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key, model or "claude-sonnet-4-20250514")

    def summarize(self, text: str, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text[:15000]}"},
            ],
        )
        return response.content[0].text


class GroqProvider(LLMProvider):
    """Groq (Llama 3/4) provider - Best speed/price ratio."""

    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key, model or "meta-llama/llama-4-scout-17b-16e-instruct")

    def summarize(self, text: str, prompt: str) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts. Be concise and structured."},
                {"role": "user", "content": f"{prompt}\n\nTRANSCRIPT:\n{text[:15000]}"},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama local provider."""

    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key, model or "llama3.2")

    def summarize(self, text: str, prompt: str) -> str:
        import httpx

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": f"{prompt}\n\nTRANSCRIPT:\n{text[:15000]}",
                "stream": False,
            },
            timeout=120,
        )
        return response.json()["response"]


class LocalProvider(LLMProvider):
    """Generic local/CLI provider."""

    def __init__(self, api_key: str = None, model: str = None):
        super().__init__(api_key, model or "llamafile")

    def summarize(self, text: str, prompt: str) -> str:
        # Placeholder for local CLI tools
        return f"[Local provider '{self.model}' not configured - implement in providers/local.py]"


# Provider registry
LLM_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "local": LocalProvider,
}


def get_llm_provider(provider_name: str, api_key: str = None, model: str = None) -> LLMProvider:
    """Get LLM provider by name."""
    if provider_name not in LLM_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(LLM_PROVIDERS.keys())}")
    return LLM_PROVIDERS[provider_name](api_key=api_key, model=model)


def summarize_transcript(
    transcript: str,
    prompt: str,
    provider: str = "openai",
    model: str = None,
    api_key: str = None,
) -> str:
    """Summarize transcript using specified LLM provider."""
    import os

    # Get API key from args, env var, or .env
    env_key = f"{provider.upper()}_API_KEY"
    api_key = api_key or os.environ.get(env_key) or os.environ.get("OPENAI_API_KEY")

    if not api_key and provider not in ["ollama", "local"]:
        return f"⚠️ API key not set. Set {env_key} or use --api-key"

    try:
        llm = get_llm_provider(provider, api_key=api_key, model=model)
        return llm.summarize(transcript, prompt)
    except ImportError as e:
        return f"⚠️ Provider not installed: {e}\nRun: uv add {provider}"
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"


if __name__ == "__main__":
    main()
