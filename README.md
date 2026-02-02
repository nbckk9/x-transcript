# X-Transcript

Transcribe and summarize X (Twitter) videos using OpenAI Whisper + LLMs.

## Features

- 🎥 **Transcribe** videos from X (Twitter) URLs
- 📄 **Summarize** transcripts with AI (GPT-4o-mini, Llama 4, Claude)
- 💾 **Local processing** - Whisper runs locally (free, private)
- 🔧 **Works with existing transcripts** - Just pass a text file
- 🐳 **Docker-ready** for easy deployment

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nbckk9/x-transcript.git
cd x-transcript
uv sync
```

## CLI Usage

### Transcribe from URL

```bash
uv run python cli.py "https://x.com/user/status/1234567890"
```

### Transcribe + Summarize

```bash
uv run python cli.py "url" --summarize "5 key takeaways"
uv run python cli.py "url" -s "What are the main arguments?"
uv run python cli.py "url" --summarize --llm-provider openai
```

### Summarize Existing Transcript

```bash
# From transcripts folder
uv run python cli.py transcripts/abc123.txt --summarize "5 key points"

# Any text file
uv run python cli.py /path/to/transcript.txt --summarize "Extract action items"
```

### Options

```bash
--whisper-model tiny|base|small|medium|large  # Default: tiny
--llm-provider groq|openai|anthropic           # Default: groq
--llm-model <model-id>                         # Provider-specific
--api-key <key>                                # Or use env var
-o <file>                                      # Output path
```

## LLM Providers

| Provider | Env Var | Default Model | Cost |
|----------|---------|---------------|------|
| **Groq** | `GROQ_API_KEY` | llama-4-scout | ~$0.01/transcript |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o-mini | ~$0.01/transcript |
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-sonnet-4 | ~$0.02/transcript |
| **Ollama** | (local) | llama3.2 | Free |

```bash
# Groq (fastest/cheapest)
export GROQ_API_KEY="your-key"
uv run python cli.py "url" --summarize

# OpenAI
export OPENAI_API_KEY="your-key"
uv run python cli.py "url" --summarize --llm-provider openai

# Anthropic
export ANTHROPIC_API_KEY="your-key"
uv run python cli.py "url" --summarize --llm-provider anthropic
```

## Project Structure

```
x-transcript/
├── cli.py              # CLI tool (transcribe + summarize)
├── app/                # FastAPI backend (for SaaS)
│   ├── main.py
│   ├── api/
│   ├── services/
│   └── workers/
├── storage/            # Downloaded videos
├── transcripts/        # Output transcripts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── pyproject.toml
```

## Development

```bash
# Run API server
uv run uvicorn app.main:app --reload

# Run Celery worker
uv run celery -A app.workers.transcriber_worker worker -l info

# Run tests
uv run pytest tests/ -v

# Code formatting
uv run black app/ tests/
uv run ruff check app/
```

## Docker

```bash
# Start API + Worker
docker-compose up -d

# View logs
docker-compose logs -f
```

## API (FastAPI)

```bash
# Create transcription job
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"tweet_url": "https://x.com/user/status/123"}'

# Check status
curl "http://localhost:8000/api/v1/jobs/JOB_ID" \
  -H "Authorization: Bearer TOKEN"
```

## Environment Variables

```bash
# LLM Providers
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Database (for API mode)
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

## License

MIT
