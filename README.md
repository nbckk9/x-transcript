# X-Transcript

Transcribe X (Twitter) videos using OpenAI Whisper.

## Features

- 🎥 **Extract** video URLs from X tweets
- ⬇️ **Download** videos for processing
- 🎙️ **Transcribe** using OpenAI Whisper (local, free)
- 📄 **Export** in multiple formats (TXT, SRT, VTT, JSON, MD)
- 🚀 **REST API** for integration
- 📊 **Background processing** with Celery
- 🐳 **Docker-ready** for deployment

## Quick Start with UV

We use [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nbckk9/x-transcript.git
cd x-transcript

# Create virtual environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # or: .\.venv\Scripts\activate on Windows
```

### CLI Usage

```bash
# Transcribe a video directly
uv run python cli_test.py "https://x.com/user/status/1234567890"

# Or using the CLI tool (after installing)
uv run xtranscript-cli "https://x.com/user/status/1234567890"
```

### Development

```bash
# Run the API server
uv run uvicorn app.main:app --reload

# Run Celery worker (separate terminal)
uv run celery -A app.workers.transcriber_worker worker -l info

# Run tests
uv run pytest tests/ -v
```

### Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## API Usage

### Create a Job

```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tweet_url": "https://x.com/user/status/1234567890"}'
```

### Check Status

```bash
curl "http://localhost:8000/api/v1/jobs/JOB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Download Transcript

```bash
curl "http://localhost:8000/api/v1/transcripts/TRANSCRIPT_ID/download" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o transcript.txt
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  API Server     │────▶│  Redis Queue    │────▶│  Celery Worker  │
│  (FastAPI)      │     │                 │     │  (Transcription)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Whisper (local)│
                                                └─────────────────┘
```

## Pricing (SaaS)

| Tier | Price | Minutes/month |
|------|-------|---------------|
| Free | $0 | 30 |
| Pro | $9/mo | 300 |
| Team | $29/mo | 1500 |

## Development

### Code Formatting

```bash
uv run black app/ tests/
uv run isort app/ tests/
uv run ruff check app/
```

### Adding Dependencies

```bash
# Add a dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Use strong `SECRET_KEY`
- [ ] Configure PostgreSQL
- [ ] Set up Redis
- [ ] Configure CORS origins
- [ ] Set up SSL/TLS
- [ ] Configure backups
- [ ] Set up monitoring

### Kubernetes

Helm charts available in `/k8s` directory (coming soon).

## License

MIT License - see [LICENSE](LICENSE) for details.
