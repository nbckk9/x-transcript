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

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg
- PostgreSQL (or use SQLite for development)
- Redis (for Celery)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/x-transcript.git
cd x-transcript

# Install dependencies
pip install -e .

# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Initialize database
python -c "import asyncio; from app.database import init_db; asyncio.run(init_db())"
```

### Development

```bash
# Run the API server
uvicorn app.main:app --reload

# Run Celery worker (separate terminal)
celery -A app.workers.transcriber_worker worker -l info
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

## CLI Tool

```bash
# Install CLI
pip install -e .

# Transcribe a video
xtranscript "https://x.com/user/status/1234567890"

# With options
xtranscript "url" --format srt --output my_transcript.srt
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

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black app/ tests/
isort app/ tests/
ruff check app/
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
