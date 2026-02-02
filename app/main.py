"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, jobs
from app.config import settings
from app.database import close_db, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting X-Transcript service...")
    await init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    logger.info("Database connection closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for transcribing X (Twitter) videos using OpenAI Whisper",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(jobs.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
    }


# Development: Create a simple test endpoint
if settings.DEBUG:

    @app.post("/auth/register")
    async def dev_register(email: str, password: str):
        """Dev-only registration for testing."""
        import uuid
        import hashlib
        from sqlalchemy import insert
        from app.models import User

        async with app.state.db_session() as db:
            await db.execute(
                insert(User).values(
                    id=uuid.uuid4(),
                    email=email,
                    password_hash=hashlib.sha256(password.encode()).hexdigest(),
                    api_key=str(uuid.uuid4().hex),
                    credits=100,
                )
            )
            await db.commit()
        return {"status": "created"}
