"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from app.models import JobStatus, OutputFormat


# ==================== Auth Schemas ====================

class Token(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[UUID] = None
    email: Optional[str] = None


class UserCreate(BaseModel):
    """User registration request."""
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response."""
    id: UUID
    email: str
    credits: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== Job Schemas ====================

class JobCreate(BaseModel):
    """Create transcription job request."""
    tweet_url: HttpUrl = Field(..., description="URL of the X (Twitter) tweet with video")
    output_format: OutputFormat = Field(
        default=OutputFormat.TXT,
        description="Desired output format"
    )
    webhook_url: Optional[HttpUrl] = Field(
        None,
        description="URL to notify when job completes"
    )


class JobResponse(BaseModel):
    """Job response."""
    id: UUID
    tweet_url: str
    video_url: Optional[str]
    status: JobStatus
    progress: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobWithTranscript(JobResponse):
    """Job response with transcript URL."""
    transcript_url: Optional[str] = None


# ==================== Transcript Schemas ====================

class TranscriptResponse(BaseModel):
    """Transcript response."""
    id: UUID
    job_id: UUID
    output_format: OutputFormat
    file_size: Optional[int]
    duration_seconds: Optional[float]
    word_count: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class TranscriptContent(BaseModel):
    """Transcript content response."""
    text: str
    format: OutputFormat
    duration_seconds: Optional[float]
    word_count: int


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    detail: list
