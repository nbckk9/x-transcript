"""Tests for the video extractor service."""

import pytest
from app.services.extractor import VideoExtractor, extract_video_info


@pytest.fixture
def extractor():
    """Create a VideoExtractor instance."""
    return VideoExtractor()


@pytest.mark.asyncio
async def test_extract_valid_tweet_url():
    """Test that valid tweet URLs are accepted."""
    extractor = VideoExtractor()
    assert extractor._is_valid_tweet_url("https://x.com/user/status/1234567890") is True
    assert extractor._is_valid_tweet_url("https://twitter.com/user/status/1234567890") is True
    assert extractor._is_valid_tweet_url("https://youtube.com/watch?v=123") is False


@pytest.mark.asyncio
async def test_extract_invalid_tweet_url():
    """Test that invalid URLs are rejected."""
    extractor = VideoExtractor()
    assert extractor._is_valid_tweet_url("not-a-url") is False
    assert extractor._is_valid_tweet_url("https://x.com/user") is False


@pytest.mark.asyncio
async def test_extract_video_from_html(extractor):
    """Test video extraction from HTML content."""
    html = '''
    <html>
    <head><meta property="og:video" content="https://video.twimg.com/ext_tw_video/123.mp4"></head>
    <body></body>
    </html>
    '''
    video_url = await extractor._extract_from_html(html)
    assert video_url == "https://video.twimg.com/ext_tw_video/123.mp4"


@pytest.mark.asyncio
async def test_extract_no_video_in_html(extractor):
    """Test handling of HTML without video."""
    html = '<html><body>No video here</body></html>'
    video_url = await extractor._extract_from_html(html)
    assert video_url is None
