"""Video URL extractor service."""

import logging
import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class VideoExtractor:
    """Extract video URLs from X (Twitter) tweets."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def extract(self, tweet_url: str) -> Optional[str]:
        """
        Extract video URL from a tweet.

        Args:
            tweet_url: URL of the tweet containing the video

        Returns:
            Video URL or None if not found
        """
        try:
            # Validate URL
            if not self._is_valid_tweet_url(tweet_url):
                raise ValueError(f"Invalid tweet URL: {tweet_url}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(tweet_url, headers=self.headers)
                response.raise_for_status()

                # Try multiple methods to extract video URL
                video_url = await self._extract_from_html(response.text)
                if video_url:
                    return video_url

                video_url = await self._extract_from_api(response.text)
                if video_url:
                    return video_url

                raise ValueError("No video found in tweet")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error extracting video: {e}")
            raise ValueError(f"Failed to fetch tweet: {e}")

    def _is_valid_tweet_url(self, url: str) -> bool:
        """Check if URL is a valid tweet URL."""
        patterns = [
            r"https://x\.com/\w+/status/\d+",
            r"https://twitter\.com/\w+/status/\d+",
        ]
        return any(re.match(p, url) for p in patterns)

    async def _extract_from_html(self, html: str) -> Optional[str]:
        """Extract video URL from HTML content."""
        soup = BeautifulSoup(html, "html.parser")

        # Method 1: Look for video tags with source
        video_tag = soup.find("video")
        if video_tag:
            source = video_tag.find("source", type="video/mp4")
            if source and source.get("src"):
                return source["src"]

        # Method 2: Look for dynamic video data
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and "videoInfo" in script.string:
                match = re.search(r'"videoURL":"([^"]+)"', script.string)
                if match:
                    return match.group(1).replace("\\/", "/")

        # Method 3: Look for media card
        meta = soup.find("meta", property="og:video")
        if meta and meta.get("content"):
            return meta["content"]

        return None

    async def _extract_from_api(self, html: str) -> Optional[str]:
        """Extract video URL from embedded API data."""
        # Look for Twitter API response in HTML
        match = re.search(
            r'"playbackUrl":"([^"]+)"',
            html.replace("\\u0026", "&")
        )
        if match:
            return match.group(1).replace("\\/", "/")

        # Look for m3u8 manifest
        match = re.search(r'(https://[^"\'&]+\.m3u8[^"\']*)', html)
        if match:
            return match.group(1)

        return None


class TweetInfo:
    """Information extracted from a tweet."""

    def __init__(
        self,
        video_url: str,
        tweet_id: str,
        author_id: str,
        duration: Optional[float] = None,
    ):
        self.video_url = video_url
        self.tweet_id = tweet_id
        self.author_id = author_id
        self.duration = duration


async def extract_video_info(tweet_url: str) -> TweetInfo:
    """Extract video information from a tweet."""
    extractor = VideoExtractor()
    video_url = await extractor.extract(tweet_url)

    # Extract tweet ID and author from URL
    parts = urlparse(tweet_url).path.strip("/").split("/")
    author_id = parts[0] if len(parts) > 1 else "unknown"
    tweet_id = parts[2] if len(parts) > 2 else "unknown"

    return TweetInfo(
        video_url=video_url,
        tweet_id=tweet_id,
        author_id=author_id,
    )
