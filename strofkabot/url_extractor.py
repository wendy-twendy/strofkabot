"""URL content extraction using Trafilatura."""

import asyncio
import logging
import re
from functools import partial
from urllib.parse import urlparse

import trafilatura

logger = logging.getLogger(__name__)

# URL regex pattern - matches http/https URLs
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)

# Domains to skip (embeds handled by Discord or not useful to extract)
SKIP_DOMAINS = {
    "discord.com",
    "discord.gg",
    "discordapp.com",
    "tenor.com",
    "giphy.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
}

MAX_URL_CONTENT_LENGTH = 2000
MAX_URLS_PER_MESSAGE = 3


def _should_skip_url(url: str) -> bool:
    """Check if URL domain should be skipped."""
    try:
        domain = urlparse(url).netloc.lower()
        return any(skip in domain for skip in SKIP_DOMAINS)
    except Exception:
        return True


def _extract_url_content(url: str) -> str | None:
    """Extract text content from a URL using Trafilatura.

    This is a sync function that should be run in an executor.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        if text:
            # Truncate if too long
            if len(text) > MAX_URL_CONTENT_LENGTH:
                text = text[:MAX_URL_CONTENT_LENGTH] + "..."
            logger.debug("Extracted %d chars from %s", len(text), url)
            return text

        return None

    except Exception:
        logger.warning("Failed to extract content from %s", url, exc_info=True)
        return None


async def extract_urls_from_messages(messages: list[dict]) -> dict[str, str]:
    """Extract and fetch content from URLs in messages.

    Args:
        messages: List of context message dicts with 'content' field.

    Returns:
        Dict mapping URL to extracted text content.
    """
    url_contents: dict[str, str] = {}
    urls_to_fetch: list[str] = []

    # Collect all URLs from messages
    for msg in messages:
        content = msg.get("content", "")
        urls = URL_PATTERN.findall(content)

        for url in urls[:MAX_URLS_PER_MESSAGE]:
            if url in url_contents or url in urls_to_fetch:
                continue  # Already seen

            if _should_skip_url(url):
                continue

            urls_to_fetch.append(url)

    # Fetch URLs in parallel using thread pool
    if urls_to_fetch:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, partial(_extract_url_content, url)) for url in urls_to_fetch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(urls_to_fetch, results, strict=True):
            if isinstance(result, str):
                url_contents[url] = result

    return url_contents


def format_url_context(url_contents: dict[str, str]) -> str:
    """Format extracted URL contents for the prompt.

    Args:
        url_contents: Dict mapping URL to extracted text content.

    Returns:
        Formatted XML string with URL contents, or empty string if none.
    """
    if not url_contents:
        return ""

    parts = ["<url_contents>"]
    for url, content in url_contents.items():
        parts.append(f'<url href="{url}">')
        parts.append(content)
        parts.append("</url>")
    parts.append("</url_contents>")

    return "\n".join(parts)
