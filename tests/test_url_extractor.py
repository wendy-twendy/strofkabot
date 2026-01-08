"""Tests for URL content extraction module."""

from unittest.mock import patch

import pytest

from strofkabot.url_extractor import (
    MAX_URL_CONTENT_LENGTH,
    SKIP_DOMAINS,
    URL_PATTERN,
    format_url_context,
)


class TestUrlPattern:
    """Tests for URL regex pattern."""

    def test_matches_http_urls(self):
        """Test that HTTP URLs are matched."""
        text = "Check out http://example.com/page"
        matches = URL_PATTERN.findall(text)
        assert matches == ["http://example.com/page"]

    def test_matches_https_urls(self):
        """Test that HTTPS URLs are matched."""
        text = "Check out https://example.com/page"
        matches = URL_PATTERN.findall(text)
        assert matches == ["https://example.com/page"]

    def test_matches_urls_with_paths(self):
        """Test that URLs with complex paths are matched."""
        text = "See https://example.com/path/to/page?query=1&foo=bar"
        matches = URL_PATTERN.findall(text)
        assert matches == ["https://example.com/path/to/page?query=1&foo=bar"]

    def test_matches_multiple_urls(self):
        """Test that multiple URLs in text are matched."""
        text = "Visit https://foo.com and https://bar.com for more"
        matches = URL_PATTERN.findall(text)
        assert matches == ["https://foo.com", "https://bar.com"]

    def test_stops_at_whitespace(self):
        """Test that URLs stop at whitespace."""
        text = "https://example.com/page is great"
        matches = URL_PATTERN.findall(text)
        assert matches == ["https://example.com/page"]

    def test_case_insensitive(self):
        """Test that URL matching is case insensitive."""
        text = "HTTPS://EXAMPLE.COM/PAGE"
        matches = URL_PATTERN.findall(text)
        assert matches == ["HTTPS://EXAMPLE.COM/PAGE"]


class TestSkipDomains:
    """Tests for the skip domains set."""

    def test_discord_domains_skipped(self):
        """Test that Discord domains are in skip list."""
        assert "discord.com" in SKIP_DOMAINS
        assert "discord.gg" in SKIP_DOMAINS
        assert "discordapp.com" in SKIP_DOMAINS

    def test_gif_services_skipped(self):
        """Test that GIF services are in skip list."""
        assert "tenor.com" in SKIP_DOMAINS
        assert "giphy.com" in SKIP_DOMAINS

    def test_social_media_skipped(self):
        """Test that social media domains are in skip list."""
        assert "twitter.com" in SKIP_DOMAINS
        assert "x.com" in SKIP_DOMAINS

    def test_video_services_skipped(self):
        """Test that video services are in skip list."""
        assert "youtube.com" in SKIP_DOMAINS
        assert "youtu.be" in SKIP_DOMAINS


class TestFormatUrlContext:
    """Tests for URL context formatting."""

    def test_empty_dict_returns_empty_string(self):
        """Test that empty dict returns empty string."""
        result = format_url_context({})
        assert result == ""

    def test_single_url_formats_correctly(self):
        """Test formatting of a single URL with content."""
        url_contents = {"https://example.com/article": "This is the article content."}
        result = format_url_context(url_contents)

        assert "<url_contents>" in result
        assert "</url_contents>" in result
        assert '<url href="https://example.com/article">' in result
        assert "This is the article content." in result
        assert "</url>" in result

    def test_multiple_urls_format_correctly(self):
        """Test formatting of multiple URLs."""
        url_contents = {
            "https://foo.com": "Foo content",
            "https://bar.com": "Bar content",
        }
        result = format_url_context(url_contents)

        assert result.count("<url href=") == 2
        assert result.count("</url>") == 2
        assert "Foo content" in result
        assert "Bar content" in result


class TestMaxUrlContentLength:
    """Tests for content length limit."""

    def test_max_length_is_reasonable(self):
        """Test that max length is set to a reasonable value."""
        # Should be enough for meaningful content but not too long
        assert MAX_URL_CONTENT_LENGTH >= 1000
        assert MAX_URL_CONTENT_LENGTH <= 10000


class TestShouldSkipUrl:
    """Tests for URL skipping logic."""

    def test_skips_discord_urls(self):
        """Test that Discord URLs are skipped."""
        from strofkabot.url_extractor import _should_skip_url

        assert _should_skip_url("https://discord.com/channels/123") is True
        assert _should_skip_url("https://discord.gg/invite123") is True
        assert _should_skip_url("https://cdn.discordapp.com/attachments/1/2/3") is True

    def test_skips_gif_services(self):
        """Test that GIF service URLs are skipped."""
        from strofkabot.url_extractor import _should_skip_url

        assert _should_skip_url("https://tenor.com/view/funny-gif") is True
        assert _should_skip_url("https://giphy.com/gifs/abc123") is True

    def test_skips_twitter_urls(self):
        """Test that Twitter/X URLs are skipped."""
        from strofkabot.url_extractor import _should_skip_url

        assert _should_skip_url("https://twitter.com/user/status/123") is True
        assert _should_skip_url("https://x.com/user/status/456") is True

    def test_skips_youtube_urls(self):
        """Test that YouTube URLs are skipped."""
        from strofkabot.url_extractor import _should_skip_url

        assert _should_skip_url("https://youtube.com/watch?v=abc123") is True
        assert _should_skip_url("https://youtu.be/abc123") is True
        assert _should_skip_url("https://www.youtube.com/shorts/xyz") is True

    def test_allows_regular_urls(self):
        """Test that regular URLs are not skipped."""
        from strofkabot.url_extractor import _should_skip_url

        assert _should_skip_url("https://example.com/article") is False
        assert _should_skip_url("https://news.ycombinator.com/item?id=123") is False
        assert _should_skip_url("https://github.com/user/repo") is False

    def test_handles_urls_without_skip_domains(self):
        """Test that URLs without skip domains are allowed.

        Note: Invalid URLs (without http/https) wouldn't be matched by URL_PATTERN
        anyway, so we only need to ensure the function doesn't crash.
        """
        from strofkabot.url_extractor import _should_skip_url

        # These don't have skip domains, so they're allowed
        assert _should_skip_url("not-a-url") is False  # No domain to skip
        assert _should_skip_url("") is False  # Empty string, no domain to skip
        # But proper URLs without skip domains are also allowed
        assert _should_skip_url("https://example.com") is False


class TestExtractUrlContent:
    """Tests for URL content extraction with mocked trafilatura."""

    def test_extracts_content_successfully(self):
        """Test successful content extraction."""
        from strofkabot.url_extractor import _extract_url_content

        with patch("strofkabot.url_extractor.trafilatura") as mock_trafilatura:
            mock_trafilatura.fetch_url.return_value = "<html>content</html>"
            mock_trafilatura.extract.return_value = "Extracted article text"

            result = _extract_url_content("https://example.com/article")

            assert result == "Extracted article text"
            mock_trafilatura.fetch_url.assert_called_once_with("https://example.com/article")

    def test_returns_none_when_fetch_fails(self):
        """Test that None is returned when fetch fails."""
        from strofkabot.url_extractor import _extract_url_content

        with patch("strofkabot.url_extractor.trafilatura") as mock_trafilatura:
            mock_trafilatura.fetch_url.return_value = None

            result = _extract_url_content("https://example.com/404")

            assert result is None

    def test_returns_none_when_extraction_fails(self):
        """Test that None is returned when extraction returns nothing."""
        from strofkabot.url_extractor import _extract_url_content

        with patch("strofkabot.url_extractor.trafilatura") as mock_trafilatura:
            mock_trafilatura.fetch_url.return_value = "<html>empty</html>"
            mock_trafilatura.extract.return_value = None

            result = _extract_url_content("https://example.com/empty")

            assert result is None

    def test_truncates_long_content(self):
        """Test that content is truncated if too long."""
        from strofkabot.url_extractor import _extract_url_content

        with patch("strofkabot.url_extractor.trafilatura") as mock_trafilatura:
            mock_trafilatura.fetch_url.return_value = "<html>long</html>"
            # Create content longer than MAX_URL_CONTENT_LENGTH
            long_content = "A" * 3000
            mock_trafilatura.extract.return_value = long_content

            result = _extract_url_content("https://example.com/long")

            assert result is not None
            assert len(result) <= MAX_URL_CONTENT_LENGTH + 3  # +3 for "..."
            assert result.endswith("...")

    def test_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully."""
        from strofkabot.url_extractor import _extract_url_content

        with patch("strofkabot.url_extractor.trafilatura") as mock_trafilatura:
            mock_trafilatura.fetch_url.side_effect = Exception("Network error")

            result = _extract_url_content("https://example.com/error")

            assert result is None


class TestExtractUrlsFromMessages:
    """Tests for async URL extraction from messages."""

    @pytest.mark.asyncio
    async def test_extracts_urls_from_single_message(self):
        """Test extracting URLs from a single message."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [{"author": "User1", "content": "Check this out: https://example.com/article"}]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            mock_extract.return_value = "Article content here"

            result = await extract_urls_from_messages(messages)

            assert "https://example.com/article" in result
            assert result["https://example.com/article"] == "Article content here"

    @pytest.mark.asyncio
    async def test_extracts_urls_from_multiple_messages(self):
        """Test extracting URLs from multiple messages."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [
            {"author": "User1", "content": "Link 1: https://example.com/a"},
            {"author": "User2", "content": "Link 2: https://example.com/b"},
        ]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            mock_extract.side_effect = ["Content A", "Content B"]

            result = await extract_urls_from_messages(messages)

            assert len(result) == 2
            assert "https://example.com/a" in result
            assert "https://example.com/b" in result

    @pytest.mark.asyncio
    async def test_skips_duplicate_urls(self):
        """Test that duplicate URLs are only fetched once."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [
            {"author": "User1", "content": "Link: https://example.com/same"},
            {"author": "User2", "content": "Same link: https://example.com/same"},
        ]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            mock_extract.return_value = "Content"

            result = await extract_urls_from_messages(messages)

            # Should only have one entry
            assert len(result) == 1
            # Should only call extract once
            assert mock_extract.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_blocked_domains(self):
        """Test that blocked domains are skipped."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [
            {"author": "User1", "content": "Discord: https://discord.gg/invite"},
            {"author": "User2", "content": "Valid: https://example.com/page"},
        ]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            mock_extract.return_value = "Valid content"

            result = await extract_urls_from_messages(messages)

            # Only the example.com URL should be fetched
            assert len(result) == 1
            assert "https://example.com/page" in result
            assert "https://discord.gg/invite" not in result

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self):
        """Test handling of empty message list."""
        from strofkabot.url_extractor import extract_urls_from_messages

        result = await extract_urls_from_messages([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_handles_messages_without_urls(self):
        """Test handling messages without any URLs."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [
            {"author": "User1", "content": "Just a regular message"},
            {"author": "User2", "content": "No links here either"},
        ]

        result = await extract_urls_from_messages(messages)

        assert result == {}

    @pytest.mark.asyncio
    async def test_limits_urls_per_message(self):
        """Test that URLs per message are limited."""
        from strofkabot.url_extractor import MAX_URLS_PER_MESSAGE, extract_urls_from_messages

        # Create message with more URLs than the limit
        urls = [f"https://example.com/{i}" for i in range(10)]
        messages = [{"author": "User1", "content": " ".join(urls)}]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            mock_extract.return_value = "Content"

            result = await extract_urls_from_messages(messages)

            # Should be limited to MAX_URLS_PER_MESSAGE
            assert len(result) <= MAX_URLS_PER_MESSAGE

    @pytest.mark.asyncio
    async def test_handles_extraction_failures(self):
        """Test that extraction failures don't break the whole process."""
        from strofkabot.url_extractor import extract_urls_from_messages

        messages = [
            {"author": "User1", "content": "https://example.com/fail https://example.com/ok"}
        ]

        with patch("strofkabot.url_extractor._extract_url_content") as mock_extract:
            # First URL fails, second succeeds
            mock_extract.side_effect = [None, "Success content"]

            result = await extract_urls_from_messages(messages)

            # Only successful extraction should be in result
            assert len(result) == 1
            assert "https://example.com/ok" in result
