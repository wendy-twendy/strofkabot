"""Tests for ask helper functions."""

import re
from datetime import UTC, datetime

import pytest

from strofkabot.openrouter_client import QueryMetadata
from strofkabot.utils.ask_helpers import (
    build_system_prompt,
    format_error_response,
    is_image_attachment,
    split_response,
)


def make_metadata(
    search: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "medium",
    query_type: str = "factual",
    key_topics: list[str] | None = None,
    suggested_response_style: str = "conversational",
    language: str = "en",
    requires_citations: bool = False,
) -> QueryMetadata:
    """Create a QueryMetadata instance with defaults."""
    return QueryMetadata(
        search=search,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        query_type=query_type,
        key_topics=key_topics or [],
        suggested_response_style=suggested_response_style,
        language=language,
        requires_citations=requires_citations,
    )


class TestBuildSystemPrompt:
    """Tests for system prompt building."""

    def test_includes_guild_name(self):
        """Test that guild name is included in prompt."""
        prompt = build_system_prompt(
            guild_name="Test Server",
            channel_name="general",
            user_name="TestUser",
        )
        assert "Test Server" in prompt

    def test_includes_channel_name(self):
        """Test that channel name is included with hash."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="test-channel",
            user_name="User",
        )
        assert "#test-channel" in prompt

    def test_includes_user_name(self):
        """Test that user name is included."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="CoolUser123",
        )
        assert "CoolUser123" in prompt

    def test_contains_key_instructions(self):
        """Test that key instructions are present."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="User",
        )
        assert "unfiltered" in prompt.lower()
        assert "discord" in prompt.lower()
        assert "250 chars" in prompt.lower()


class TestSplitResponse:
    """Tests for response splitting."""

    def test_short_response_not_split(self):
        """Test that short responses are not split."""
        text = "Short response"
        chunks = split_response(text, max_length=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_length_not_split(self):
        """Test that text at exact max length is not split."""
        text = "A" * 2000
        chunks = split_response(text, max_length=2000)
        assert len(chunks) == 1

    def test_splits_at_paragraph_break(self):
        """Test splitting at paragraph boundaries."""
        text = "First paragraph." + " " * 50 + "\n\nSecond paragraph."
        chunks = split_response(text, max_length=60)
        assert len(chunks) >= 2
        assert "First paragraph." in chunks[0]

    def test_splits_at_sentence_break(self):
        """Test splitting at sentence boundaries when no paragraph break."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_response(text, max_length=30)
        assert len(chunks) >= 2

    def test_preserves_all_content(self):
        """Test that all content is preserved after splitting."""
        text = "A" * 5000
        chunks = split_response(text, max_length=2000)
        total_length = sum(len(chunk) for chunk in chunks)
        assert total_length == len(text)

    def test_no_chunk_exceeds_max(self):
        """Test that no chunk exceeds max length."""
        text = "Word " * 1000
        chunks = split_response(text, max_length=100)
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_handles_no_natural_breaks(self):
        """Test handling text with no natural break points."""
        text = "A" * 3000
        chunks = split_response(text, max_length=2000)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2000
        assert len(chunks[1]) == 1000


class TestFormatErrorResponse:
    """Tests for error message formatting."""

    @pytest.mark.parametrize(
        "error_type,expected_substring",
        [
            pytest.param("no_question", "!ask", id="no-question-mentions-command"),
            pytest.param("rate_limit", "wait", id="rate-limit-suggests-waiting"),
            pytest.param("config", "configured", id="config-mentions-configuration"),
            pytest.param("too_long", "20,000", id="too-long-shows-limit"),
            pytest.param("exhausted", "tomorrow", id="exhausted-mentions-tomorrow"),
            pytest.param("unknown_type", "wrong", id="unknown-fallback"),
        ],
    )
    def test_error_response_contains_expected_text(self, error_type, expected_substring):
        """Test that error responses contain expected text."""
        msg = format_error_response(error_type)
        assert expected_substring.lower() in msg.lower()

    def test_exhausted_error_shows_limit(self):
        """Test exhausted error shows the daily limit number."""
        msg = format_error_response("exhausted")
        assert "60" in msg

    def test_no_question_mentions_question(self):
        """Test no question error mentions 'question'."""
        msg = format_error_response("no_question")
        assert "question" in msg.lower()

    def test_api_error_with_details(self):
        """Test API error includes debug details."""
        msg = format_error_response("api", "Connection timeout")
        assert "Connection timeout" in msg
        assert "||" in msg  # Discord spoiler tags

    def test_api_error_truncates_long_details(self):
        """Test that long error details are truncated."""
        long_detail = "A" * 200
        msg = format_error_response("api", long_detail)
        assert len(msg) < 250


class TestIsImageAttachment:
    """Tests for image detection."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("image.png", True),
            ("photo.jpg", True),
            ("pic.JPEG", True),
            ("pic.JPG", True),
            ("animation.gif", True),
            ("picture.webp", True),
            ("image.bmp", True),
            ("image.tiff", True),
            ("document.pdf", False),
            ("video.mp4", False),
            ("file.txt", False),
            ("archive.zip", False),
            ("music.mp3", False),
            ("noextension", False),
        ],
    )
    def test_image_detection(self, filename, expected):
        """Test various file extensions for image detection."""
        assert is_image_attachment(filename) == expected

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert is_image_attachment("IMAGE.PNG") is True
        assert is_image_attachment("Photo.JpG") is True
        assert is_image_attachment("pic.GIF") is True


class TestBuildSystemPromptDateTime:
    """Tests for date/time inclusion in system prompt."""

    def test_includes_current_date(self):
        """Test that current date is included in prompt."""
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
        )
        # Should contain the current year
        today = datetime.now(UTC)
        assert str(today.year) in prompt

    def test_includes_utc_time_indicator(self):
        """Test that UTC time indicator is included in prompt."""
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
        )
        # Should contain UTC indicator or time format
        assert "UTC" in prompt or re.search(r"\d{1,2}:\d{2}", prompt)


class TestBuildSystemPromptWithMetadata:
    """Tests for system prompt with QueryMetadata."""

    @pytest.mark.parametrize(
        "query_type,expected_keywords",
        [
            pytest.param("technical", ["technical", "code"], id="technical-query"),
            pytest.param("creative", ["creative", "imaginative"], id="creative-query"),
            pytest.param("factual", ["factual", "accurate"], id="factual-query"),
        ],
    )
    def test_query_type_adds_instructions(self, query_type, expected_keywords):
        """Test that query types add specific instructions."""
        metadata = make_metadata(query_type=query_type)
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        prompt_lower = prompt.lower()
        assert any(kw in prompt_lower for kw in expected_keywords)

    @pytest.mark.parametrize(
        "style,expected_keywords",
        [
            pytest.param("brief", ["brief", "concise", "direct"], id="brief-style"),
            pytest.param("detailed", ["detailed", "thorough"], id="detailed-style"),
            pytest.param("sarcastic", ["sarcastic", "witty", "humor"], id="sarcastic-style"),
        ],
    )
    def test_response_style_adds_instructions(self, style, expected_keywords):
        """Test that response styles add specific instructions."""
        metadata = make_metadata(suggested_response_style=style)
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        prompt_lower = prompt.lower()
        assert any(kw in prompt_lower for kw in expected_keywords)

    def test_key_topics_included(self):
        """Test that key topics are included in prompt."""
        metadata = make_metadata(key_topics=["Python", "async"])
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert "Python" in prompt
        assert "async" in prompt


class TestBuildSystemPromptLanguage:
    """Tests for language handling in system prompt."""

    def test_english_language_no_extra_instruction(self):
        """Test that English doesn't add extra language instruction."""
        metadata = make_metadata(language="en")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        # Should not have explicit "respond in en" instruction
        assert "respond in en" not in prompt.lower()

    def test_non_english_language_adds_instruction(self):
        """Test that non-English language adds instruction."""
        metadata = make_metadata(language="sr")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        # Should have instruction to respond in that language
        assert "sr" in prompt.lower() or "language" in prompt.lower()


class TestBuildSystemPromptWithoutMetadata:
    """Tests for system prompt when metadata is None."""

    def test_works_without_metadata(self):
        """Test that prompt builds successfully without metadata."""
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=None,
        )
        assert len(prompt) > 0
        assert "Test" in prompt

    def test_still_includes_datetime_without_metadata(self):
        """Test that date/time is included even without metadata."""
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=None,
        )
        today = datetime.now(UTC)
        assert str(today.year) in prompt


class TestFetchContextMessages:
    """Tests for fetch_context_messages function."""

    @pytest.mark.asyncio
    async def test_fetches_messages_excluding_command(self):
        """Test that command message is excluded from results."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import fetch_context_messages

        # Create mock messages
        msg1 = MagicMock()
        msg1.id = 100
        msg2 = MagicMock()
        msg2.id = 200  # Command message to exclude
        msg3 = MagicMock()
        msg3.id = 300

        # Create mock channel with async iterator
        mock_channel = MagicMock()

        async def mock_history(limit):
            for msg in [msg3, msg2, msg1]:  # Discord returns newest first
                yield msg

        mock_channel.history = mock_history

        result = await fetch_context_messages(mock_channel, exclude_message_id=200, limit=10)

        # Should exclude message 200 and return in oldest-first order
        assert len(result) == 2
        assert result[0].id == 100
        assert result[1].id == 300

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Test that limit is respected."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import fetch_context_messages

        messages = [MagicMock(id=i) for i in range(20)]

        mock_channel = MagicMock()

        async def mock_history(limit):
            for msg in messages[:limit]:
                yield msg

        mock_channel.history = mock_history

        result = await fetch_context_messages(mock_channel, exclude_message_id=999, limit=5)

        assert len(result) == 5


class TestPrepareContext:
    """Tests for prepare_context function."""

    @pytest.mark.asyncio
    async def test_prepares_basic_context(self):
        """Test that basic context is prepared correctly."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import prepare_context

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.id = 12345
        msg.author.display_name = "TestUser"
        msg.content = "Hello world"
        msg.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        msg.reference = None
        msg.attachments = []

        result = await prepare_context([msg])

        assert len(result) == 1
        assert result[0]["author"] == "TestUser"
        assert result[0]["content"] == "Hello world"
        assert result[0]["timestamp"] == "10:30"
        assert result[0]["image_count"] == 0

    @pytest.mark.asyncio
    async def test_handles_empty_content(self):
        """Test that empty content is replaced with placeholder."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import prepare_context

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.id = 12345
        msg.author.display_name = "TestUser"
        msg.content = ""
        msg.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        msg.reference = None
        msg.attachments = []

        result = await prepare_context([msg])

        assert result[0]["content"] == "[no text]"

    @pytest.mark.asyncio
    async def test_counts_image_attachments(self):
        """Test that image attachments are counted."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import prepare_context

        attachment1 = MagicMock()
        attachment1.filename = "photo.jpg"
        attachment2 = MagicMock()
        attachment2.filename = "image.png"
        attachment3 = MagicMock()
        attachment3.filename = "document.pdf"

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.id = 12345
        msg.author.display_name = "TestUser"
        msg.content = "Check these images"
        msg.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        msg.reference = None
        msg.attachments = [attachment1, attachment2, attachment3]

        result = await prepare_context([msg])

        # Should count only image attachments (jpg, png), not pdf
        assert result[0]["image_count"] == 2

    @pytest.mark.asyncio
    async def test_handles_reply_info(self):
        """Test that reply information is included."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import prepare_context

        replied_msg = MagicMock()
        replied_msg.id = 100
        replied_msg.author = MagicMock()
        replied_msg.author.id = 54321
        replied_msg.author.display_name = "OriginalUser"
        replied_msg.content = "Original message content here"

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.id = 12345
        msg.author.display_name = "Replier"
        msg.content = "My reply"
        msg.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        msg.reference = MagicMock()
        msg.reference.resolved = replied_msg
        msg.attachments = []

        result = await prepare_context([msg])

        assert result[0]["reply_to_author"] == "OriginalUser"
        assert result[0]["reply_to_content"][:20] == "Original message con"

    @pytest.mark.asyncio
    async def test_uses_nicknames_when_provided(self):
        """Test that nicknames override display names when provided."""
        from unittest.mock import MagicMock

        from strofkabot.utils.ask_helpers import prepare_context

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.id = 12345
        msg.author.display_name = "RealName"
        msg.content = "Hello"
        msg.created_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)
        msg.reference = None
        msg.attachments = []

        nicknames = {12345: ["Nickname", "AltNickname"]}

        result = await prepare_context([msg], nicknames=nicknames)

        assert result[0]["author"] == "Nickname"


class TestExtractImagesFromMessages:
    """Tests for extract_images_from_messages function."""

    @pytest.mark.asyncio
    async def test_extracts_images(self):
        """Test that images are extracted from messages."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import extract_images_from_messages

        attachment = MagicMock()
        attachment.filename = "photo.jpg"
        attachment.url = "http://example.com/photo.jpg"

        msg = MagicMock()
        msg.attachments = [attachment]

        with patch(
            "strofkabot.utils.ask_helpers.download_and_encode_image",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = {"data": "base64data", "mime_type": "image/jpeg"}

            result = await extract_images_from_messages([msg])

            assert len(result) == 1
            assert result[0]["data"] == "base64data"

    @pytest.mark.asyncio
    async def test_skips_non_image_attachments(self):
        """Test that non-image attachments are skipped."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import extract_images_from_messages

        attachment = MagicMock()
        attachment.filename = "document.pdf"
        attachment.url = "http://example.com/doc.pdf"

        msg = MagicMock()
        msg.attachments = [attachment]

        with patch(
            "strofkabot.utils.ask_helpers.download_and_encode_image",
            new_callable=AsyncMock,
        ) as mock_download:
            result = await extract_images_from_messages([msg])

            assert len(result) == 0
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_limits_to_3_images_per_message(self):
        """Test that only 3 images per message are processed."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import extract_images_from_messages

        attachments = [
            MagicMock(filename=f"image{i}.jpg", url=f"http://example.com/{i}.jpg") for i in range(5)
        ]

        msg = MagicMock()
        msg.attachments = attachments

        with patch(
            "strofkabot.utils.ask_helpers.download_and_encode_image",
            new_callable=AsyncMock,
        ) as mock_download:
            mock_download.return_value = {"data": "base64", "mime_type": "image/jpeg"}

            result = await extract_images_from_messages([msg])

            assert len(result) == 3
            assert mock_download.call_count == 3

    @pytest.mark.asyncio
    async def test_skips_failed_downloads(self):
        """Test that failed downloads are skipped."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import extract_images_from_messages

        attachments = [
            MagicMock(filename="good.jpg", url="http://example.com/good.jpg"),
            MagicMock(filename="bad.jpg", url="http://example.com/bad.jpg"),
        ]

        msg = MagicMock()
        msg.attachments = attachments

        with patch(
            "strofkabot.utils.ask_helpers.download_and_encode_image",
            new_callable=AsyncMock,
        ) as mock_download:
            # First succeeds, second fails
            mock_download.side_effect = [
                {"data": "base64", "mime_type": "image/jpeg"},
                None,
            ]

            result = await extract_images_from_messages([msg])

            assert len(result) == 1


class TestDownloadAndEncodeImage:
    """Tests for download_and_encode_image function."""

    @pytest.mark.asyncio
    async def test_successful_download_and_encode(self):
        """Test successful image download and encoding."""
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        from PIL import Image

        from strofkabot.utils.ask_helpers import download_and_encode_image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=image_bytes)

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession", return_value=mock_session):
            result = await download_and_encode_image("http://example.com/image.jpg", "image.jpg")

        assert result is not None
        assert "data" in result
        assert result["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        """Test that HTTP errors return None."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import download_and_encode_image

        mock_response = AsyncMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession", return_value=mock_session):
            result = await download_and_encode_image("http://example.com/image.jpg", "image.jpg")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_oversized_image(self):
        """Test that oversized images return None."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from strofkabot.utils.ask_helpers import download_and_encode_image

        # Create image data larger than 8MB (2x the 4MB limit)
        large_image_bytes = b"x" * (9 * 1024 * 1024)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=large_image_bytes)

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession", return_value=mock_session):
            result = await download_and_encode_image("http://example.com/large.jpg", "large.jpg")

        assert result is None

    @pytest.mark.asyncio
    async def test_resizes_large_dimension_images(self):
        """Test that images with large dimensions are resized."""
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        from PIL import Image

        from strofkabot.utils.ask_helpers import download_and_encode_image

        # Create a large image (3000x3000)
        img = Image.new("RGB", (3000, 3000), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=50)
        image_bytes = buffer.getvalue()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=image_bytes)

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession", return_value=mock_session):
            result = await download_and_encode_image("http://example.com/large.jpg", "large.jpg")

        assert result is not None
        assert result["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_converts_rgba_to_rgb(self):
        """Test that RGBA images are converted to RGB."""
        import io
        from unittest.mock import AsyncMock, MagicMock, patch

        from PIL import Image

        from strofkabot.utils.ask_helpers import download_and_encode_image

        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=image_bytes)

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession", return_value=mock_session):
            result = await download_and_encode_image("http://example.com/image.png", "image.png")

        assert result is not None
        assert result["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """Test that exceptions return None."""
        from unittest.mock import patch

        from strofkabot.utils.ask_helpers import download_and_encode_image

        with patch("strofkabot.utils.ask_helpers.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = Exception("Connection error")

            result = await download_and_encode_image("http://example.com/image.jpg", "image.jpg")

        assert result is None
