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
    is_followup: bool = False,
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
        is_followup=is_followup,
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
        assert "helpful" in prompt.lower()
        assert "discord" in prompt.lower()
        assert "concise" in prompt.lower()


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

    def test_no_question_error(self):
        """Test no question error message."""
        msg = format_error_response("no_question")
        assert "!ask" in msg
        assert "question" in msg.lower()

    def test_rate_limit_error(self):
        """Test rate limit error message."""
        msg = format_error_response("rate_limit")
        assert "wait" in msg.lower()

    def test_config_error(self):
        """Test config error message."""
        msg = format_error_response("config")
        assert "configured" in msg.lower()

    def test_too_long_error(self):
        """Test too long question error."""
        msg = format_error_response("too_long")
        assert "20,000" in msg

    def test_exhausted_error(self):
        """Test exhausted daily limit error."""
        msg = format_error_response("exhausted")
        assert "60" in msg
        assert "tomorrow" in msg.lower()

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

    def test_unknown_error_type(self):
        """Test fallback for unknown error types."""
        msg = format_error_response("unknown_type")
        assert "wrong" in msg.lower()


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

    def test_technical_query_type_adds_instructions(self):
        """Test that technical query type adds specific instructions."""
        metadata = make_metadata(query_type="technical")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        # Should contain technical-specific guidance
        assert "technical" in prompt.lower() or "code" in prompt.lower()

    def test_creative_query_type_adds_instructions(self):
        """Test that creative query type adds specific instructions."""
        metadata = make_metadata(query_type="creative")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert "creative" in prompt.lower() or "imaginative" in prompt.lower()

    def test_factual_query_type_adds_instructions(self):
        """Test that factual query type adds specific instructions."""
        metadata = make_metadata(query_type="factual")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert "factual" in prompt.lower() or "accurate" in prompt.lower()

    def test_brief_style_adds_instructions(self):
        """Test that brief response style adds instructions."""
        metadata = make_metadata(suggested_response_style="brief")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert (
            "brief" in prompt.lower() or "concise" in prompt.lower() or "direct" in prompt.lower()
        )

    def test_detailed_style_adds_instructions(self):
        """Test that detailed response style adds instructions."""
        metadata = make_metadata(suggested_response_style="detailed")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert "detailed" in prompt.lower() or "thorough" in prompt.lower()

    def test_sarcastic_style_adds_instructions(self):
        """Test that sarcastic response style adds instructions."""
        metadata = make_metadata(suggested_response_style="sarcastic")
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert (
            "sarcastic" in prompt.lower() or "witty" in prompt.lower() or "humor" in prompt.lower()
        )

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

    def test_is_followup_adds_hint(self):
        """Test that is_followup adds context hint to prompt."""
        metadata = make_metadata(is_followup=True)
        prompt = build_system_prompt(
            guild_name="Test",
            channel_name="general",
            user_name="User",
            query_metadata=metadata,
        )
        assert "follow-up" in prompt.lower() or "previous" in prompt.lower()


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
