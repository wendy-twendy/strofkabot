"""Tests for ask helper functions."""

import pytest

from strofkabot.utils.ask_helpers import (
    build_system_prompt,
    format_error_response,
    is_image_attachment,
    split_response,
)


class TestBuildSystemPrompt:
    """Tests for system prompt building."""

    def test_includes_guild_name(self):
        """Test that guild name is included in prompt."""
        prompt = build_system_prompt(
            guild_name="Test Server",
            channel_name="general",
            user_name="TestUser",
            user_roles=["Member"],
        )
        assert "Test Server" in prompt

    def test_includes_channel_name(self):
        """Test that channel name is included with hash."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="test-channel",
            user_name="User",
            user_roles=[],
        )
        assert "#test-channel" in prompt

    def test_includes_user_name(self):
        """Test that user name is included."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="CoolUser123",
            user_roles=[],
        )
        assert "CoolUser123" in prompt

    def test_handles_empty_roles(self):
        """Test fallback to 'Member' when no roles provided."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="User",
            user_roles=[],
        )
        assert "Member" in prompt

    def test_includes_roles(self):
        """Test that roles are included."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="User",
            user_roles=["Admin", "Moderator"],
        )
        assert "Admin" in prompt
        assert "Moderator" in prompt

    def test_limits_roles_to_five(self):
        """Test that only first 5 roles are included."""
        roles = ["Role1", "Role2", "Role3", "Role4", "Role5", "Role6", "Role7"]
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="User",
            user_roles=roles,
        )
        assert "Role5" in prompt
        assert "Role6" not in prompt

    def test_contains_key_instructions(self):
        """Test that key instructions are present."""
        prompt = build_system_prompt(
            guild_name="Server",
            channel_name="general",
            user_name="User",
            user_roles=[],
        )
        assert "helpful" in prompt.lower()
        assert "discord" in prompt.lower()
        assert "google search" in prompt.lower()


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
