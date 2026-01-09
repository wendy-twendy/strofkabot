"""Test file for message_filter module.

Uses parametrization for data-driven testing of message validation rules.
"""

# pylint: disable=redefined-outer-name
import pytest

from strofkabot.message_filter import MessageFilter


@pytest.fixture
def message_filter():
    """Fixture to create an instance of MessageFilter."""
    return MessageFilter()


# ============================================================================
# Parametrized Tests for is_valid_message
# ============================================================================


class TestValidMessages:
    """Tests for messages that should be considered valid."""

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param("This is a valid message", id="basic-valid"),
            pytest.param("This message has a colon: but it's not an emoji", id="colon-not-emoji"),
            pytest.param("This message has an @ symbol but not a valid tag", id="at-not-tag"),
            pytest.param("1234567890111111", id="exactly-16-chars"),
            pytest.param("This has unicode: кириллица текст", id="cyrillic-unicode"),
            pytest.param("Chinese characters: 这是一个测试消息", id="chinese-unicode"),
        ],
    )
    def test_valid_messages(self, message_filter, message):
        """Test that valid messages are correctly identified."""
        assert message_filter.is_valid_message(message)


class TestInvalidMessages:
    """Tests for messages that should be considered invalid."""

    @pytest.mark.parametrize(
        "message,reason",
        [
            # Too short
            pytest.param("Short", "too-short", id="short-message"),
            pytest.param("12345678911111", "too-short", id="14-chars"),
            pytest.param("", "empty", id="empty-string"),
            pytest.param("   ", "whitespace-only", id="spaces-only"),
            pytest.param("\t\n\r", "whitespace-only", id="tabs-newlines"),
            # Contains links
            pytest.param("Check out http://example.com", "http-link", id="http-link"),
            pytest.param(
                " https://www.reddit.com/r/science/comments/8cih30/a_new_study?utm_source=reddit",
                "https-link",
                id="https-link-with-params",
            ),
            # Contains Discord custom emojis
            pytest.param("<:GWqlabsBan:398950688555663360>", "single-emoji", id="single-emoji"),
            pytest.param(
                "<:GWqlabsBan:398950688555663360> <:GWqlabsBan:398950688555663360>",
                "multiple-emojis",
                id="multiple-emojis",
            ),
            # Contains user mentions
            pytest.param(
                "test <@!416623828920172544> tesdfsdfsdfst",
                "user-mention",
                id="user-mention-with-text",
            ),
            # Multiple exclusions
            pytest.param(
                "@user1234 check http://example.com :smiley:",
                "multiple-exclusions",
                id="multiple-exclusions",
            ),
        ],
    )
    def test_invalid_messages(self, message_filter, message, reason):
        """Test that invalid messages are correctly rejected."""
        assert not message_filter.is_valid_message(message), f"Should be invalid due to: {reason}"


# ============================================================================
# Parametrized Tests for is_link
# ============================================================================


class TestIsLink:
    """Tests for the is_link method."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            pytest.param("http://example.com", True, id="http-simple"),
            pytest.param("https://example.com", True, id="https-simple"),
            pytest.param("check this http://test.com out", True, id="http-in-text"),
            pytest.param("no link here", False, id="no-link"),
            pytest.param("htt://not-a-link", False, id="malformed-protocol"),
            pytest.param("ftp://files.example.com", False, id="ftp-not-matched"),
        ],
    )
    def test_is_link(self, message_filter, text, expected):
        """Test link detection for various inputs."""
        assert message_filter.is_link(text) == expected


# ============================================================================
# Parametrized Tests for is_emoji
# ============================================================================


class TestIsEmoji:
    """Tests for the is_emoji method (Discord custom emojis)."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            pytest.param("<:smile:123456789>", True, id="simple-emoji"),
            pytest.param("<:GWqlabsBan:398950688555663360>", True, id="complex-emoji-name"),
            pytest.param("some text <:emoji:123> more text", True, id="emoji-in-text"),
            pytest.param(":smile:", False, id="standard-emoji-syntax"),
            pytest.param("no emoji here", False, id="no-emoji"),
            pytest.param("<:>", False, id="malformed-empty"),
        ],
    )
    def test_is_emoji(self, message_filter, text, expected):
        """Test Discord custom emoji detection."""
        assert message_filter.is_emoji(text) == expected


# ============================================================================
# Parametrized Tests for is_tag
# ============================================================================


class TestIsTag:
    """Tests for the is_tag method (Discord mentions)."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            pytest.param("<@123456789>", True, id="user-mention"),
            pytest.param("<@!416623828920172544>", True, id="nickname-mention"),
            pytest.param("hey <@user> check this", True, id="mention-in-text"),
            pytest.param("@username", False, id="at-symbol-only"),
            pytest.param("no mention here", False, id="no-mention"),
        ],
    )
    def test_is_tag(self, message_filter, text, expected):
        """Test Discord mention detection."""
        assert message_filter.is_tag(text) == expected


# ============================================================================
# Regression Tests
# ============================================================================


class TestRegressions:
    """Regression tests for previously fixed bugs."""

    def test_channel_method_removed(self, message_filter):
        """Regression test: verify is_channel method was removed (had undefined channel_pattern)."""
        assert not hasattr(message_filter, "is_channel")
