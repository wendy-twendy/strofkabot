"""
Tests for simple Discord commands (!unsubscribe, !artan).
"""
import discord.ext.test as dpytest
import pytest


class TestUnsubscribeCommand:
    """Tests for the !unsubscribe command."""

    @pytest.mark.asyncio
    async def test_unsubscribe_returns_expected_message(self, bot_with_mocked_db):
        """Test !unsubscribe returns the expected message."""
        bot, _, _, _ = bot_with_mocked_db

        await dpytest.message("!unsubscribe")
        expected = "dhe unsubscribe e ki, katolik i karit a orthodox i mutit a shka pidhsome je"
        assert dpytest.verify().message().content(expected)


class TestArtanCommand:
    """Tests for the !artan command."""

    @pytest.mark.asyncio
    async def test_artan_returns_quote(self, bot_with_mocked_db):
        """Test !artan returns a quote from the collection."""
        bot, _, _, _ = bot_with_mocked_db

        await dpytest.message("!artan")
        # Should return one of the test quotes
        response = dpytest.get_message()
        assert response.content in ["Test quote 1", "Test quote 2"]

    @pytest.mark.asyncio
    async def test_artan_handles_missing_quotes(self, bot_with_mocked_db):
        """Test !artan handles uninitialized quotes gracefully."""
        bot, _, _, cog = bot_with_mocked_db

        # Set artan_quotes to None to simulate unavailable quotes
        cog.artan_quotes = None

        await dpytest.message("!artan")
        assert dpytest.verify().message().content("Quote feature is currently unavailable.")
