"""
Tests for the !llumi command.
"""
import datetime

import discord.ext.test as dpytest
import pytest

from strofkabot.discord_db import Message


class TestLlumiCommand:
    """Tests for the !llumi command."""

    @pytest.mark.asyncio
    async def test_llumi_returns_random_message(self, bot_with_mocked_db):
        """Test that !llumi returns a message from the database."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_random_message.return_value = Message(
            id=12345,
            content="This is a test message from the database",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=99999
        )

        await dpytest.message("!llumi")
        assert dpytest.verify().message().content("This is a test message from the database")

    @pytest.mark.asyncio
    async def test_llumi_handles_empty_database(self, bot_with_mocked_db):
        """Test that !llumi handles empty database gracefully."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_random_message.return_value = None

        await dpytest.message("!llumi")
        assert dpytest.verify().message().content("No messages available at the moment.")

    @pytest.mark.asyncio
    async def test_llumi_calls_get_random_message(self, bot_with_mocked_db):
        """Test that !llumi calls get_random_message on the database."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_random_message.return_value = Message(
            id=1,
            content="Test",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=1
        )

        await dpytest.message("!llumi")
        mock_db.get_random_message.assert_called_once()
