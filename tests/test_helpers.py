"""
Tests for helper methods in the BackgroundTaskManager.
"""
import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager


class TestGetAllReacts:
    """Tests for the _get_all_reacts helper method."""

    def test_get_all_reacts_sums_reactions(self):
        """Test _get_all_reacts correctly sums all reaction counts."""
        mock_bot = MagicMock()
        mock_db = MagicMock()
        mock_user_stats = MagicMock()
        mock_filter = MagicMock(spec=MessageFilter)
        mock_logger = MagicMock()

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )

        # Create mock message with reactions
        mock_message = MagicMock()
        reaction1 = MagicMock()
        reaction1.count = 5
        reaction2 = MagicMock()
        reaction2.count = 3
        mock_message.reactions = [reaction1, reaction2]

        result = manager._get_all_reacts(mock_message)

        assert result == 8

    def test_get_all_reacts_handles_no_reactions(self):
        """Test _get_all_reacts returns 0 for messages without reactions."""
        mock_bot = MagicMock()
        mock_db = MagicMock()
        mock_user_stats = MagicMock()
        mock_filter = MagicMock(spec=MessageFilter)
        mock_logger = MagicMock()

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )

        mock_message = MagicMock()
        mock_message.reactions = []

        result = manager._get_all_reacts(mock_message)

        assert result == 0

    def test_get_all_reacts_handles_single_reaction(self):
        """Test _get_all_reacts with a single reaction."""
        mock_bot = MagicMock()
        mock_db = MagicMock()
        mock_user_stats = MagicMock()
        mock_filter = MagicMock(spec=MessageFilter)
        mock_logger = MagicMock()

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )

        mock_message = MagicMock()
        reaction = MagicMock()
        reaction.count = 10
        mock_message.reactions = [reaction]

        result = manager._get_all_reacts(mock_message)

        assert result == 10


class TestUpdateDb:
    """Tests for the update_db method."""

    @pytest.mark.asyncio
    async def test_update_db_skips_when_no_guild(self, mock_logger):
        """Test update_db does nothing when guild is not set."""
        mock_bot = MagicMock()
        mock_db = AsyncMock()
        mock_user_stats = AsyncMock()
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )
        manager.guild = None

        await manager.update_db()

        mock_db.add_messages.assert_not_called()


class TestUpdateUsernames:
    """Tests for the update_usernames method."""

    @pytest.mark.asyncio
    async def test_update_usernames_skips_when_no_guild(self, mock_logger):
        """Test update_usernames does nothing when guild is not set."""
        mock_bot = MagicMock()
        mock_db = AsyncMock()
        mock_user_stats = AsyncMock()
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )
        manager.guild = None

        await manager.update_usernames()

        mock_user_stats.update_user_mapping.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_usernames_respects_cooldown(self, mock_logger):
        """Test update_usernames skips if called within 24 hours."""
        mock_bot = MagicMock()
        mock_db = AsyncMock()
        mock_user_stats = AsyncMock()
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            mock_bot, mock_db, mock_user_stats, mock_filter, mock_logger
        )

        # Set up mock guild
        mock_guild = MagicMock()
        mock_guild.members = []
        manager.guild = mock_guild

        # Set last update to recent time
        manager.last_username_update = datetime.datetime.now(datetime.UTC)

        await manager.update_usernames()

        mock_user_stats.update_user_mapping.assert_not_called()
