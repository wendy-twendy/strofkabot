"""
Tests for background tasks and helper functions in LlumiBot.
"""
import datetime
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.discord_db import MessageDatabase
from strofkabot.llumi import LlumiBot, setup_logging
from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager
from strofkabot.user_stats import UserStats


@pytest.fixture
async def llumi_cog(tmp_path: Path, mock_logger: logging.Logger):
    """Create a LlumiBot cog with mocked dependencies for testing."""
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True

    bot = commands.Bot(command_prefix='!', intents=intents)
    await bot._async_setup_hook()

    # Mock database
    mock_db = AsyncMock(spec=MessageDatabase)
    mock_user_stats = AsyncMock(spec=UserStats)

    # Create mock artan quotes
    quotes_file = tmp_path / "test_quotes.yaml"
    quotes_file.write_text('- "Test quote"\n')
    mock_artan = ArtanQuotes(quotes_file)

    cog = LlumiBot(bot, mock_db, mock_user_stats, mock_artan, mock_logger)

    yield cog, mock_db, mock_user_stats


@pytest.fixture
async def task_manager(mock_logger: logging.Logger):
    """Create a BackgroundTaskManager with mocked dependencies for testing."""
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True

    bot = commands.Bot(command_prefix='!', intents=intents)
    await bot._async_setup_hook()

    mock_db = AsyncMock(spec=MessageDatabase)
    mock_user_stats = AsyncMock(spec=UserStats)
    mock_filter = MagicMock(spec=MessageFilter)

    manager = BackgroundTaskManager(
        bot=bot,
        db=mock_db,
        user_stats=mock_user_stats,
        message_filter=mock_filter,
        logger=mock_logger
    )

    yield manager, mock_db, mock_user_stats


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_debug_level(self):
        """Test setting DEBUG log level."""
        logger = setup_logging('DEBUG')
        assert logger.level == logging.DEBUG
        assert logger.name == 'LlumiBot'

    def test_info_level(self):
        """Test setting INFO log level."""
        logger = setup_logging('INFO')
        assert logger.level == logging.INFO

    def test_warning_level(self):
        """Test setting WARNING log level."""
        logger = setup_logging('WARNING')
        assert logger.level == logging.WARNING

    def test_error_level(self):
        """Test setting ERROR log level."""
        logger = setup_logging('ERROR')
        assert logger.level == logging.ERROR

    def test_critical_level(self):
        """Test setting CRITICAL log level."""
        logger = setup_logging('CRITICAL')
        assert logger.level == logging.CRITICAL

    def test_returns_logger(self):
        """Test that function returns a logger instance."""
        logger = setup_logging('INFO')
        assert isinstance(logger, logging.Logger)
        assert logger.handlers  # Has at least one handler

    def test_lowercase_level(self):
        """Test handling lowercase level strings."""
        logger = setup_logging('debug')
        assert logger.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        """Test invalid level defaults to INFO."""
        logger = setup_logging('INVALID')
        # getattr with default returns logging.INFO when attribute not found
        assert logger.level == logging.INFO


class TestProcessReactions:
    """Tests for the _process_reactions method."""

    @pytest.mark.asyncio
    async def test_no_reactions(self, task_manager):
        """Test processing message with no reactions."""
        manager, _, mock_user_stats = task_manager

        mock_message = MagicMock()
        mock_message.reactions = []
        mock_message.author.id = 12345

        await manager._process_reactions(mock_message)

        # Should not call update_reaction_stats
        mock_user_stats.update_reaction_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_reactions(self, task_manager):
        """Test processing message with multiple reactions."""
        manager, _, mock_user_stats = task_manager

        # Create mock users
        user1 = MagicMock()
        user1.bot = False
        user1.id = 111

        user2 = MagicMock()
        user2.bot = False
        user2.id = 222

        # Create mock reaction with async users iterator
        mock_reaction = MagicMock()

        async def mock_users():
            for user in [user1, user2]:
                yield user

        mock_reaction.users = mock_users

        mock_message = MagicMock()
        mock_message.reactions = [mock_reaction]
        mock_message.author = MagicMock()
        mock_message.author.id = 12345
        mock_message.created_at = datetime.datetime(2024, 1, 15, tzinfo=datetime.UTC)

        await manager._process_reactions(mock_message)

        # Should call update_reaction_stats for each non-bot user
        assert mock_user_stats.update_reaction_stats.call_count == 2

    @pytest.mark.asyncio
    async def test_filters_bot_reactors(self, task_manager):
        """Test that bot reactors are filtered out."""
        manager, _, mock_user_stats = task_manager

        # Create bot user and human user
        bot_user = MagicMock()
        bot_user.bot = True
        bot_user.id = 999

        human_user = MagicMock()
        human_user.bot = False
        human_user.id = 111

        mock_reaction = MagicMock()

        async def mock_users():
            for user in [bot_user, human_user]:
                yield user

        mock_reaction.users = mock_users

        mock_message = MagicMock()
        mock_message.reactions = [mock_reaction]
        mock_message.author = MagicMock()
        mock_message.author.id = 12345
        mock_message.created_at = datetime.datetime(2024, 1, 15, tzinfo=datetime.UTC)

        await manager._process_reactions(mock_message)

        # Only human user should trigger update
        assert mock_user_stats.update_reaction_stats.call_count == 1


class TestUpdateUsernames:
    """Tests for the update_usernames method."""

    @pytest.mark.asyncio
    async def test_skips_when_no_guild(self, task_manager):
        """Test that update skips when guild is not set."""
        manager, _, mock_user_stats = task_manager

        manager.guild = None
        await manager.update_usernames()

        mock_user_stats.update_user_mapping.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_cooldown(self, task_manager):
        """Test that update is skipped if less than 24 hours since last update."""
        manager, _, mock_user_stats = task_manager

        manager.guild = MagicMock()
        manager.guild.members = []
        # Set last update to now (less than 24 hours ago)
        manager.last_username_update = datetime.datetime.now(datetime.UTC)

        await manager.update_usernames()

        mock_user_stats.update_user_mapping.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_all_members(self, task_manager):
        """Test that all members' usernames are updated."""
        manager, _, mock_user_stats = task_manager

        member1 = MagicMock()
        member1.id = 111
        member1.display_name = "Alice"

        member2 = MagicMock()
        member2.id = 222
        member2.display_name = "Bob"

        manager.guild = MagicMock()
        manager.guild.name = "Test Guild"
        manager.guild.id = 12345
        manager.guild.member_count = 2
        manager.guild.members = [member1, member2]
        # Set last update to more than 24 hours ago
        manager.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        await manager.update_usernames()

        assert mock_user_stats.update_user_mapping.call_count == 2

    @pytest.mark.asyncio
    async def test_strips_emojis(self, task_manager):
        """Test that emojis are removed from display names."""
        manager, _, mock_user_stats = task_manager

        member = MagicMock()
        member.id = 111
        member.display_name = "Alice🎭🎨Test"

        manager.guild = MagicMock()
        manager.guild.name = "Test Guild"
        manager.guild.id = 12345
        manager.guild.member_count = 1
        manager.guild.members = [member]
        manager.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        await manager.update_usernames()

        # Verify the cleaned name was used (without emojis)
        call_args = mock_user_stats.update_user_mapping.call_args
        assert call_args[0][0] == 111  # user_id
        # The cleaned name should not contain emojis
        cleaned_name = call_args[0][1]
        assert "🎭" not in cleaned_name
        assert "🎨" not in cleaned_name


class TestGetAllReacts:
    """Tests for the _get_all_reacts method."""

    @pytest.mark.asyncio
    async def test_no_reactions(self, task_manager):
        """Test message with no reactions returns 0."""
        manager, _, _ = task_manager

        mock_message = MagicMock()
        mock_message.reactions = []

        result = manager._get_all_reacts(mock_message)

        assert result == 0

    @pytest.mark.asyncio
    async def test_single_reaction(self, task_manager):
        """Test message with single reaction type."""
        manager, _, _ = task_manager

        reaction = MagicMock()
        reaction.count = 5

        mock_message = MagicMock()
        mock_message.reactions = [reaction]

        result = manager._get_all_reacts(mock_message)

        assert result == 5

    @pytest.mark.asyncio
    async def test_multiple_reactions(self, task_manager):
        """Test message with multiple reaction types."""
        manager, _, _ = task_manager

        reaction1 = MagicMock()
        reaction1.count = 5

        reaction2 = MagicMock()
        reaction2.count = 3

        reaction3 = MagicMock()
        reaction3.count = 2

        mock_message = MagicMock()
        mock_message.reactions = [reaction1, reaction2, reaction3]

        result = manager._get_all_reacts(mock_message)

        assert result == 10
