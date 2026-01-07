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
from strofkabot.discord_db import Database
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
    mock_db = AsyncMock(spec=Database)
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

    mock_db = AsyncMock(spec=Database)
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


class TestUpdateDb:
    """Tests for the update_db method."""

    @pytest.mark.asyncio
    async def test_skips_when_no_guild(self, task_manager):
        """Test that update_db returns early if guild is not set."""
        manager, mock_db, mock_user_stats = task_manager

        manager.guild = None
        await manager.update_db()

        # Should not call any database methods
        mock_db.get_last_scanned_timestamp.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_unreadable_channels(self, task_manager):
        """Test that only channels with read permission are processed."""
        manager, mock_db, mock_user_stats = task_manager

        # Create channels with different permissions
        readable_channel = MagicMock(spec=discord.TextChannel)
        readable_channel.name = "readable"
        readable_channel.id = 111
        readable_perms = MagicMock()
        readable_perms.read_messages = True
        readable_channel.permissions_for.return_value = readable_perms

        async def empty_history(*args, **kwargs):
            return
            yield  # Make it an async generator that yields nothing

        readable_channel.history = empty_history

        unreadable_channel = MagicMock(spec=discord.TextChannel)
        unreadable_channel.name = "unreadable"
        unreadable_channel.id = 222
        unreadable_perms = MagicMock()
        unreadable_perms.read_messages = False
        unreadable_channel.permissions_for.return_value = unreadable_perms

        mock_guild = MagicMock()
        mock_guild.text_channels = [readable_channel, unreadable_channel]
        mock_guild.me = MagicMock()

        manager.guild = mock_guild
        mock_db.get_last_scanned_timestamp.return_value = None

        await manager.update_db()

        # Only readable channel should have permissions checked
        readable_channel.permissions_for.assert_called()


class TestProcessChannel:
    """Tests for the _process_channel method."""

    @pytest.mark.asyncio
    async def test_process_channel_empty(self, task_manager):
        """Test processing channel with no messages."""
        manager, mock_db, mock_user_stats = task_manager

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def empty_history(*args, **kwargs):
            return
            yield  # Empty async generator

        mock_channel.history = empty_history
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should update timestamp even with no messages
        mock_db.update_last_scanned_timestamp.assert_called_once()
        callback.assert_called_once_with(0, 0)

    @pytest.mark.asyncio
    async def test_process_channel_skips_bot_messages(self, mock_logger):
        """Test that bot's own messages are skipped."""
        # Create a fully mocked bot
        mock_bot = MagicMock()
        mock_bot.user = MagicMock()
        mock_bot.user.id = 999

        mock_db = AsyncMock(spec=Database)
        mock_user_stats = AsyncMock(spec=UserStats)
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            bot=mock_bot,
            db=mock_db,
            user_stats=mock_user_stats,
            message_filter=mock_filter,
            logger=mock_logger
        )

        # Create a message from the bot
        bot_message = MagicMock()
        bot_message.author = MagicMock()
        bot_message.author.id = 999  # Same as bot user id
        bot_message.content = "Bot message"
        bot_message.created_at = datetime.datetime.now(datetime.UTC)
        bot_message.reactions = []

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def history_with_bot_message(*args, **kwargs):
            yield bot_message

        mock_channel.history = history_with_bot_message
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should not update stats for bot messages
        mock_user_stats.batch_update_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_channel_skips_empty_content(self, mock_logger):
        """Test that messages with empty content are skipped."""
        mock_bot = MagicMock()
        mock_bot.user = MagicMock()
        mock_bot.user.id = 999

        mock_db = AsyncMock(spec=Database)
        mock_user_stats = AsyncMock(spec=UserStats)
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            bot=mock_bot,
            db=mock_db,
            user_stats=mock_user_stats,
            message_filter=mock_filter,
            logger=mock_logger
        )

        empty_message = MagicMock()
        empty_message.author = MagicMock()
        empty_message.author.id = 12345
        empty_message.content = "   "  # Whitespace only
        empty_message.created_at = datetime.datetime.now(datetime.UTC)
        empty_message.reactions = []

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def history_with_empty(*args, **kwargs):
            yield empty_message

        mock_channel.history = history_with_empty
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should not add stats for empty messages
        mock_user_stats.batch_update_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_channel_filters_low_reactions(self, mock_logger):
        """Test that messages below reaction threshold are not added to messages table."""
        mock_bot = MagicMock()
        mock_bot.user = MagicMock()
        mock_bot.user.id = 999

        mock_db = AsyncMock(spec=Database)
        mock_user_stats = AsyncMock(spec=UserStats)
        mock_filter = MagicMock(spec=MessageFilter)
        mock_filter.is_valid_message.return_value = True

        manager = BackgroundTaskManager(
            bot=mock_bot,
            db=mock_db,
            user_stats=mock_user_stats,
            message_filter=mock_filter,
            logger=mock_logger
        )

        # Message with reactions below threshold (default is 4)
        low_react_message = MagicMock()
        low_react_message.author = MagicMock()
        low_react_message.author.id = 12345
        low_react_message.content = "This is a valid message content"
        low_react_message.created_at = datetime.datetime.now(datetime.UTC)

        reaction = MagicMock()
        reaction.count = 2  # Below threshold of 4
        low_react_message.reactions = [reaction]

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def history_with_low_react(*args, **kwargs):
            yield low_react_message

        mock_channel.history = history_with_low_react
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should not add to messages table (below threshold)
        mock_db.add_messages.assert_not_called()
        # But should still update stats
        mock_user_stats.batch_update_stats.assert_called()

    @pytest.mark.asyncio
    async def test_process_channel_applies_message_filter(self, mock_logger):
        """Test that MessageFilter.is_valid_message is called for qualifying messages."""
        mock_bot = MagicMock()
        mock_bot.user = MagicMock()
        mock_bot.user.id = 999

        mock_db = AsyncMock(spec=Database)
        mock_user_stats = AsyncMock(spec=UserStats)
        mock_filter = MagicMock(spec=MessageFilter)
        mock_filter.is_valid_message.return_value = False  # Filter rejects

        manager = BackgroundTaskManager(
            bot=mock_bot,
            db=mock_db,
            user_stats=mock_user_stats,
            message_filter=mock_filter,
            logger=mock_logger
        )

        high_react_message = MagicMock()
        high_react_message.author = MagicMock()
        high_react_message.author.id = 12345
        high_react_message.content = "http://link.com"  # Would be filtered
        high_react_message.created_at = datetime.datetime.now(datetime.UTC)

        reaction = MagicMock()
        reaction.count = 10  # Above threshold
        high_react_message.reactions = [reaction]

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def history_with_filtered(*args, **kwargs):
            yield high_react_message

        mock_channel.history = history_with_filtered
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Message filter should be called
        mock_filter.is_valid_message.assert_called_with("http://link.com")
        # Should not add to messages table (filter rejected)
        mock_db.add_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_channel_updates_timestamp(self, task_manager):
        """Test that last scanned timestamp is updated after processing."""
        manager, mock_db, mock_user_stats = task_manager

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def empty_history(*args, **kwargs):
            return
            yield

        mock_channel.history = empty_history
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime(2024, 6, 15, tzinfo=datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should update timestamp with scan_until time
        mock_db.update_last_scanned_timestamp.assert_called_once()
        call_args = mock_db.update_last_scanned_timestamp.call_args
        assert call_args[0][0] == 12345  # channel_id

    @pytest.mark.asyncio
    async def test_process_channel_batching(self, mock_logger):
        """Test that stats are batched at 100 items."""
        mock_bot = MagicMock()
        mock_bot.user = MagicMock()
        mock_bot.user.id = 999

        mock_db = AsyncMock(spec=Database)
        mock_user_stats = AsyncMock(spec=UserStats)
        mock_filter = MagicMock(spec=MessageFilter)

        manager = BackgroundTaskManager(
            bot=mock_bot,
            db=mock_db,
            user_stats=mock_user_stats,
            message_filter=mock_filter,
            logger=mock_logger
        )

        # Create 150 messages to trigger batching
        messages = []
        for i in range(150):
            msg = MagicMock()
            msg.author = MagicMock()
            msg.author.id = 12345
            msg.content = f"Message {i} with enough content"
            msg.created_at = datetime.datetime.now(datetime.UTC)
            msg.reactions = []
            messages.append(msg)

        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_channel.name = "test-channel"
        mock_channel.id = 12345

        async def history_with_many(*args, **kwargs):
            for msg in messages:
                yield msg

        mock_channel.history = history_with_many
        mock_db.get_last_scanned_timestamp.return_value = None

        callback = MagicMock()
        scan_until = datetime.datetime.now(datetime.UTC)

        await manager._process_channel(mock_channel, scan_until, callback)

        # Should call batch_update_stats twice: once at 100, once for remaining 50
        assert mock_user_stats.batch_update_stats.call_count == 2


class TestTaskLoops:
    """Tests for LlumiBot task loop methods."""

    @pytest.mark.asyncio
    async def test_update_db_task_calls_update_db(self, llumi_cog):
        """Test that update_db_task executes task_manager.update_db."""
        cog, mock_db, mock_user_stats = llumi_cog

        cog.task_manager.update_db = AsyncMock()

        await cog.update_db_task()

        cog.task_manager.update_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_db_task_handles_exception(self, llumi_cog, caplog):
        """Test that update_db_task logs exception without crashing."""
        cog, mock_db, mock_user_stats = llumi_cog

        cog.task_manager.update_db = AsyncMock(side_effect=Exception("Test error"))

        # Should not raise exception
        await cog.update_db_task()

        # Should log exception
        assert "Error during periodic database update" in caplog.text

    @pytest.mark.asyncio
    async def test_update_usernames_task_calls_update_usernames(self, llumi_cog):
        """Test that update_usernames_task executes task_manager.update_usernames."""
        cog, mock_db, mock_user_stats = llumi_cog

        cog.task_manager.update_usernames = AsyncMock()

        await cog.update_usernames_task()

        cog.task_manager.update_usernames.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_usernames_task_handles_exception(self, llumi_cog, caplog):
        """Test that update_usernames_task logs exception without crashing."""
        cog, mock_db, mock_user_stats = llumi_cog

        cog.task_manager.update_usernames = AsyncMock(side_effect=Exception("Test error"))

        # Should not raise exception
        await cog.update_usernames_task()

        # Should log exception
        assert "Error during periodic username update" in caplog.text
