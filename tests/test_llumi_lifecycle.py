"""
Tests for LlumiBot lifecycle methods.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.config import GUILD_ID
from strofkabot.discord_db import Database
from strofkabot.llumi import LlumiBot
from strofkabot.user_stats import UserStats


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_dependencies(tmp_path: Path, mock_logger):
    """Create mock dependencies for LlumiBot."""
    mock_db = AsyncMock(spec=Database)
    mock_user_stats = AsyncMock(spec=UserStats)

    # Create test quotes file
    quotes_file = tmp_path / "test_quotes.yaml"
    quotes_file.write_text('- "Test quote"\n')
    artan_quotes = ArtanQuotes(quotes_file)

    return mock_db, mock_user_stats, artan_quotes, mock_logger


@pytest.fixture
async def llumi_cog(mock_dependencies):
    """Create a LlumiBot cog for testing."""
    mock_db, mock_user_stats, artan_quotes, mock_logger = mock_dependencies

    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True

    bot = commands.Bot(command_prefix="!", intents=intents)
    await bot._async_setup_hook()

    cog = LlumiBot(bot, mock_db, mock_user_stats, artan_quotes, None, mock_logger)
    return cog, mock_db, mock_user_stats, mock_logger


class TestLlumiBotInit:
    """Tests for LlumiBot.__init__."""

    @pytest.mark.asyncio
    async def test_attributes_initialized(self, llumi_cog):
        """Verify all attributes are initialized correctly."""
        cog, mock_db, mock_user_stats, mock_logger = llumi_cog

        assert cog.db is mock_db
        assert cog.user_stats is mock_user_stats
        assert cog.logger is mock_logger
        assert cog.guild is None  # Initially None
        assert cog.bot is not None
        assert cog.task_manager is not None

    @pytest.mark.asyncio
    async def test_task_manager_created(self, llumi_cog):
        """Verify task manager is created with correct dependencies."""
        cog, mock_db, mock_user_stats, mock_logger = llumi_cog

        assert cog.task_manager.db is mock_db
        assert cog.task_manager.user_stats is mock_user_stats
        assert cog.task_manager.logger is mock_logger


class TestCogLoad:
    """Tests for LlumiBot.cog_load."""

    @pytest.mark.asyncio
    async def test_initializes_database(self, llumi_cog):
        """Test that cog_load calls db.initialize()."""
        cog, mock_db, _, _ = llumi_cog

        await cog.cog_load()

        mock_db.initialize.assert_called_once()


class TestOnReady:
    """Tests for LlumiBot.on_ready."""

    @pytest.mark.asyncio
    async def test_sets_guild(self, mock_dependencies):
        """Test that on_ready retrieves and sets the guild."""
        mock_db, mock_user_stats, artan_quotes, mock_logger = mock_dependencies

        # Create a mock bot with user and get_guild
        mock_bot = MagicMock(spec=commands.Bot)
        mock_bot.user = MagicMock()
        mock_bot.user.id = 12345
        mock_bot.user.__str__ = lambda self: "TestBot#1234"

        mock_guild = MagicMock(spec=discord.Guild)
        mock_guild.name = "Test Guild"
        mock_bot.get_guild.return_value = mock_guild

        cog = LlumiBot(mock_bot, mock_db, mock_user_stats, artan_quotes, None, mock_logger)
        # Mock the task methods to prevent actual task starting
        cog.update_db_task = MagicMock()
        cog.update_usernames_task = MagicMock()

        await cog.on_ready()

        mock_bot.get_guild.assert_called_once_with(GUILD_ID)
        assert cog.guild is mock_guild

    @pytest.mark.asyncio
    async def test_guild_not_found(self, mock_dependencies):
        """Test that on_ready logs error when guild not found."""
        mock_db, mock_user_stats, artan_quotes, mock_logger = mock_dependencies

        mock_bot = MagicMock(spec=commands.Bot)
        mock_bot.user = MagicMock()
        mock_bot.user.id = 12345
        mock_bot.user.__str__ = lambda self: "TestBot#1234"
        mock_bot.get_guild.return_value = None  # Guild not found

        cog = LlumiBot(mock_bot, mock_db, mock_user_stats, artan_quotes, None, mock_logger)

        await cog.on_ready()

        # Should log error
        mock_logger.error.assert_called()
        assert cog.guild is None

    @pytest.mark.asyncio
    async def test_starts_tasks(self, mock_dependencies):
        """Test that on_ready starts background tasks."""
        mock_db, mock_user_stats, artan_quotes, mock_logger = mock_dependencies

        mock_bot = MagicMock(spec=commands.Bot)
        mock_bot.user = MagicMock()
        mock_bot.user.id = 12345
        mock_bot.user.__str__ = lambda self: "TestBot#1234"

        mock_guild = MagicMock(spec=discord.Guild)
        mock_guild.name = "Test Guild"
        mock_bot.get_guild.return_value = mock_guild

        cog = LlumiBot(mock_bot, mock_db, mock_user_stats, artan_quotes, None, mock_logger)
        # Mock the task start methods
        cog.update_db_task = MagicMock()
        cog.update_usernames_task = MagicMock()

        await cog.on_ready()

        cog.update_db_task.start.assert_called_once()
        cog.update_usernames_task.start.assert_called_once()
