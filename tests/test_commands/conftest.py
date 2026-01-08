"""
Fixtures for Discord bot command tests using dpytest.
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import discord
import discord.ext.test as dpytest
import pytest
from discord.ext import commands

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.discord_db import Database
from strofkabot.llumi import LlumiBot
from strofkabot.user_stats import UserStats


@pytest.fixture
def mock_artan_quotes(tmp_path: Path):
    """Create ArtanQuotes with test data."""
    quotes_file = tmp_path / "test_quotes.yaml"
    quotes_file.write_text('- "Test quote 1"\n- "Test quote 2"\n')
    return ArtanQuotes(quotes_file)


@pytest.fixture
async def bot_with_mocked_db(tmp_path: Path, mock_logger: logging.Logger):
    """Create a bot with fully mocked database for isolated tests.

    Returns:
        tuple: (bot, mock_db, mock_user_stats, main_cog)
        Access sub-cogs via bot.get_cog('EntertainmentCog'), etc.
    """
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    intents.reactions = True

    b = commands.Bot(command_prefix="!", intents=intents)
    await b._async_setup_hook()

    # Mock database
    mock_db = AsyncMock(spec=Database)
    mock_user_stats = AsyncMock(spec=UserStats)

    # Create mock artan quotes
    quotes_file = tmp_path / "test_quotes.yaml"
    quotes_file.write_text('- "Test quote 1"\n- "Test quote 2"\n')
    mock_artan = ArtanQuotes(quotes_file)

    cog = LlumiBot(b, mock_db, mock_user_stats, mock_artan, mock_logger)
    await b.add_cog(cog)

    dpytest.configure(b)

    yield b, mock_db, mock_user_stats, cog

    await dpytest.empty_queue()


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a mock logger for testing."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    return logger
