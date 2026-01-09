"""
Fixtures for integration tests.

These fixtures provide real database instances (not mocks) for testing
full flows from command execution through database operations.
"""

import datetime
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import discord
import discord.ext.test as dpytest
import pytest
from discord.ext import commands

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.discord_db import Database, Message
from strofkabot.llumi import LlumiBot
from strofkabot.user_stats import UserStats

# ============================================================================
# Real Database Fixtures (not mocked)
# ============================================================================


@pytest.fixture
async def real_database(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """
    Fixture providing a real Database instance for integration tests.

    Unlike the mocked version, this creates an actual SQLite database
    allowing tests to verify full data flows.
    """
    db_path = tmp_path / f"integration_test_{uuid.uuid4().hex[:8]}.sqlite3"
    db = Database(db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def real_user_stats(real_database: Database, integration_logger: logging.Logger) -> UserStats:
    """
    Fixture providing a UserStats instance backed by a real database.
    """
    return UserStats(real_database, logger=integration_logger)


@pytest.fixture
def integration_logger() -> logging.Logger:
    """Logger for integration tests with DEBUG level."""
    logger = logging.getLogger("integration_test")
    logger.setLevel(logging.DEBUG)
    return logger


# ============================================================================
# Factory Fixtures for Flexible Test Data Creation
# ============================================================================


@pytest.fixture
def message_factory(real_database: Database):
    """
    Factory fixture for creating test messages with custom attributes.

    Usage:
        async def test_something(message_factory):
            msg = await message_factory(content="Custom content", reactions=10)
            # ... test with msg
    """
    _counter = [0]  # Use list to allow mutation in nested function

    async def _create_message(
        message_id: int | None = None,
        content: str = "Test message content for integration testing",
        author_id: int = 12345,
        reaction_count: int = 5,
        timestamp: datetime.datetime | None = None,
        reply_to_id: int | None = None,
        reply_to_author: str | None = None,
        reply_to_content: str | None = None,
    ) -> Message:
        """Create and store a message in the database."""
        _counter[0] += 1
        if message_id is None:
            message_id = 1000000 + _counter[0]
        if timestamp is None:
            timestamp = datetime.datetime(2024, 1, 15, 10, 30, _counter[0] % 60)

        msg = Message(
            id=message_id,
            content=content,
            timestamp=timestamp,
            reaction_count=reaction_count,
            author_id=author_id,
            reply_to_id=reply_to_id,
            reply_to_author=reply_to_author,
            reply_to_content=reply_to_content,
        )
        await real_database.add_messages([msg])
        return msg

    return _create_message


@pytest.fixture
def user_stats_factory(real_user_stats: UserStats):
    """
    Factory fixture for creating user statistics data.

    Usage:
        async def test_something(user_stats_factory):
            await user_stats_factory(author_id=123, messages=10, reactions=50)
    """

    async def _create_stats(
        author_id: int,
        display_name: str = "TestUser",
        year: int = 2024,
        month: int = 1,
        message_count: int = 10,
        total_reactions: int = 50,
    ):
        """Create user mapping and stats entries."""
        await real_user_stats.update_user_mapping(author_id, display_name)
        # Create stats by simulating messages
        for _ in range(message_count):
            timestamp = datetime.datetime(year, month, 15, 10, 30, 0)
            await real_user_stats.batch_update_stats(
                [(author_id, total_reactions // message_count, timestamp)]
            )

    return _create_stats


@pytest.fixture
def reaction_factory(real_user_stats: UserStats):
    """
    Factory fixture for creating reaction relationship data.

    Usage:
        async def test_something(reaction_factory):
            await reaction_factory(giver=123, receiver=456, count=5)
    """

    async def _create_reactions(
        giver_id: int,
        receiver_id: int,
        count: int = 1,
        year: int = 2024,
        month: int = 1,
    ):
        """Create reaction relationship entries."""
        timestamp = datetime.datetime(year, month, 15, 10, 30, 0)
        reaction_data = [(giver_id, receiver_id, timestamp) for _ in range(count)]
        await real_user_stats.batch_update_reaction_stats(reaction_data)

    return _create_reactions


# ============================================================================
# Bot with Real Database
# ============================================================================


@pytest.fixture
async def bot_with_real_db(
    real_database: Database,
    real_user_stats: UserStats,
    tmp_path: Path,
    integration_logger: logging.Logger,
):
    """
    Create a bot instance connected to a real database.

    This allows integration tests to verify the full flow from
    command execution through database operations and back.

    Returns:
        tuple: (bot, database, user_stats, cog)
    """
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    intents.reactions = True

    bot = commands.Bot(command_prefix="!", intents=intents)
    await bot._async_setup_hook()

    # Create test quotes file
    quotes_file = tmp_path / "test_quotes.yaml"
    quotes_file.write_text('- "Integration test quote 1"\n- "Integration test quote 2"\n')
    artan_quotes = ArtanQuotes(quotes_file)

    cog = LlumiBot(bot, real_database, real_user_stats, artan_quotes, integration_logger)
    await bot.add_cog(cog)

    dpytest.configure(bot)

    yield bot, real_database, real_user_stats, cog

    await dpytest.empty_queue()
