"""
Shared pytest fixtures for Strofka Discord bot tests.
"""
import datetime
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from strofkabot.discord_db import Message, MessageDatabase
from strofkabot.message_filter import MessageFilter
from strofkabot.user_stats import UserStats

# ============================================================================
# Async Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for testing."""
    return tmp_path / "test_db.sqlite3"


@pytest.fixture
async def message_database(temp_db_path: Path) -> AsyncGenerator[MessageDatabase, None]:
    """
    Fixture providing an initialized MessageDatabase instance.

    Yields an initialized database, then cleans up after the test.
    """
    db = MessageDatabase(temp_db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a mock logger for testing."""
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
async def user_stats_db(temp_db_path: Path, mock_logger: logging.Logger) -> AsyncGenerator[UserStats, None]:
    """
    Fixture providing an initialized UserStats instance.

    Yields an initialized database, then cleans up after the test.
    """
    db = UserStats(str(temp_db_path), logger=mock_logger)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def populated_message_db(message_database: MessageDatabase) -> MessageDatabase:
    """
    Fixture providing a MessageDatabase with sample messages.
    """
    sample_messages = [
        Message(
            id=1,
            content="Hello, this is a test message",
            timestamp=datetime.datetime(2024, 1, 15, 10, 30, 0),
            reaction_count=5,
            author_id=12345,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None
        ),
        Message(
            id=2,
            content="Another test message with reactions",
            timestamp=datetime.datetime(2024, 1, 16, 14, 20, 0),
            reaction_count=10,
            author_id=67890,
            reply_to_id=1,
            reply_to_author="User12345",
            reply_to_content="Hello, this is a test message"
        ),
        Message(
            id=3,
            content="Third message for variety",
            timestamp=datetime.datetime(2024, 2, 1, 8, 0, 0),
            reaction_count=3,
            author_id=12345,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None
        ),
    ]
    await message_database.add_messages(sample_messages)
    return message_database


@pytest.fixture
async def populated_user_stats(user_stats_db: UserStats) -> UserStats:
    """
    Fixture providing a UserStats database with sample data.
    """
    # Add user mappings
    await user_stats_db.update_user_mapping(12345, "Alice")
    await user_stats_db.update_user_mapping(67890, "Bob")
    await user_stats_db.update_user_mapping(11111, "Charlie")

    # Add monthly stats
    stats_data = [
        (12345, 5, datetime.datetime(2024, 1, 15)),  # author_id, reaction_count, timestamp
        (12345, 3, datetime.datetime(2024, 1, 20)),
        (67890, 10, datetime.datetime(2024, 1, 18)),
        (11111, 2, datetime.datetime(2024, 1, 25)),
        (12345, 8, datetime.datetime(2024, 2, 5)),
    ]
    await user_stats_db.batch_update_stats(stats_data)

    # Add reaction data
    reaction_data = [
        (12345, 67890, datetime.datetime(2024, 1, 15)),  # giver_id, receiver_id, timestamp
        (12345, 67890, datetime.datetime(2024, 1, 16)),
        (67890, 12345, datetime.datetime(2024, 1, 17)),
        (11111, 12345, datetime.datetime(2024, 1, 18)),
    ]
    await user_stats_db.batch_update_reaction_stats(reaction_data)

    return user_stats_db


# ============================================================================
# Synchronous Fixtures
# ============================================================================

@pytest.fixture
def message_filter() -> MessageFilter:
    """Fixture to create an instance of MessageFilter."""
    return MessageFilter()


@pytest.fixture
def sample_monthly_data() -> list:
    """Sample monthly inflation data for testing (ascending order - correct)."""
    return [
        {'year': 2024, 'month': 1, 'average_rpm': 2.0, 'total_reactions': 400, 'total_messages': 200},
        {'year': 2024, 'month': 2, 'average_rpm': 2.3, 'total_reactions': 460, 'total_messages': 200},
        {'year': 2024, 'month': 3, 'average_rpm': 2.5, 'total_reactions': 500, 'total_messages': 200},
    ]


@pytest.fixture
def sample_monthly_data_descending() -> list:
    """Sample monthly data in descending order (as incorrectly returned by old DB query).

    This fixture documents the bug where data was returned newest-first,
    causing inflation percentages to have inverted signs.
    """
    return [
        {'year': 2024, 'month': 3, 'average_rpm': 2.5, 'total_reactions': 500, 'total_messages': 200},
        {'year': 2024, 'month': 2, 'average_rpm': 2.3, 'total_reactions': 460, 'total_messages': 200},
        {'year': 2024, 'month': 1, 'average_rpm': 2.0, 'total_reactions': 400, 'total_messages': 200},
    ]


@pytest.fixture
def sample_yearly_data() -> list:
    """Sample yearly inflation data for testing."""
    return [
        {'year': 2022, 'average_rpm': 1.8, 'total_reactions': 3600, 'total_messages': 2000},
        {'year': 2023, 'average_rpm': 2.0, 'total_reactions': 4000, 'total_messages': 2000},
        {'year': 2024, 'average_rpm': 2.5, 'total_reactions': 5000, 'total_messages': 2000},
    ]


@pytest.fixture
def sample_gdp_data() -> list:
    """Sample GDP data for testing plots."""
    return [
        {'year': 2024, 'month': 3, 'total_messages': 1500},
        {'year': 2024, 'month': 2, 'total_messages': 1200},
        {'year': 2024, 'month': 1, 'total_messages': 1100},
    ]


@pytest.fixture
def sample_hdi_data() -> list:
    """Sample HDI data for testing plots."""
    return [
        {'year': 2024, 'month': 3, 'quality_count': 150, 'total_count': 1500, 'hdi_ratio': 0.10},
        {'year': 2024, 'month': 2, 'quality_count': 108, 'total_count': 1200, 'hdi_ratio': 0.09},
        {'year': 2024, 'month': 1, 'quality_count': 88, 'total_count': 1100, 'hdi_ratio': 0.08},
    ]
