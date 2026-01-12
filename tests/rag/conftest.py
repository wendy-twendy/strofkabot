# pytest configuration for RAG testing

"""Pytest fixtures for RAG tests."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from strofkabot.db.message_history import HistoryMessage


@pytest.fixture
def sample_messages() -> list[HistoryMessage]:
    """Create sample messages for testing conversation grouping."""
    base_time = datetime.datetime(2025, 10, 15, 14, 0, 0, tzinfo=datetime.UTC)

    return [
        # Conversation 1: Simple time-based grouping (3 messages within 5 min)
        HistoryMessage(
            id=1,
            channel_id=100,
            channel_name="kanapeja",
            author_id=301411562487545857,  # Takarak
            author_name="Takarak",
            content="Hey everyone, what's up?",
            timestamp=base_time,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        ),
        HistoryMessage(
            id=2,
            channel_id=100,
            channel_name="kanapeja",
            author_id=686998161163812968,  # basstein
            author_name="basstein",
            content="Not much, just chilling",
            timestamp=base_time + datetime.timedelta(minutes=2),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        ),
        HistoryMessage(
            id=3,
            channel_id=100,
            channel_name="kanapeja",
            author_id=301411562487545857,  # Takarak
            author_name="Takarak",
            content="Same here, pretty quiet day",
            timestamp=base_time + datetime.timedelta(minutes=4),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        ),
        # Conversation 2: Reply chain (outside time window but linked by reply)
        HistoryMessage(
            id=4,
            channel_id=100,
            channel_name="kanapeja",
            author_id=693462918569918514,  # Cappuccino Assassino
            author_name="Cappuccino Assassino",
            content="Did you see the match yesterday?",
            timestamp=base_time + datetime.timedelta(minutes=30),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        ),
        HistoryMessage(
            id=5,
            channel_id=100,
            channel_name="kanapeja",
            author_id=301411562487545857,  # Takarak
            author_name="Takarak",
            content="Yeah it was crazy!",
            timestamp=base_time + datetime.timedelta(minutes=45),
            reply_to_id=4,
            reply_to_author="Cappuccino Assassino",
            reply_to_content="Did you see the match yesterday?",
            reactions="[]",
        ),
        HistoryMessage(
            id=6,
            channel_id=100,
            channel_name="kanapeja",
            author_id=693462918569918514,  # Cappuccino Assassino
            author_name="Cappuccino Assassino",
            content="That last goal was insane",
            timestamp=base_time + datetime.timedelta(minutes=46),
            reply_to_id=5,
            reply_to_author="Takarak",
            reply_to_content="Yeah it was crazy!",
            reactions='[{"emoji": "fire", "count": 3}]',
        ),
        # Different channel (should not be grouped with above)
        HistoryMessage(
            id=7,
            channel_id=200,
            channel_name="muzika",
            author_id=686998161163812968,  # basstein
            author_name="basstein",
            content="Anyone listening to the new album?",
            timestamp=base_time + datetime.timedelta(minutes=35),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        ),
    ]


@pytest.fixture
def sample_nicknames() -> dict[int, list[str]]:
    """Sample nickname mappings for testing."""
    return {
        301411562487545857: ["taka"],  # Takarak
        686998161163812968: ["bas", "bass", "basi"],  # basstein
        693462918569918514: ["jezi", "yeezi", "jez", "yez", "yezi"],  # Cappuccino Assassino
        416623828920172544: ["shark", "sharku", "sharko"],  # Hildegard
    }


@pytest.fixture
def test_db_path() -> Path:
    """Path to the test database."""
    return Path(__file__).parent.parent.parent / "data" / "db.sqlite3"


@pytest.fixture
def nicknames_path() -> Path:
    """Path to the nicknames YAML file."""
    return Path(__file__).parent.parent.parent / "data" / "nicknames.yaml"
