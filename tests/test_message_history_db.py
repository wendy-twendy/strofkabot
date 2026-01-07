"""
Tests for the MessageHistoryDatabase class.
"""

import datetime
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from strofkabot.message_history_db import HistoryMessage, MessageHistoryDatabase


@pytest.fixture
def temp_history_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path for testing."""
    return tmp_path / "test_message_history.db"


@pytest.fixture
async def history_database(
    temp_history_db_path: Path,
) -> AsyncGenerator[MessageHistoryDatabase, None]:
    """Fixture providing an initialized MessageHistoryDatabase instance."""
    db = MessageHistoryDatabase(temp_history_db_path)
    await db.initialize()
    yield db
    await db.close()


class TestMessageHistoryDatabaseInitialization:
    """Tests for database initialization and table creation."""

    async def test_initialize_creates_tables(self, history_database: MessageHistoryDatabase):
        """Verify that initialize() creates the required tables."""
        async with history_database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]

        assert "messages" in table_names
        assert "scrape_progress" in table_names

    async def test_ensure_connection_is_idempotent(self, history_database: MessageHistoryDatabase):
        """Verify that multiple ensure_connection calls don't fail."""
        await history_database.ensure_connection()
        await history_database.ensure_connection()
        assert history_database.conn is not None

    async def test_close_sets_conn_to_none(self, history_database: MessageHistoryDatabase):
        """Verify that close() properly closes and nulls the connection."""
        await history_database.close()
        assert history_database.conn is None


class TestHistoryMessageCRUD:
    """Tests for history message create and read operations."""

    async def test_add_single_message(self, history_database: MessageHistoryDatabase):
        """Test adding a single history message."""
        msg = HistoryMessage(
            id=100,
            channel_id=111222,
            channel_name="general",
            author_id=12345,
            author_name="TestUser",
            content="Test content",
            timestamp=datetime.datetime(2024, 5, 1, 12, 0, 0, tzinfo=datetime.UTC),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions='[{"emoji": "👍", "count": 5}]',
        )
        await history_database.add_messages([msg])

        async with history_database.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (100,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == 100  # id
        assert row[1] == 111222  # channel_id
        assert row[2] == "general"  # channel_name
        assert row[3] == 12345  # author_id
        assert row[4] == "TestUser"  # author_name
        assert row[5] == "Test content"  # content

    async def test_add_multiple_messages(self, history_database: MessageHistoryDatabase):
        """Test batch adding history messages."""
        messages = [
            HistoryMessage(
                id=i,
                channel_id=111222,
                channel_name="general",
                author_id=12345,
                author_name="TestUser",
                content=f"Message {i}",
                timestamp=datetime.datetime.now(datetime.UTC),
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            )
            for i in range(1, 101)
        ]
        await history_database.add_messages(messages)

        count = await history_database.get_message_count()
        assert count == 100

    async def test_add_message_with_reply_info(self, history_database: MessageHistoryDatabase):
        """Test adding a history message that is a reply."""
        msg = HistoryMessage(
            id=200,
            channel_id=111222,
            channel_name="general",
            author_id=12345,
            author_name="ReplyUser",
            content="This is a reply",
            timestamp=datetime.datetime.now(datetime.UTC),
            reply_to_id=100,
            reply_to_author="OriginalUser",
            reply_to_content="Original message content",
            reactions="[]",
        )
        await history_database.add_messages([msg])

        async with history_database.conn.execute(
            "SELECT reply_to_id, reply_to_author, reply_to_content FROM messages WHERE id = ?",
            (200,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 100
        assert row[1] == "OriginalUser"
        assert row[2] == "Original message content"

    async def test_upsert_message_updates_existing(self, history_database: MessageHistoryDatabase):
        """Test that adding a message with existing ID updates it (INSERT OR REPLACE)."""
        msg1 = HistoryMessage(
            id=300,
            channel_id=111222,
            channel_name="general",
            author_id=12345,
            author_name="TestUser",
            content="Original",
            timestamp=datetime.datetime.now(datetime.UTC),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )
        await history_database.add_messages([msg1])

        msg2 = HistoryMessage(
            id=300,
            channel_id=111222,
            channel_name="general",
            author_id=12345,
            author_name="TestUser",
            content="Updated",
            timestamp=datetime.datetime.now(datetime.UTC),
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions='[{"emoji": "🎉", "count": 10}]',
        )
        await history_database.add_messages([msg2])

        async with history_database.conn.execute(
            "SELECT content, reactions FROM messages WHERE id = ?", (300,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Updated"
        assert "🎉" in row[1]


class TestScrapeProgress:
    """Tests for scrape progress tracking."""

    async def test_update_and_get_scrape_progress(self, history_database: MessageHistoryDatabase):
        """Test storing and retrieving scrape progress."""
        channel_id = 111222333
        last_message_id = 999888777

        await history_database.update_scrape_progress(channel_id, last_message_id)
        retrieved = await history_database.get_last_message_id(channel_id)

        assert retrieved == last_message_id

    async def test_get_last_message_id_for_new_channel(
        self, history_database: MessageHistoryDatabase
    ):
        """Test that new channels return None for last message ID."""
        result = await history_database.get_last_message_id(999999999)
        assert result is None

    async def test_update_scrape_progress_overwrites(
        self, history_database: MessageHistoryDatabase
    ):
        """Test that updating progress overwrites the previous value."""
        channel_id = 111222333
        id1 = 100
        id2 = 200

        await history_database.update_scrape_progress(channel_id, id1)
        await history_database.update_scrape_progress(channel_id, id2)

        retrieved = await history_database.get_last_message_id(channel_id)
        assert retrieved == id2


class TestMessageCount:
    """Tests for message count operation."""

    async def test_get_message_count_empty(self, history_database: MessageHistoryDatabase):
        """Test message count on empty database returns 0."""
        count = await history_database.get_message_count()
        assert count == 0

    async def test_get_message_count_with_data(self, history_database: MessageHistoryDatabase):
        """Test message count returns correct number."""
        messages = [
            HistoryMessage(
                id=i,
                channel_id=111222,
                channel_name="general",
                author_id=12345,
                author_name="TestUser",
                content=f"Message {i}",
                timestamp=datetime.datetime.now(datetime.UTC),
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            )
            for i in range(1, 11)
        ]
        await history_database.add_messages(messages)

        count = await history_database.get_message_count()
        assert count == 10


class TestHourlyActivityQuery:
    """Tests for hourly activity aggregation query."""

    async def test_get_hourly_activity_empty_database(
        self, history_database: MessageHistoryDatabase
    ):
        """Test returns empty list for user with no messages."""
        result = await history_database.get_hourly_activity_by_user(999999)
        assert result == []

    async def test_get_hourly_activity_single_message(
        self, history_database: MessageHistoryDatabase
    ):
        """Test aggregation with single message."""
        # Use a recent timestamp (within 3 months)
        recent_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
        msg = HistoryMessage(
            id=100,
            channel_id=111,
            channel_name="general",
            author_id=12345,
            author_name="TestUser",
            content="Test",
            timestamp=recent_time,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )
        await history_database.add_messages([msg])

        result = await history_database.get_hourly_activity_by_user(12345)

        assert len(result) == 1
        day, hour, count = result[0]
        assert hour == recent_time.hour
        assert count == 1

    async def test_get_hourly_activity_with_timezone_offset(
        self, history_database: MessageHistoryDatabase
    ):
        """Test timezone offset shifts hours correctly."""
        # Message at 14:00 UTC should become 15:00 with +1 offset
        recent_time = datetime.datetime.now(datetime.UTC).replace(
            hour=14, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=1)
        msg = HistoryMessage(
            id=101,
            channel_id=111,
            channel_name="general",
            author_id=12345,
            author_name="TestUser",
            content="Test",
            timestamp=recent_time,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )
        await history_database.add_messages([msg])

        result = await history_database.get_hourly_activity_by_user(12345, timezone_offset=1)

        day, hour, count = result[0]
        assert hour == 15  # Shifted by +1

    async def test_get_hourly_activity_multiple_messages_same_slot(
        self, history_database: MessageHistoryDatabase
    ):
        """Test that messages in same hour/day slot are counted together."""
        base_time = datetime.datetime.now(datetime.UTC).replace(
            hour=10, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=7)
        messages = [
            HistoryMessage(
                id=i,
                channel_id=111,
                channel_name="general",
                author_id=12345,
                author_name="TestUser",
                content=f"Test {i}",
                timestamp=base_time + datetime.timedelta(minutes=i),
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            )
            for i in range(5)
        ]
        await history_database.add_messages(messages)

        result = await history_database.get_hourly_activity_by_user(12345)

        assert len(result) == 1
        _, _, count = result[0]
        assert count == 5

    async def test_get_hourly_activity_excludes_old_messages(
        self, history_database: MessageHistoryDatabase
    ):
        """Test that messages older than 3 months are excluded."""
        # Recent message (should be included)
        recent_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
        # Old message (should be excluded - 4 months ago)
        old_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=130)

        messages = [
            HistoryMessage(
                id=1,
                channel_id=111,
                channel_name="general",
                author_id=12345,
                author_name="TestUser",
                content="Recent message",
                timestamp=recent_time,
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            ),
            HistoryMessage(
                id=2,
                channel_id=111,
                channel_name="general",
                author_id=12345,
                author_name="TestUser",
                content="Old message",
                timestamp=old_time,
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            ),
        ]
        await history_database.add_messages(messages)

        result = await history_database.get_hourly_activity_by_user(12345)

        # Should only count the recent message
        total_count = sum(count for _, _, count in result)
        assert total_count == 1

    async def test_get_hourly_activity_filters_by_user(
        self, history_database: MessageHistoryDatabase
    ):
        """Test that only messages from the specified user are counted."""
        recent_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
        messages = [
            HistoryMessage(
                id=1,
                channel_id=111,
                channel_name="general",
                author_id=12345,
                author_name="User1",
                content="User1 message",
                timestamp=recent_time,
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            ),
            HistoryMessage(
                id=2,
                channel_id=111,
                channel_name="general",
                author_id=67890,
                author_name="User2",
                content="User2 message",
                timestamp=recent_time,
                reply_to_id=None,
                reply_to_author=None,
                reply_to_content=None,
                reactions="[]",
            ),
        ]
        await history_database.add_messages(messages)

        result = await history_database.get_hourly_activity_by_user(12345)

        total_count = sum(count for _, _, count in result)
        assert total_count == 1
