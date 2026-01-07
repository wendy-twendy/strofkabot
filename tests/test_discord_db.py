"""
Tests for the Database class.
"""
import datetime

from strofkabot.discord_db import Database, Message


class TestDatabaseInitialization:
    """Tests for database initialization and table creation."""

    async def test_initialize_creates_tables(self, message_database: Database):
        """Verify that initialize() creates the required tables."""
        async with message_database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]

        assert 'messages' in table_names
        assert 'metadata' in table_names

    async def test_ensure_connection_is_idempotent(self, message_database: Database):
        """Verify that multiple ensure_connection calls don't fail."""
        await message_database.ensure_connection()
        await message_database.ensure_connection()
        assert message_database.conn is not None

    async def test_close_sets_conn_to_none(self, message_database: Database):
        """Verify that close() properly closes and nulls the connection."""
        await message_database.close()
        assert message_database.conn is None


class TestMessageCRUD:
    """Tests for message create, read, update operations."""

    async def test_add_single_message(self, message_database: Database):
        """Test adding a single message."""
        msg = Message(
            id=100,
            content="Test content",
            timestamp=datetime.datetime(2024, 5, 1, 12, 0, 0),
            reaction_count=5,
            author_id=999
        )
        await message_database.add_messages([msg])

        async with message_database.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (100,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[1] == "Test content"
        assert row[3] == 5

    async def test_add_multiple_messages(self, message_database: Database):
        """Test batch adding messages."""
        messages = [
            Message(id=i, content=f"Message {i}", timestamp=datetime.datetime.now(),
                    reaction_count=i, author_id=1000+i)
            for i in range(1, 6)
        ]
        await message_database.add_messages(messages)

        async with message_database.conn.execute("SELECT COUNT(*) FROM messages") as cursor:
            count = (await cursor.fetchone())[0]

        assert count == 5

    async def test_add_message_with_reply_info(self, message_database: Database):
        """Test adding a message that is a reply."""
        msg = Message(
            id=200,
            content="This is a reply",
            timestamp=datetime.datetime.now(),
            reaction_count=2,
            author_id=1001,
            reply_to_id=100,
            reply_to_author="OriginalAuthor",
            reply_to_content="Original message content"
        )
        await message_database.add_messages([msg])

        async with message_database.conn.execute(
            "SELECT reply_to_id, reply_to_author, reply_to_content FROM messages WHERE id = ?",
            (200,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 100
        assert row[1] == "OriginalAuthor"
        assert row[2] == "Original message content"

    async def test_upsert_message_updates_existing(self, message_database: Database):
        """Test that adding a message with existing ID updates it (INSERT OR REPLACE)."""
        msg1 = Message(id=300, content="Original", timestamp=datetime.datetime.now(),
                       reaction_count=1, author_id=1002)
        await message_database.add_messages([msg1])

        msg2 = Message(id=300, content="Updated", timestamp=datetime.datetime.now(),
                       reaction_count=10, author_id=1002)
        await message_database.add_messages([msg2])

        async with message_database.conn.execute(
            "SELECT content, reaction_count FROM messages WHERE id = ?", (300,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Updated"
        assert row[1] == 10


class TestRandomMessageRetrieval:
    """Tests for random message retrieval."""

    async def test_get_random_message_returns_message_object(
        self, populated_message_db: Database
    ):
        """Test that get_random_message returns a Message dataclass."""
        msg = await populated_message_db.get_random_message()

        assert msg is not None
        assert isinstance(msg, Message)
        assert msg.id in [1, 2, 3]

    async def test_get_random_message_empty_database(self, message_database: Database):
        """Test get_random_message on empty database returns None."""
        msg = await message_database.get_random_message()
        assert msg is None

    async def test_random_message_timestamp_is_datetime(
        self, populated_message_db: Database
    ):
        """Verify that timestamp is properly converted to datetime object."""
        msg = await populated_message_db.get_random_message()
        assert isinstance(msg.timestamp, datetime.datetime)


class TestTimestampHandling:
    """Tests for timestamp metadata operations."""

    async def test_update_and_get_last_scanned_timestamp(self, message_database: Database):
        """Test storing and retrieving channel scan timestamps."""
        channel_id = 123456789
        timestamp = datetime.datetime(2024, 6, 15, 10, 30, 0)

        await message_database.update_last_scanned_timestamp(channel_id, timestamp)
        retrieved = await message_database.get_last_scanned_timestamp(channel_id)

        assert retrieved == timestamp

    async def test_get_timestamp_for_unscanned_channel(self, message_database: Database):
        """Test that unscanned channels return None."""
        result = await message_database.get_last_scanned_timestamp(999999999)
        assert result is None

    async def test_update_timestamp_overwrites(self, message_database: Database):
        """Test that updating timestamp overwrites the previous value."""
        channel_id = 111222333
        ts1 = datetime.datetime(2024, 1, 1, 0, 0, 0)
        ts2 = datetime.datetime(2024, 6, 1, 0, 0, 0)

        await message_database.update_last_scanned_timestamp(channel_id, ts1)
        await message_database.update_last_scanned_timestamp(channel_id, ts2)

        retrieved = await message_database.get_last_scanned_timestamp(channel_id)
        assert retrieved == ts2
