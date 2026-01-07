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

        assert "messages" in table_names
        assert "metadata" in table_names

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
            author_id=999,
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
            Message(
                id=i,
                content=f"Message {i}",
                timestamp=datetime.datetime.now(),
                reaction_count=i,
                author_id=1000 + i,
            )
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
            reply_to_content="Original message content",
        )
        await message_database.add_messages([msg])

        async with message_database.conn.execute(
            "SELECT reply_to_id, reply_to_author, reply_to_content FROM messages WHERE id = ?",
            (200,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 100
        assert row[1] == "OriginalAuthor"
        assert row[2] == "Original message content"

    async def test_upsert_message_updates_existing(self, message_database: Database):
        """Test that adding a message with existing ID updates it (INSERT OR REPLACE)."""
        msg1 = Message(
            id=300,
            content="Original",
            timestamp=datetime.datetime.now(),
            reaction_count=1,
            author_id=1002,
        )
        await message_database.add_messages([msg1])

        msg2 = Message(
            id=300,
            content="Updated",
            timestamp=datetime.datetime.now(),
            reaction_count=10,
            author_id=1002,
        )
        await message_database.add_messages([msg2])

        async with message_database.conn.execute(
            "SELECT content, reaction_count FROM messages WHERE id = ?", (300,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Updated"
        assert row[1] == 10


class TestRandomMessageRetrieval:
    """Tests for random message retrieval."""

    async def test_get_random_message_returns_message_object(self, populated_message_db: Database):
        """Test that get_random_message returns a Message dataclass."""
        msg = await populated_message_db.get_random_message()

        assert msg is not None
        assert isinstance(msg, Message)
        assert msg.id in [1, 2, 3]

    async def test_get_random_message_empty_database(self, message_database: Database):
        """Test get_random_message on empty database returns None."""
        msg = await message_database.get_random_message()
        assert msg is None

    async def test_random_message_timestamp_is_datetime(self, populated_message_db: Database):
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


class TestUserStatsOperations:
    """Tests for user statistics CRUD operations."""

    async def test_upsert_user_stats_insert(self, message_database: Database):
        """Test inserting new user stats."""
        await message_database.upsert_user_stats(
            author_id=12345, year=2024, month=1, reaction_count=5
        )

        async with message_database.conn.execute(
            "SELECT total_messages, total_reactions FROM user_stats_monthly WHERE author_id = ?",
            (12345,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1  # total_messages
        assert row[1] == 5  # total_reactions

    async def test_upsert_user_stats_update(self, message_database: Database):
        """Test that upsert increments message count and adds reactions."""
        await message_database.upsert_user_stats(12345, 2024, 1, 5)
        await message_database.upsert_user_stats(12345, 2024, 1, 3)

        async with message_database.conn.execute(
            "SELECT total_messages, total_reactions FROM user_stats_monthly WHERE author_id = ?",
            (12345,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 2  # total_messages incremented
        assert row[1] == 8  # total_reactions accumulated

    async def test_batch_upsert_user_stats(self, message_database: Database):
        """Test batch inserting/updating user stats."""
        stats = [
            (12345, 2024, 1, 5),
            (12345, 2024, 1, 3),
            (67890, 2024, 1, 10),
        ]
        await message_database.batch_upsert_user_stats(stats)

        async with message_database.conn.execute(
            "SELECT author_id, total_messages, total_reactions FROM user_stats_monthly ORDER BY author_id"
        ) as cursor:
            rows = await cursor.fetchall()

        assert len(rows) == 2
        assert rows[0] == (12345, 2, 8)  # Two messages, 5+3 reactions
        assert rows[1] == (67890, 1, 10)

    async def test_upsert_user_mapping(self, message_database: Database):
        """Test inserting and updating user mapping."""
        await message_database.upsert_user_mapping(12345, "Alice")

        async with message_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (12345,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Alice"

        # Update the name
        await message_database.upsert_user_mapping(12345, "Alice_New")

        async with message_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (12345,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Alice_New"

    async def test_fetch_monthly_stats_with_data(self, message_database: Database):
        """Test fetching monthly stats with user mapping JOIN."""
        await message_database.upsert_user_mapping(12345, "Alice")
        await message_database.upsert_user_mapping(67890, "Bob")
        await message_database.upsert_user_stats(12345, 2024, 1, 10)
        await message_database.upsert_user_stats(12345, 2024, 1, 10)
        await message_database.upsert_user_stats(67890, 2024, 1, 5)

        result = await message_database.fetch_monthly_stats(2024, 1)

        assert len(result) == 2
        # Results should be ordered by avg_reactions DESC
        # Alice: 20 reactions / 2 messages = 10 avg
        # Bob: 5 reactions / 1 message = 5 avg
        assert result[0][1] == "Alice"  # username
        assert result[0][4] == 10.0  # avg_reactions
        assert result[1][1] == "Bob"

    async def test_fetch_monthly_stats_empty(self, message_database: Database):
        """Test fetching monthly stats for empty month returns empty list."""
        result = await message_database.fetch_monthly_stats(2024, 12)
        assert result == []

    async def test_fetch_user_monthly_stats_exists(self, message_database: Database):
        """Test fetching specific user's monthly stats."""
        await message_database.upsert_user_stats(12345, 2024, 1, 10)
        await message_database.upsert_user_stats(12345, 2024, 1, 10)

        result = await message_database.fetch_user_monthly_stats(12345, 2024, 1)

        assert result is not None
        assert result[0] == 2  # total_messages
        assert result[1] == 20  # total_reactions
        assert result[2] == 10.0  # avg_reactions

    async def test_fetch_user_monthly_stats_not_found(self, message_database: Database):
        """Test fetching stats for non-existent user returns None."""
        result = await message_database.fetch_user_monthly_stats(99999, 2024, 1)
        assert result is None


class TestReactionStatsOperations:
    """Tests for reaction statistics operations."""

    async def test_upsert_reaction_stats_insert(self, message_database: Database):
        """Test inserting new reaction stats."""
        await message_database.upsert_reaction_stats(
            giver_id=12345, receiver_id=67890, year=2024, month=1
        )

        async with message_database.conn.execute(
            "SELECT reaction_count FROM user_reactions_monthly WHERE giver_id = ? AND receiver_id = ?",
            (12345, 67890),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1

    async def test_upsert_reaction_stats_increment(self, message_database: Database):
        """Test that upsert increments reaction count."""
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)

        async with message_database.conn.execute(
            "SELECT reaction_count FROM user_reactions_monthly WHERE giver_id = ? AND receiver_id = ?",
            (12345, 67890),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 3

    async def test_batch_upsert_reaction_stats(self, message_database: Database):
        """Test batch inserting reaction stats."""
        stats = [
            (12345, 67890, 2024, 1),
            (12345, 67890, 2024, 1),
            (67890, 12345, 2024, 1),
        ]
        await message_database.batch_upsert_reaction_stats(stats)

        async with message_database.conn.execute(
            "SELECT giver_id, receiver_id, reaction_count FROM user_reactions_monthly ORDER BY giver_id, receiver_id"
        ) as cursor:
            rows = await cursor.fetchall()

        assert len(rows) == 2
        assert rows[0] == (12345, 67890, 2)  # Two reactions from 12345 to 67890
        assert rows[1] == (67890, 12345, 1)

    async def test_delete_all_user_stats(self, message_database: Database):
        """Test deleting all user stats and reactions."""
        await message_database.upsert_user_stats(12345, 2024, 1, 5)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)

        await message_database.delete_all_user_stats()

        async with message_database.conn.execute(
            "SELECT COUNT(*) FROM user_stats_monthly"
        ) as cursor:
            stats_count = (await cursor.fetchone())[0]

        async with message_database.conn.execute(
            "SELECT COUNT(*) FROM user_reactions_monthly"
        ) as cursor:
            reactions_count = (await cursor.fetchone())[0]

        assert stats_count == 0
        assert reactions_count == 0


class TestComplexQueries:
    """Tests for complex query operations and edge cases."""

    async def test_fetch_inflation_data_monthly(self, message_database: Database):
        """Test fetching monthly inflation data."""
        # Set up data for two months (reactions stored in user_stats_monthly)
        await message_database.upsert_user_stats(12345, 2024, 1, 10)  # 10 reactions
        await message_database.upsert_user_stats(12345, 2024, 2, 15)  # 15 reactions

        result = await message_database.fetch_inflation_data(monthly=True, limit=12)

        assert len(result) == 2
        # Results ordered by year DESC, month DESC
        assert result[0][0] == 2024  # year
        assert result[0][1] == 2  # month
        assert result[0][2] == 15  # total_reactions for month 2 (from user_stats_monthly)
        assert result[1][1] == 1  # month 1
        assert result[1][2] == 10  # total_reactions for month 1

    async def test_fetch_inflation_data_yearly(self, message_database: Database):
        """Test fetching yearly inflation data."""
        await message_database.upsert_user_stats(12345, 2023, 6, 10)
        await message_database.upsert_user_stats(12345, 2024, 6, 15)

        result = await message_database.fetch_inflation_data(monthly=False)

        assert len(result) == 2
        # Results ordered by year ASC for yearly
        assert result[0][0] == 2023
        assert result[1][0] == 2024

    async def test_fetch_inflation_data_empty(self, message_database: Database):
        """Test fetching inflation data from empty database."""
        result = await message_database.fetch_inflation_data(monthly=True, limit=12)
        assert result == []

    async def test_fetch_inflation_uses_user_stats_monthly_reactions(
        self, message_database: Database
    ):
        """Test that inflation query uses user_stats_monthly.total_reactions.

        This is important for retro-scraped data where user_reactions_monthly
        (who->whom tracking) may not be populated, but user_stats_monthly
        has the total reaction counts.
        """
        # Only populate user_stats_monthly (simulating retro-scraped data)
        # Do NOT populate user_reactions_monthly
        await message_database.upsert_user_stats(12345, 2023, 6, 100)  # 100 reactions
        await message_database.upsert_user_stats(12345, 2024, 6, 150)  # 150 reactions

        # Monthly query should return reactions from user_stats_monthly
        monthly_result = await message_database.fetch_inflation_data(monthly=True, limit=12)
        assert len(monthly_result) == 2
        # Result format: (year, month, total_reactions, total_messages)
        assert monthly_result[0][2] == 150  # 2024-06 reactions
        assert monthly_result[1][2] == 100  # 2023-06 reactions

        # Yearly query should also return reactions from user_stats_monthly
        yearly_result = await message_database.fetch_inflation_data(monthly=False)
        assert len(yearly_result) == 2
        # Result format: (year, total_reactions, total_messages)
        assert yearly_result[0][1] == 100  # 2023 reactions
        assert yearly_result[1][1] == 150  # 2024 reactions

    async def test_fetch_gdp_data(self, message_database: Database):
        """Test fetching GDP (total messages) data."""
        await message_database.upsert_user_stats(12345, 2024, 1, 5)
        await message_database.upsert_user_stats(12345, 2024, 1, 5)
        await message_database.upsert_user_stats(67890, 2024, 1, 3)
        await message_database.upsert_user_stats(12345, 2024, 2, 10)

        result = await message_database.fetch_gdp_data(limit=24)

        assert len(result) == 2
        # Ordered by year DESC, month DESC
        assert result[0] == (2024, 2, 1)  # 1 message in Feb
        assert result[1] == (2024, 1, 3)  # 3 messages in Jan

    async def test_fetch_gdp_data_respects_limit(self, message_database: Database):
        """Test that fetch_gdp_data respects the limit parameter."""
        for month in range(1, 13):
            await message_database.upsert_user_stats(12345, 2024, month, 5)

        result = await message_database.fetch_gdp_data(limit=3)

        assert len(result) == 3

    async def test_fetch_hdi_data(self, message_database: Database):
        """Test fetching HDI (quality ratio) data."""
        # Add quality messages to messages table
        msg = Message(
            id=1,
            content="Quality message",
            timestamp=datetime.datetime(2024, 1, 15),
            reaction_count=10,
            author_id=12345,
        )
        await message_database.add_messages([msg])

        # Add user stats (total messages)
        await message_database.upsert_user_stats(12345, 2024, 1, 5)
        await message_database.upsert_user_stats(12345, 2024, 1, 5)

        result = await message_database.fetch_hdi_data(limit=24)

        assert len(result) == 1
        assert result[0][0] == "2024"  # year (string from strftime)
        assert result[0][1] == "01"  # month (string from strftime)
        assert result[0][2] == 1  # quality_count
        assert result[0][3] == 2  # total_count
        assert result[0][4] == 0.5  # hdi_ratio

    async def test_fetch_hdi_data_division_by_zero(self, message_database: Database):
        """Test that HDI handles division by zero with NULLIF."""
        # Add a quality message but no user stats
        msg = Message(
            id=1,
            content="Quality message",
            timestamp=datetime.datetime(2024, 1, 15),
            reaction_count=10,
            author_id=12345,
        )
        await message_database.add_messages([msg])

        # Query should not crash due to division by zero
        result = await message_database.fetch_hdi_data(limit=24)

        # The JOIN requires matching month in both tables, so with no user_stats
        # for that month, the result should be empty (not a division error)
        assert result == []

    async def test_fetch_trade_data(self, message_database: Database):
        """Test fetching trade balance data for a user."""
        # User 12345 gives reactions
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 11111, 2024, 1)

        # User 12345 receives reactions
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(11111, 12345, 2024, 1)

        result = await message_database.fetch_trade_data(12345, 2024, 1, limit=5)

        assert result["total_given"] == 3
        assert result["total_received"] == 5
        assert result["trade_balance"] == 2  # received - given

        # Check exports (who user gave reactions to)
        assert len(result["exports"]) == 2
        assert result["exports"][0][0] == 67890  # top export partner
        assert result["exports"][0][1] == 2  # gave 2 reactions

    async def test_fetch_trade_data_no_activity(self, message_database: Database):
        """Test fetching trade data for user with no reactions."""
        result = await message_database.fetch_trade_data(99999, 2024, 1, limit=5)

        assert result["total_given"] == 0
        assert result["total_received"] == 0
        assert result["trade_balance"] == 0
        assert result["exports"] == []
        assert result["imports"] == []

    async def test_fetch_reaction_network(self, message_database: Database):
        """Test fetching reaction network for most-liked calculation."""
        # Set up user mappings
        await message_database.upsert_user_mapping(12345, "Alice")
        await message_database.upsert_user_mapping(67890, "Bob")

        # Set up user stats for the month
        await message_database.upsert_user_stats(12345, 2024, 1, 10)
        await message_database.upsert_user_stats(67890, 2024, 1, 5)

        # Set up reactions
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)

        result = await message_database.fetch_reaction_network(2024, 1)

        assert len(result) == 1
        assert result[0][0] == "Alice"  # giver_username
        assert result[0][1] == "Bob"  # receiver_username
        assert result[0][2] == 2  # reaction_count

    async def test_fetch_reaction_network_empty_month(self, message_database: Database):
        """Test fetching reaction network for empty month returns empty list."""
        result = await message_database.fetch_reaction_network(2024, 12)
        assert result == []
