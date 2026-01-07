"""
Tests for the Database class.
"""

import datetime

from strofkabot.discord_db import Attachment, Database, Message


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

    async def test_get_username_by_id(self, message_database: Database):
        """Test getting username by author_id."""
        # Test with non-existent user
        result = await message_database.get_username_by_id(99999)
        assert result is None

        # Add a user and retrieve
        await message_database.upsert_user_mapping(12345, "TestUser")
        result = await message_database.get_username_by_id(12345)
        assert result == "TestUser"

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

    async def test_fetch_gdp_data_no_limit(self, message_database: Database):
        """Test that fetch_gdp_data with limit=None returns all data."""
        # Insert data for 30 months (more than default limit of 24)
        for i in range(30):
            month = (i % 12) + 1
            year = 2022 + (i // 12)
            await message_database.upsert_user_stats(12345, year, month, 5)

        result = await message_database.fetch_gdp_data(limit=None)

        # Should return all 30 months, not limited to 24
        assert len(result) == 30

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

    async def test_fetch_trade_data_for_month(self, message_database: Database):
        """Test fetching trade data for a single specific month."""
        # User 12345 gives reactions in January 2024
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 11111, 2024, 1)

        # User 12345 receives reactions in January 2024
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 1)
        await message_database.upsert_reaction_stats(11111, 12345, 2024, 1)

        # Add data for February 2024 (should NOT be included)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 2)
        await message_database.upsert_reaction_stats(67890, 12345, 2024, 2)

        result = await message_database.fetch_trade_data_for_month(12345, 2024, 1, limit=5)

        # Should only include January data
        assert result["total_given"] == 3
        assert result["total_received"] == 3
        assert result["trade_balance"] == 0  # received - given

        # Check exports (who user gave reactions to)
        assert len(result["exports"]) == 2
        assert result["exports"][0][0] == 67890  # top export partner
        assert result["exports"][0][1] == 2  # gave 2 reactions

    async def test_fetch_trade_data_for_month_no_activity(self, message_database: Database):
        """Test fetching single-month trade data for user with no reactions."""
        result = await message_database.fetch_trade_data_for_month(99999, 2024, 1, limit=5)

        assert result["total_given"] == 0
        assert result["total_received"] == 0
        assert result["trade_balance"] == 0
        assert result["exports"] == []
        assert result["imports"] == []

    async def test_fetch_trade_data_for_month_excludes_other_months(
        self, message_database: Database
    ):
        """Test that fetch_trade_data_for_month only queries the specified month."""
        # Add data for multiple months
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 1)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 2)
        await message_database.upsert_reaction_stats(12345, 67890, 2024, 3)

        # Query only February
        result = await message_database.fetch_trade_data_for_month(12345, 2024, 2, limit=5)

        assert result["total_given"] == 1  # Only February data
        assert len(result["exports"]) == 1
        assert result["exports"][0][1] == 1

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


class TestAttachmentOperations:
    """Tests for attachment CRUD operations."""

    async def test_add_single_attachment(self, message_database: Database):
        """Test adding a single attachment."""
        attachment = Attachment(
            id=100,
            message_id=1000,
            message_content="Test message with image",
            author_id=12345,
            timestamp=datetime.datetime(2024, 5, 1, 12, 0, 0),
            reaction_count=5,
            original_filename="test.jpg",
            local_path="1000/100.jpg",
        )
        await message_database.add_attachments([attachment])

        async with message_database.conn.execute(
            "SELECT * FROM attachments WHERE id = ?", (100,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[0] == 100  # id
        assert row[1] == 1000  # message_id
        assert row[2] == "Test message with image"  # message_content
        assert row[3] == 12345  # author_id
        assert row[5] == 5  # reaction_count
        assert row[6] == "test.jpg"  # original_filename
        assert row[7] == "1000/100.jpg"  # local_path

    async def test_add_multiple_attachments(self, message_database: Database):
        """Test batch adding attachments."""
        attachments = [
            Attachment(
                id=i,
                message_id=1000 + i,
                message_content=f"Message {i}",
                author_id=12345,
                timestamp=datetime.datetime.now(),
                reaction_count=i,
                original_filename=f"file{i}.jpg",
                local_path=f"{1000+i}/{i}.jpg",
            )
            for i in range(1, 6)
        ]
        await message_database.add_attachments(attachments)

        async with message_database.conn.execute("SELECT COUNT(*) FROM attachments") as cursor:
            count = (await cursor.fetchone())[0]

        assert count == 5

    async def test_add_attachment_with_null_content(self, message_database: Database):
        """Test adding an attachment with no message content."""
        attachment = Attachment(
            id=200,
            message_id=2000,
            message_content=None,
            author_id=12345,
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            original_filename="image.png",
            local_path="2000/200.png",
        )
        await message_database.add_attachments([attachment])

        async with message_database.conn.execute(
            "SELECT message_content FROM attachments WHERE id = ?", (200,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] is None

    async def test_get_random_attachment_returns_attachment_object(
        self, message_database: Database
    ):
        """Test that get_random_attachment returns an Attachment dataclass."""
        attachment = Attachment(
            id=100,
            message_id=1000,
            message_content="Test",
            author_id=12345,
            timestamp=datetime.datetime(2024, 5, 1, 12, 0, 0),
            reaction_count=5,
            original_filename="test.jpg",
            local_path="1000/100.jpg",
        )
        await message_database.add_attachments([attachment])

        result = await message_database.get_random_attachment()

        assert result is not None
        assert isinstance(result, Attachment)
        assert result.id == 100

    async def test_get_random_attachment_empty_database(self, message_database: Database):
        """Test get_random_attachment on empty database returns None."""
        result = await message_database.get_random_attachment()
        assert result is None

    async def test_get_random_attachment_timestamp_is_datetime(self, message_database: Database):
        """Verify that timestamp is properly converted to datetime object."""
        attachment = Attachment(
            id=100,
            message_id=1000,
            message_content="Test",
            author_id=12345,
            timestamp=datetime.datetime(2024, 5, 1, 12, 0, 0),
            reaction_count=5,
            original_filename="test.jpg",
            local_path="1000/100.jpg",
        )
        await message_database.add_attachments([attachment])

        result = await message_database.get_random_attachment()
        assert isinstance(result.timestamp, datetime.datetime)

    async def test_get_attachment_count(self, message_database: Database):
        """Test getting attachment count."""
        assert await message_database.get_attachment_count() == 0

        attachments = [
            Attachment(
                id=i,
                message_id=1000 + i,
                message_content=f"Message {i}",
                author_id=12345,
                timestamp=datetime.datetime.now(),
                reaction_count=i,
                original_filename=f"file{i}.jpg",
                local_path=f"{1000+i}/{i}.jpg",
            )
            for i in range(1, 4)
        ]
        await message_database.add_attachments(attachments)

        assert await message_database.get_attachment_count() == 3

    async def test_get_message_count(self, message_database: Database):
        """Test getting message count."""
        assert await message_database.get_message_count() == 0

        messages = [
            Message(
                id=i,
                content=f"Message {i}",
                timestamp=datetime.datetime.now(),
                reaction_count=i,
                author_id=12345,
            )
            for i in range(1, 6)
        ]
        await message_database.add_messages(messages)

        assert await message_database.get_message_count() == 5

    async def test_attachment_exists_true(self, message_database: Database):
        """Test that attachment_exists returns True for existing attachment."""
        attachment = Attachment(
            id=100,
            message_id=1000,
            message_content="Test",
            author_id=12345,
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            original_filename="test.jpg",
            local_path="1000/100.jpg",
        )
        await message_database.add_attachments([attachment])

        assert await message_database.attachment_exists(100) is True

    async def test_attachment_exists_false(self, message_database: Database):
        """Test that attachment_exists returns False for non-existing attachment."""
        assert await message_database.attachment_exists(99999) is False

    async def test_upsert_attachment_updates_existing(self, message_database: Database):
        """Test that adding an attachment with existing ID updates it (INSERT OR REPLACE)."""
        attachment1 = Attachment(
            id=100,
            message_id=1000,
            message_content="Original",
            author_id=12345,
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            original_filename="original.jpg",
            local_path="1000/100.jpg",
        )
        await message_database.add_attachments([attachment1])

        attachment2 = Attachment(
            id=100,
            message_id=1000,
            message_content="Updated",
            author_id=12345,
            timestamp=datetime.datetime.now(),
            reaction_count=10,
            original_filename="updated.jpg",
            local_path="1000/100_updated.jpg",
        )
        await message_database.add_attachments([attachment2])

        async with message_database.conn.execute(
            "SELECT message_content, reaction_count, local_path FROM attachments WHERE id = ?",
            (100,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Updated"
        assert row[1] == 10
        assert row[2] == "1000/100_updated.jpg"


class TestOnThisDayOperations:
    """Tests for on-this-day query operations."""

    async def test_get_on_this_day_years_returns_distinct_years(self, message_database: Database):
        """Test that get_on_this_day_years returns distinct years."""
        messages = [
            Message(
                id=1,
                content="2022 msg",
                timestamp=datetime.datetime(2022, 1, 7, 10, 0, 0),
                reaction_count=5,
                author_id=123,
            ),
            Message(
                id=2,
                content="2023 msg",
                timestamp=datetime.datetime(2023, 1, 7, 14, 0, 0),
                reaction_count=8,
                author_id=123,
            ),
            Message(
                id=3,
                content="Different day",
                timestamp=datetime.datetime(2023, 1, 8, 10, 0, 0),
                reaction_count=10,
                author_id=123,
            ),
        ]
        await message_database.add_messages(messages)

        years = await message_database.get_on_this_day_years(1, 7)

        assert years == [2022, 2023]

    async def test_get_on_this_day_years_includes_attachments(self, message_database: Database):
        """Test that attachments are included in year results."""
        attachment = Attachment(
            id=1,
            message_id=100,
            message_content="Test",
            author_id=123,
            timestamp=datetime.datetime(2021, 1, 7, 12, 0, 0),
            reaction_count=10,
            original_filename="test.jpg",
            local_path="100/1.jpg",
        )
        await message_database.add_attachments([attachment])

        years = await message_database.get_on_this_day_years(1, 7)

        assert 2021 in years

    async def test_get_on_this_day_years_combines_messages_and_attachments(
        self, message_database: Database
    ):
        """Test that both messages and attachments contribute to year list."""
        msg = Message(
            id=1,
            content="2022 message",
            timestamp=datetime.datetime(2022, 3, 15, 10, 0, 0),
            reaction_count=5,
            author_id=123,
        )
        att = Attachment(
            id=1,
            message_id=100,
            message_content="2023 attachment",
            author_id=123,
            timestamp=datetime.datetime(2023, 3, 15, 12, 0, 0),
            reaction_count=10,
            original_filename="test.jpg",
            local_path="100/1.jpg",
        )
        await message_database.add_messages([msg])
        await message_database.add_attachments([att])

        years = await message_database.get_on_this_day_years(3, 15)

        assert years == [2022, 2023]

    async def test_get_on_this_day_years_empty(self, message_database: Database):
        """Test that empty database returns empty list."""
        years = await message_database.get_on_this_day_years(12, 25)
        assert years == []

    async def test_get_top_message_returns_highest_reacted(self, message_database: Database):
        """Test that get_top_message returns highest reaction count."""
        messages = [
            Message(
                id=1,
                content="Low reactions",
                timestamp=datetime.datetime(2023, 1, 7, 10, 0, 0),
                reaction_count=5,
                author_id=123,
            ),
            Message(
                id=2,
                content="High reactions",
                timestamp=datetime.datetime(2023, 1, 7, 14, 0, 0),
                reaction_count=20,
                author_id=123,
            ),
        ]
        await message_database.add_messages(messages)

        message, attachment = await message_database.get_top_message_on_this_day(2023, 1, 7)

        assert message is not None
        assert attachment is None
        assert message.id == 2
        assert message.reaction_count == 20

    async def test_get_top_message_returns_attachment_when_higher(self, message_database: Database):
        """Test that attachment is returned when it has more reactions."""
        msg = Message(
            id=1,
            content="Message",
            timestamp=datetime.datetime(2023, 1, 7, 10, 0, 0),
            reaction_count=5,
            author_id=123,
        )
        att = Attachment(
            id=1,
            message_id=100,
            message_content="Attachment",
            author_id=123,
            timestamp=datetime.datetime(2023, 1, 7, 12, 0, 0),
            reaction_count=15,
            original_filename="img.jpg",
            local_path="100/1.jpg",
        )
        await message_database.add_messages([msg])
        await message_database.add_attachments([att])

        message, attachment = await message_database.get_top_message_on_this_day(2023, 1, 7)

        assert message is None
        assert attachment is not None
        assert attachment.reaction_count == 15

    async def test_get_top_message_prefers_message_on_tie(self, message_database: Database):
        """Test that message is preferred when reaction counts are equal."""
        msg = Message(
            id=1,
            content="Message",
            timestamp=datetime.datetime(2023, 1, 7, 10, 0, 0),
            reaction_count=10,
            author_id=123,
        )
        att = Attachment(
            id=1,
            message_id=100,
            message_content="Attachment",
            author_id=123,
            timestamp=datetime.datetime(2023, 1, 7, 12, 0, 0),
            reaction_count=10,
            original_filename="img.jpg",
            local_path="100/1.jpg",
        )
        await message_database.add_messages([msg])
        await message_database.add_attachments([att])

        message, attachment = await message_database.get_top_message_on_this_day(2023, 1, 7)

        assert message is not None
        assert attachment is None

    async def test_get_top_message_no_content(self, message_database: Database):
        """Test that no content returns (None, None)."""
        message, attachment = await message_database.get_top_message_on_this_day(2023, 12, 25)

        assert message is None
        assert attachment is None

    async def test_get_top_message_only_attachment_exists(self, message_database: Database):
        """Test that attachment is returned when no messages exist."""
        att = Attachment(
            id=1,
            message_id=100,
            message_content="Only attachment",
            author_id=123,
            timestamp=datetime.datetime(2023, 5, 20, 12, 0, 0),
            reaction_count=8,
            original_filename="img.jpg",
            local_path="100/1.jpg",
        )
        await message_database.add_attachments([att])

        message, attachment = await message_database.get_top_message_on_this_day(2023, 5, 20)

        assert message is None
        assert attachment is not None
        assert attachment.reaction_count == 8

    async def test_get_top_message_only_message_exists(self, message_database: Database):
        """Test that message is returned when no attachments exist."""
        msg = Message(
            id=1,
            content="Only message",
            timestamp=datetime.datetime(2023, 5, 20, 10, 0, 0),
            reaction_count=12,
            author_id=123,
        )
        await message_database.add_messages([msg])

        message, attachment = await message_database.get_top_message_on_this_day(2023, 5, 20)

        assert message is not None
        assert attachment is None
        assert message.reaction_count == 12


class TestPredictionOperations:
    """Tests for prediction CRUD operations."""

    async def test_add_prediction_returns_id(self, message_database: Database):
        """Test that add_prediction returns the new prediction ID."""
        pred_id = await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 6, 1),
            prediction_text="Test prediction",
        )
        assert pred_id == 1

    async def test_add_prediction_stores_data(self, message_database: Database):
        """Test that add_prediction correctly stores all fields."""
        pred_id = await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 6, 1),
            prediction_text="Test prediction text",
        )

        async with message_database.conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row[1] == 12345  # author_id
        assert row[2] == "TestUser"  # author_name
        assert row[3] == 67890  # channel_id
        assert row[4] == "2025-06-01"  # target_date
        assert row[5] == "Test prediction text"  # prediction_text
        assert row[7] is False or row[7] == 0  # posted (False)
        assert row[8] is None  # posted_at

    async def test_get_due_predictions_returns_unposted(self, message_database: Database):
        """Test that get_due_predictions returns unposted predictions."""
        await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 1, 1),
            prediction_text="Due prediction",
        )

        predictions = await message_database.get_due_predictions(datetime.date(2025, 1, 1))
        assert len(predictions) == 1
        assert predictions[0].prediction_text == "Due prediction"

    async def test_get_due_predictions_excludes_posted(self, message_database: Database):
        """Test that posted predictions are excluded."""
        pred_id = await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 1, 1),
            prediction_text="Posted prediction",
        )
        await message_database.mark_prediction_posted(pred_id)

        predictions = await message_database.get_due_predictions(datetime.date(2025, 1, 1))
        assert len(predictions) == 0

    async def test_get_due_predictions_includes_overdue(self, message_database: Database):
        """Test that overdue predictions are included."""
        await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 1, 1),
            prediction_text="Overdue prediction",
        )

        # Query for a later date
        predictions = await message_database.get_due_predictions(datetime.date(2025, 1, 15))
        assert len(predictions) == 1

    async def test_get_due_predictions_excludes_future(self, message_database: Database):
        """Test that future predictions are not returned."""
        await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 6, 1),
            prediction_text="Future prediction",
        )

        # Query for an earlier date
        predictions = await message_database.get_due_predictions(datetime.date(2025, 1, 1))
        assert len(predictions) == 0

    async def test_mark_prediction_posted(self, message_database: Database):
        """Test that mark_prediction_posted updates the prediction."""
        pred_id = await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 1, 1),
            prediction_text="Test prediction",
        )

        await message_database.mark_prediction_posted(pred_id)

        async with message_database.conn.execute(
            "SELECT posted, posted_at FROM predictions WHERE id = ?", (pred_id,)
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1  # posted = True
        assert row[1] is not None  # posted_at has a timestamp

    async def test_prediction_dataclass_fields(self, message_database: Database):
        """Test that returned Prediction has all expected fields."""
        from strofkabot.discord_db import Prediction

        await message_database.add_prediction(
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date(2025, 1, 1),
            prediction_text="Test prediction",
        )

        predictions = await message_database.get_due_predictions(datetime.date(2025, 1, 1))
        pred = predictions[0]

        assert isinstance(pred, Prediction)
        assert pred.id == 1
        assert pred.author_id == 12345
        assert pred.author_name == "TestUser"
        assert pred.channel_id == 67890
        assert pred.target_date == datetime.date(2025, 1, 1)
        assert pred.prediction_text == "Test prediction"
        assert pred.posted is False
        assert pred.posted_at is None
        assert isinstance(pred.created_at, datetime.datetime)
