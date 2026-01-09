"""
Integration tests for database flows.

These tests verify that data flows correctly through the system
without mocking the database layer, catching issues that unit tests
with mocks might miss.
"""

import datetime
from unittest.mock import patch

import discord.ext.test as dpytest
import pytest

from strofkabot.discord_db import Message


class TestMessageDatabaseFlow:
    """Integration tests for message storage and retrieval."""

    async def test_add_and_retrieve_message(self, real_database, message_factory):
        """Test that messages can be added and retrieved correctly."""
        # Create a message via factory
        msg = await message_factory(
            content="Integration test message",
            author_id=99999,
            reaction_count=7,
        )

        # Retrieve via get_random_message (only message in DB)
        retrieved = await real_database.get_random_message()

        assert retrieved is not None
        assert retrieved.content == "Integration test message"
        assert retrieved.author_id == 99999
        assert retrieved.reaction_count == 7

    async def test_message_count_accuracy(self, real_database, message_factory):
        """Test that message count is accurate after multiple insertions."""
        # Add multiple messages
        for i in range(5):
            await message_factory(content=f"Message {i}")

        count = await real_database.get_message_count()
        assert count == 5

    async def test_message_with_reply_roundtrip(self, real_database, message_factory):
        """Test that reply information is preserved through storage."""
        # Create original message
        original = await message_factory(
            message_id=1,
            content="Original message",
            author_id=111,
        )

        # Create reply
        reply = await message_factory(
            message_id=2,
            content="This is a reply",
            author_id=222,
            reply_to_id=1,
            reply_to_author="OriginalUser",
            reply_to_content="Original message",
        )

        # Verify reply info is stored
        # Query the database directly to check
        async with real_database.conn.execute(
            "SELECT reply_to_id, reply_to_author, reply_to_content FROM messages WHERE id = ?",
            (2,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1
        assert row[1] == "OriginalUser"
        assert row[2] == "Original message"


class TestUserStatsFlow:
    """Integration tests for user statistics operations."""

    async def test_stats_aggregation_accuracy(self, real_user_stats, real_database):
        """Test that user stats aggregate correctly over multiple updates."""
        author_id = 12345

        # Add multiple stats entries for the same user/month
        stats_data = [
            (author_id, 5, datetime.datetime(2024, 1, 10)),
            (author_id, 3, datetime.datetime(2024, 1, 15)),
            (author_id, 7, datetime.datetime(2024, 1, 20)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Query aggregated stats
        async with real_database.conn.execute(
            """
            SELECT total_messages, total_reactions
            FROM user_stats_monthly
            WHERE author_id = ? AND year = 2024 AND month = 1
            """,
            (author_id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 3  # 3 messages
        assert row[1] == 15  # 5 + 3 + 7 = 15 reactions

    async def test_reaction_network_data_integrity(
        self, real_user_stats, real_database, reaction_factory
    ):
        """Test that reaction network data maintains integrity."""
        # Create reaction relationships
        await reaction_factory(giver_id=111, receiver_id=222, count=3)
        await reaction_factory(giver_id=111, receiver_id=333, count=2)
        await reaction_factory(giver_id=222, receiver_id=111, count=1)

        # Query reaction data
        async with real_database.conn.execute(
            """
            SELECT giver_id, receiver_id, reaction_count
            FROM user_reactions_monthly
            WHERE year = 2024 AND month = 1
            ORDER BY giver_id, receiver_id
            """,
        ) as cursor:
            rows = await cursor.fetchall()

        # Verify relationships
        assert len(rows) == 3
        # 111 -> 222: 3 reactions
        assert rows[0] == (111, 222, 3)
        # 111 -> 333: 2 reactions
        assert rows[1] == (111, 333, 2)
        # 222 -> 111: 1 reaction
        assert rows[2] == (222, 111, 1)

    async def test_user_mapping_persistence(self, real_user_stats, real_database):
        """Test that user mappings are correctly stored and retrieved."""
        await real_user_stats.update_user_mapping(12345, "Alice")
        await real_user_stats.update_user_mapping(67890, "Bob")

        async with real_database.conn.execute(
            "SELECT author_id, username FROM user_mapping ORDER BY author_id"
        ) as cursor:
            rows = await cursor.fetchall()

        assert len(rows) == 2
        assert rows[0] == (12345, "Alice")
        assert rows[1] == (67890, "Bob")


class TestCommandToDatabase:
    """Integration tests for command execution with real database."""

    @pytest.mark.asyncio
    async def test_llumi_retrieves_real_message(self, bot_with_real_db, message_factory):
        """Test that !llumi retrieves an actual message from the database."""
        bot, db, user_stats, cog = bot_with_real_db

        # Add a message to the database
        await message_factory(
            content="This message was stored in a real database",
            reaction_count=10,
        )

        # Execute command
        await dpytest.message("!llumi")

        # Verify the response contains our message
        assert dpytest.verify().message().content("This message was stored in a real database")

    @pytest.mark.asyncio
    async def test_llumi_with_multiple_messages(self, bot_with_real_db, message_factory):
        """Test that !llumi returns one of multiple stored messages."""
        bot, db, user_stats, cog = bot_with_real_db

        # Add multiple messages
        contents = [
            "First integration test message",
            "Second integration test message",
            "Third integration test message",
        ]
        for content in contents:
            await message_factory(content=content)

        # Mock random to avoid Easter egg (1/50 chance) and image selection
        with patch("strofkabot.cogs.entertainment.random") as mock_random:
            mock_random.randint.return_value = 2  # Not 1, so no Easter egg
            mock_random.random.return_value = 0.6  # > 0.5, so message not image

            # Execute command
            await dpytest.message("!llumi")

        # Get the response message
        response = dpytest.get_message()
        assert response.content in contents

    @pytest.mark.asyncio
    async def test_llumi_empty_database(self, bot_with_real_db):
        """Test that !llumi handles empty database correctly."""
        # Don't add any messages - database is empty

        await dpytest.message("!llumi")
        assert dpytest.verify().message().content("No messages available at the moment.")


class TestDataConsistency:
    """Tests for data consistency across operations."""

    async def test_concurrent_message_inserts(self, real_database):
        """Test that concurrent message insertions don't cause data loss."""
        import asyncio

        messages = [
            Message(
                id=i,
                content=f"Concurrent message {i}",
                timestamp=datetime.datetime.now(),
                reaction_count=i,
                author_id=1000 + i,
            )
            for i in range(100)
        ]

        # Insert messages concurrently
        await asyncio.gather(*[real_database.add_messages([msg]) for msg in messages])

        # Verify all messages were inserted
        count = await real_database.get_message_count()
        assert count == 100

    async def test_upsert_preserves_latest_data(self, real_database):
        """Test that upserting a message preserves the latest data."""
        msg1 = Message(
            id=999,
            content="Original content",
            timestamp=datetime.datetime(2024, 1, 1),
            reaction_count=5,
            author_id=12345,
        )
        await real_database.add_messages([msg1])

        # Upsert with updated data
        msg2 = Message(
            id=999,
            content="Updated content",
            timestamp=datetime.datetime(2024, 1, 2),
            reaction_count=10,
            author_id=12345,
        )
        await real_database.add_messages([msg2])

        # Verify the update
        async with real_database.conn.execute(
            "SELECT content, reaction_count FROM messages WHERE id = ?",
            (999,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "Updated content"
        assert row[1] == 10

        # Should still be just one message
        count = await real_database.get_message_count()
        assert count == 1


class TestTimestampMetadata:
    """Integration tests for timestamp metadata operations."""

    async def test_channel_timestamp_roundtrip(self, real_database):
        """Test that channel scan timestamps are correctly stored and retrieved."""
        channel_id = 123456789
        timestamp = datetime.datetime(2024, 6, 15, 14, 30, 0)

        await real_database.update_last_scanned_timestamp(channel_id, timestamp)
        retrieved = await real_database.get_last_scanned_timestamp(channel_id)

        assert retrieved == timestamp

    async def test_multiple_channel_timestamps(self, real_database):
        """Test managing timestamps for multiple channels."""
        channels = {
            111: datetime.datetime(2024, 1, 1, 10, 0, 0),
            222: datetime.datetime(2024, 2, 15, 12, 30, 0),
            333: datetime.datetime(2024, 3, 20, 18, 45, 0),
        }

        for channel_id, timestamp in channels.items():
            await real_database.update_last_scanned_timestamp(channel_id, timestamp)

        # Verify each channel has correct timestamp
        for channel_id, expected in channels.items():
            retrieved = await real_database.get_last_scanned_timestamp(channel_id)
            assert retrieved == expected
