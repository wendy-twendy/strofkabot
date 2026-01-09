"""
Integration tests for background task data flows.

Tests the data processing logic used by background tasks
including message filtering, stats collection, and reaction processing.
"""

import datetime

import pytest

from strofkabot.message_filter import MessageFilter


class TestMessageFilteringIntegration:
    """Integration tests for message filtering in background tasks."""

    @pytest.fixture
    def message_filter(self):
        """Create a MessageFilter instance."""
        return MessageFilter()

    @pytest.mark.parametrize(
        "content,should_pass",
        [
            # Valid messages (15+ chars, no links/emojis/mentions)
            pytest.param(
                "This is a great message that should be stored",
                True,
                id="normal-message",
            ),
            pytest.param(
                "Short but 15+ chars message",
                True,
                id="exactly-minimum-length",
            ),
            pytest.param(
                "Кириллица текст который должен пройти фильтр",
                True,
                id="cyrillic-text",
            ),
            # Invalid messages
            pytest.param("Too short", False, id="too-short"),
            pytest.param("", False, id="empty"),
            pytest.param(
                "Check this out http://example.com",
                False,
                id="contains-http-link",
            ),
            pytest.param(
                "Look at https://reddit.com/post",
                False,
                id="contains-https-link",
            ),
            pytest.param(
                "<:custom_emoji:123456789> nice",
                False,
                id="contains-custom-emoji",
            ),
            pytest.param(
                "Hey <@123456789> check this",
                False,
                id="contains-user-mention",
            ),
        ],
    )
    def test_message_filter_quality_check(self, message_filter, content, should_pass):
        """Test that message filter correctly identifies quality messages."""
        result = message_filter.is_valid_message(content)
        assert result == should_pass


class TestStatsCollectionFlow:
    """Integration tests for stats collection from messages."""

    async def test_stats_accumulation_from_channel_scan(self, real_user_stats, real_database):
        """Test that stats accumulate correctly simulating a channel scan."""
        # Simulate processing messages from a channel scan
        # User 111 posts 5 messages with varying reaction counts
        messages = [
            (111, 3, datetime.datetime(2024, 1, 10, 10, 0)),
            (111, 5, datetime.datetime(2024, 1, 10, 11, 0)),
            (111, 2, datetime.datetime(2024, 1, 10, 12, 0)),
            (111, 8, datetime.datetime(2024, 1, 10, 13, 0)),
            (111, 2, datetime.datetime(2024, 1, 10, 14, 0)),
        ]

        # Simulate batch processing (as background task does)
        await real_user_stats.batch_update_stats(messages)

        # Verify accumulated stats
        stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)

        assert stats["total_msgs"] == 5
        assert stats["total_reacts"] == 20  # 3+5+2+8+2
        assert stats["avg_reacts"] == pytest.approx(4.0, rel=0.01)

    async def test_multi_user_stats_collection(self, real_user_stats):
        """Test collecting stats for multiple users in a single scan."""
        # Simulate multiple users posting in a channel
        messages = [
            # User 111
            (111, 5, datetime.datetime(2024, 1, 10, 10, 0)),
            (111, 3, datetime.datetime(2024, 1, 10, 14, 0)),
            # User 222
            (222, 10, datetime.datetime(2024, 1, 10, 11, 0)),
            # User 333
            (333, 2, datetime.datetime(2024, 1, 10, 12, 0)),
            (333, 4, datetime.datetime(2024, 1, 10, 15, 0)),
            (333, 6, datetime.datetime(2024, 1, 10, 16, 0)),
        ]

        await real_user_stats.batch_update_stats(messages)

        # Verify each user's stats
        user_111 = await real_user_stats.get_user_monthly_stats(111, 2024, 1)
        user_222 = await real_user_stats.get_user_monthly_stats(222, 2024, 1)
        user_333 = await real_user_stats.get_user_monthly_stats(333, 2024, 1)

        assert user_111["total_msgs"] == 2
        assert user_111["total_reacts"] == 8

        assert user_222["total_msgs"] == 1
        assert user_222["total_reacts"] == 10

        assert user_333["total_msgs"] == 3
        assert user_333["total_reacts"] == 12


class TestReactionCollectionFlow:
    """Integration tests for reaction data collection from messages."""

    async def test_reaction_relationship_collection(self, real_user_stats, real_database):
        """Test collecting reaction relationships simulating message processing."""
        # Set up user mappings and stats required for network query
        for user_id in [111, 222, 333]:
            await real_user_stats.update_user_mapping(user_id, f"User{user_id}")
            await real_user_stats.batch_update_stats([(user_id, 1, datetime.datetime(2024, 1, 10))])

        # Simulate collecting reactions from messages
        # Message by user 111 gets reactions from 222 and 333
        reactions = [
            (222, 111, datetime.datetime(2024, 1, 10)),  # 222 reacts to 111's msg
            (333, 111, datetime.datetime(2024, 1, 10)),  # 333 reacts to 111's msg
            (222, 111, datetime.datetime(2024, 1, 10)),  # 222 reacts again
        ]

        await real_user_stats.batch_update_reaction_stats(reactions)

        # Verify reaction network
        network = await real_user_stats.get_reaction_network_for_month(2024, 1)

        assert len(network) == 2  # Two unique giver->receiver pairs

        # 222 -> 111 should have 2 reactions
        reaction_222 = next((r for r in network if r["giver_username"] == "User222"), None)
        assert reaction_222 is not None
        assert reaction_222["reaction_count"] == 2

        # 333 -> 111 should have 1 reaction
        reaction_333 = next((r for r in network if r["giver_username"] == "User333"), None)
        assert reaction_333 is not None
        assert reaction_333["reaction_count"] == 1

    async def test_cross_channel_reaction_aggregation(self, real_user_stats):
        """Test that reactions from multiple channels aggregate correctly."""
        # Set up user mappings and stats
        for user_id in [111, 222]:
            await real_user_stats.update_user_mapping(user_id, f"User{user_id}")
            await real_user_stats.batch_update_stats([(user_id, 1, datetime.datetime(2024, 1, 10))])

        # Reactions collected from different channels
        # (all in same month, should aggregate)
        channel_1_reactions = [
            (222, 111, datetime.datetime(2024, 1, 5)),
            (222, 111, datetime.datetime(2024, 1, 6)),
        ]

        channel_2_reactions = [
            (222, 111, datetime.datetime(2024, 1, 15)),
            (222, 111, datetime.datetime(2024, 1, 16)),
            (222, 111, datetime.datetime(2024, 1, 17)),
        ]

        await real_user_stats.batch_update_reaction_stats(channel_1_reactions)
        await real_user_stats.batch_update_reaction_stats(channel_2_reactions)

        # Should have single entry with combined count
        network = await real_user_stats.get_reaction_network_for_month(2024, 1)

        assert len(network) == 1
        assert network[0]["reaction_count"] == 5  # 2 + 3


class TestTimestampTrackingFlow:
    """Integration tests for channel timestamp tracking."""

    async def test_channel_scan_progress_tracking(self, real_database):
        """Test tracking scan progress via timestamps."""
        channel_id = 123456789

        # Initial scan - no timestamp
        initial_ts = await real_database.get_last_scanned_timestamp(channel_id)
        assert initial_ts is None

        # After first scan
        first_scan = datetime.datetime(2024, 1, 15, 12, 0, 0)
        await real_database.update_last_scanned_timestamp(channel_id, first_scan)

        retrieved = await real_database.get_last_scanned_timestamp(channel_id)
        assert retrieved == first_scan

        # After second scan (next day)
        second_scan = datetime.datetime(2024, 1, 16, 12, 0, 0)
        await real_database.update_last_scanned_timestamp(channel_id, second_scan)

        retrieved = await real_database.get_last_scanned_timestamp(channel_id)
        assert retrieved == second_scan

    async def test_multi_channel_timestamp_tracking(self, real_database):
        """Test tracking timestamps for multiple channels independently."""
        channels = {
            111: datetime.datetime(2024, 1, 10, 10, 0, 0),
            222: datetime.datetime(2024, 1, 12, 14, 30, 0),
            333: datetime.datetime(2024, 1, 15, 8, 15, 0),
        }

        # Set timestamps
        for channel_id, timestamp in channels.items():
            await real_database.update_last_scanned_timestamp(channel_id, timestamp)

        # Verify independence
        for channel_id, expected in channels.items():
            retrieved = await real_database.get_last_scanned_timestamp(channel_id)
            assert retrieved == expected


class TestQualityMessageFlow:
    """Integration tests for quality message storage flow."""

    async def test_quality_message_storage(self, real_database, message_factory):
        """Test storing messages that meet quality threshold."""
        # Messages with 4+ reactions are "quality" messages
        quality_msg = await message_factory(
            content="This is a quality message with good engagement",
            reaction_count=10,
        )

        # Verify stored
        count = await real_database.get_message_count()
        assert count == 1

        # Verify retrievable
        retrieved = await real_database.get_random_message()
        assert retrieved is not None
        assert retrieved.content == "This is a quality message with good engagement"
        assert retrieved.reaction_count == 10

    async def test_quality_vs_stats_separation(
        self, real_database, real_user_stats, message_factory
    ):
        """Test that quality messages and stats are tracked separately."""
        # Add a quality message
        await message_factory(
            message_id=1001,
            content="Quality message",
            author_id=111,
            reaction_count=15,
        )

        # Add stats for same user (would come from all messages, not just quality)
        await real_user_stats.batch_update_stats(
            [
                (111, 15, datetime.datetime(2024, 1, 10)),  # The quality message
                (111, 2, datetime.datetime(2024, 1, 11)),  # A non-quality message
                (111, 3, datetime.datetime(2024, 1, 12)),  # Another non-quality
            ]
        )

        # Quality messages table: 1 message
        quality_count = await real_database.get_message_count()
        assert quality_count == 1

        # Stats: 3 messages
        stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)
        assert stats["total_msgs"] == 3
        assert stats["total_reacts"] == 20  # 15 + 2 + 3


class TestUsernameUpdateFlow:
    """Integration tests for username mapping updates."""

    async def test_username_update_flow(self, real_user_stats, real_database):
        """Test username update during background task."""
        # Initial mapping
        await real_user_stats.update_user_mapping(111, "OldName")

        # Verify initial
        async with real_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (111,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == "OldName"

        # User changes display name
        await real_user_stats.update_user_mapping(111, "NewDisplayName")

        # Verify update
        async with real_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (111,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == "NewDisplayName"

    async def test_bulk_username_update(self, real_user_stats, real_database):
        """Test updating many usernames (as would happen during full scan)."""
        users = [(100 + i, f"User{i}") for i in range(50)]

        for author_id, username in users:
            await real_user_stats.update_user_mapping(author_id, username)

        # Verify all stored
        async with real_database.conn.execute("SELECT COUNT(*) FROM user_mapping") as cursor:
            count = (await cursor.fetchone())[0]
            assert count == 50
