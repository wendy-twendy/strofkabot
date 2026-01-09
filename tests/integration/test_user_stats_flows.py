"""
Integration tests for user statistics data flows.

Tests the full flow from data insertion through statistics retrieval
for RPM (reactions per message), leaderboards, and monthly stats.
"""

import datetime

import pytest


class TestMonthlyStatsFlow:
    """Integration tests for monthly statistics data flows."""

    async def test_monthly_stats_aggregation(self, real_user_stats):
        """Test that monthly stats correctly aggregate user data."""
        # Add stats for multiple users in January 2024
        await real_user_stats.update_user_mapping(111, "Alice")
        await real_user_stats.update_user_mapping(222, "Bob")
        await real_user_stats.update_user_mapping(333, "Charlie")

        stats_data = [
            # Alice: 3 messages, 15 total reactions (avg 5.0)
            (111, 5, datetime.datetime(2024, 1, 1)),
            (111, 5, datetime.datetime(2024, 1, 10)),
            (111, 5, datetime.datetime(2024, 1, 20)),
            # Bob: 2 messages, 20 total reactions (avg 10.0)
            (222, 10, datetime.datetime(2024, 1, 5)),
            (222, 10, datetime.datetime(2024, 1, 15)),
            # Charlie: 1 message, 2 reactions (avg 2.0)
            (333, 2, datetime.datetime(2024, 1, 25)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch monthly stats
        monthly_stats = await real_user_stats.get_monthly_stats(2024, 1)

        # Convert to dict for easier assertion
        stats_by_user = {s["author_id"]: s for s in monthly_stats}

        # Verify Alice's stats
        assert stats_by_user[111]["username"] == "Alice"
        assert stats_by_user[111]["total_msgs"] == 3
        assert stats_by_user[111]["total_reacts"] == 15
        assert stats_by_user[111]["avg_reacts"] == pytest.approx(5.0, rel=0.01)

        # Verify Bob's stats
        assert stats_by_user[222]["username"] == "Bob"
        assert stats_by_user[222]["total_msgs"] == 2
        assert stats_by_user[222]["total_reacts"] == 20
        assert stats_by_user[222]["avg_reacts"] == pytest.approx(10.0, rel=0.01)

        # Verify Charlie's stats
        assert stats_by_user[333]["username"] == "Charlie"
        assert stats_by_user[333]["total_msgs"] == 1
        assert stats_by_user[333]["total_reacts"] == 2
        assert stats_by_user[333]["avg_reacts"] == pytest.approx(2.0, rel=0.01)

    async def test_user_specific_monthly_stats(self, real_user_stats):
        """Test fetching stats for a specific user."""
        await real_user_stats.update_user_mapping(111, "Alice")

        stats_data = [
            (111, 5, datetime.datetime(2024, 1, 1)),
            (111, 8, datetime.datetime(2024, 1, 15)),
            (111, 2, datetime.datetime(2024, 1, 20)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch specific user's stats
        user_stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)

        assert user_stats is not None
        assert user_stats["total_msgs"] == 3
        assert user_stats["total_reacts"] == 15  # 5 + 8 + 2
        assert user_stats["avg_reacts"] == pytest.approx(5.0, rel=0.01)

    async def test_user_monthly_stats_not_found(self, real_user_stats):
        """Test that missing user stats return None."""
        result = await real_user_stats.get_user_monthly_stats(99999, 2024, 1)
        assert result is None


class TestLeaderboardDataFlow:
    """Integration tests for leaderboard data generation."""

    async def test_leaderboard_ordering(self, real_user_stats):
        """Test that leaderboard is ordered by average reactions descending."""
        # Create users with different stats
        await real_user_stats.update_user_mapping(111, "Low")
        await real_user_stats.update_user_mapping(222, "Medium")
        await real_user_stats.update_user_mapping(333, "High")

        stats_data = [
            # Low: avg 2.0
            (111, 2, datetime.datetime(2024, 1, 1)),
            (111, 2, datetime.datetime(2024, 1, 10)),
            # Medium: avg 5.0
            (222, 5, datetime.datetime(2024, 1, 5)),
            (222, 5, datetime.datetime(2024, 1, 15)),
            # High: avg 10.0
            (333, 10, datetime.datetime(2024, 1, 8)),
            (333, 10, datetime.datetime(2024, 1, 18)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch monthly stats (should be ordered by avg_reacts)
        monthly_stats = await real_user_stats.get_monthly_stats(2024, 1)

        # Verify ordering (highest first)
        assert monthly_stats[0]["username"] == "High"
        assert monthly_stats[1]["username"] == "Medium"
        assert monthly_stats[2]["username"] == "Low"

    async def test_leaderboard_with_ties(self, real_user_stats):
        """Test leaderboard handling of tied average reactions."""
        await real_user_stats.update_user_mapping(111, "User1")
        await real_user_stats.update_user_mapping(222, "User2")

        # Both users have same average
        stats_data = [
            (111, 5, datetime.datetime(2024, 1, 1)),
            (222, 5, datetime.datetime(2024, 1, 5)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        monthly_stats = await real_user_stats.get_monthly_stats(2024, 1)

        # Both should be present with same average
        assert len(monthly_stats) == 2
        assert all(s["avg_reacts"] == pytest.approx(5.0, rel=0.01) for s in monthly_stats)


class TestUserMappingFlow:
    """Integration tests for user mapping (username cache) operations."""

    async def test_user_mapping_update(self, real_user_stats, real_database):
        """Test that user mapping updates correctly."""
        await real_user_stats.update_user_mapping(111, "OriginalName")

        # Verify initial mapping
        async with real_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (111,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == "OriginalName"

        # Update the mapping
        await real_user_stats.update_user_mapping(111, "NewName")

        # Verify updated mapping
        async with real_database.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?", (111,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == "NewName"

    async def test_bulk_user_mapping(self, real_user_stats, real_database):
        """Test adding multiple user mappings."""
        users = [
            (111, "Alice"),
            (222, "Bob"),
            (333, "Charlie"),
            (444, "Diana"),
        ]

        for author_id, username in users:
            await real_user_stats.update_user_mapping(author_id, username)

        # Verify all mappings
        async with real_database.conn.execute("SELECT COUNT(*) FROM user_mapping") as cursor:
            count = (await cursor.fetchone())[0]
            assert count == 4


class TestStatsResetFlow:
    """Integration tests for statistics reset operations."""

    async def test_reset_clears_all_data(self, real_user_stats, real_database):
        """Test that reset_stats clears all user statistics."""
        # Add some data
        await real_user_stats.update_user_mapping(111, "Test")
        await real_user_stats.batch_update_stats(
            [
                (111, 5, datetime.datetime(2024, 1, 1)),
                (111, 10, datetime.datetime(2024, 2, 1)),
            ]
        )
        await real_user_stats.batch_update_reaction_stats(
            [
                (111, 222, datetime.datetime(2024, 1, 5)),
            ]
        )

        # Verify data exists
        async with real_database.conn.execute("SELECT COUNT(*) FROM user_stats_monthly") as cursor:
            assert (await cursor.fetchone())[0] > 0

        # Reset
        await real_user_stats.reset_stats()

        # Verify stats are cleared
        async with real_database.conn.execute("SELECT COUNT(*) FROM user_stats_monthly") as cursor:
            assert (await cursor.fetchone())[0] == 0

        async with real_database.conn.execute(
            "SELECT COUNT(*) FROM user_reactions_monthly"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0


class TestBatchOperations:
    """Integration tests for batch data operations."""

    async def test_batch_stats_update_performance(self, real_user_stats, real_database):
        """Test that batch updates handle large datasets correctly."""
        # Create 100 stat entries
        stats_data = [
            (1000 + i, i % 10, datetime.datetime(2024, 1, i % 28 + 1)) for i in range(100)
        ]

        await real_user_stats.batch_update_stats(stats_data)

        # Verify all entries were created
        async with real_database.conn.execute("SELECT COUNT(*) FROM user_stats_monthly") as cursor:
            count = (await cursor.fetchone())[0]
            assert count == 100

    async def test_batch_reaction_update(self, real_user_stats, real_database):
        """Test batch reaction stats updates."""
        # Create 50 reaction relationships
        reaction_data = [
            (100 + i, 200 + (i % 10), datetime.datetime(2024, 1, 15)) for i in range(50)
        ]

        await real_user_stats.batch_update_reaction_stats(reaction_data)

        # Verify all relationships were created
        async with real_database.conn.execute(
            "SELECT COUNT(*) FROM user_reactions_monthly"
        ) as cursor:
            count = (await cursor.fetchone())[0]
            # Some entries will be combined if same giver->receiver in same month
            assert count <= 50
            assert count > 0


class TestEdgeCases:
    """Integration tests for edge cases in user statistics."""

    async def test_zero_reaction_messages(self, real_user_stats):
        """Test handling of messages with zero reactions."""
        await real_user_stats.update_user_mapping(111, "ZeroReacts")

        stats_data = [
            (111, 0, datetime.datetime(2024, 1, 1)),
            (111, 0, datetime.datetime(2024, 1, 5)),
            (111, 0, datetime.datetime(2024, 1, 10)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        user_stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)

        assert user_stats["total_msgs"] == 3
        assert user_stats["total_reacts"] == 0
        assert user_stats["avg_reacts"] == 0.0

    async def test_single_message_statistics(self, real_user_stats):
        """Test statistics for a user with only one message."""
        await real_user_stats.update_user_mapping(111, "SingleMsg")

        await real_user_stats.batch_update_stats(
            [
                (111, 7, datetime.datetime(2024, 1, 15)),
            ]
        )

        user_stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)

        assert user_stats["total_msgs"] == 1
        assert user_stats["total_reacts"] == 7
        assert user_stats["avg_reacts"] == pytest.approx(7.0, rel=0.01)

    async def test_high_reaction_count(self, real_user_stats):
        """Test handling of very high reaction counts."""
        await real_user_stats.update_user_mapping(111, "PopularUser")

        # Message with many reactions
        await real_user_stats.batch_update_stats(
            [
                (111, 1000, datetime.datetime(2024, 1, 1)),
            ]
        )

        user_stats = await real_user_stats.get_user_monthly_stats(111, 2024, 1)

        assert user_stats["total_reacts"] == 1000
        assert user_stats["avg_reacts"] == pytest.approx(1000.0, rel=0.01)
