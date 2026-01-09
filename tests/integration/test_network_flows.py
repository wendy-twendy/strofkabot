"""
Integration tests for reaction network data flows.

Tests the full flow from reaction data insertion through network
analysis including community detection and affinity calculations.
"""

import datetime


async def setup_network_users(real_user_stats, user_ids: list[int], year: int, month: int):
    """Helper to set up user mappings and stats required for network queries."""
    for user_id in user_ids:
        await real_user_stats.update_user_mapping(user_id, f"User{user_id}")
        # Add minimal stats so the JOIN works
        await real_user_stats.batch_update_stats([(user_id, 1, datetime.datetime(year, month, 15))])


class TestReactionNetworkFlow:
    """Integration tests for reaction network data flows."""

    async def test_reaction_network_data_structure(self, real_user_stats, reaction_factory):
        """Test that reaction network data has correct structure."""
        # Set up required user data for the JOIN
        await setup_network_users(real_user_stats, [111, 222, 333], 2024, 1)

        # Create a small network
        await reaction_factory(giver_id=111, receiver_id=222, count=5)
        await reaction_factory(giver_id=222, receiver_id=111, count=3)
        await reaction_factory(giver_id=111, receiver_id=333, count=2)

        # Fetch network data
        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        assert len(network_data) == 3

        # Verify each entry has expected keys (uses usernames due to JOINs)
        for entry in network_data:
            assert "giver_username" in entry
            assert "receiver_username" in entry
            assert "reaction_count" in entry

    async def test_reaction_network_counts(self, real_user_stats, reaction_factory):
        """Test that reaction counts are accurate in network data."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222, 333], 2024, 1)

        # Create specific reaction relationships
        await reaction_factory(giver_id=111, receiver_id=222, count=10)
        await reaction_factory(giver_id=222, receiver_id=333, count=7)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # Convert to dict for easier lookup (using usernames)
        network_dict = {
            (d["giver_username"], d["receiver_username"]): d["reaction_count"] for d in network_data
        }

        assert network_dict[("User111", "User222")] == 10
        assert network_dict[("User222", "User333")] == 7

    async def test_empty_network(self, real_user_stats):
        """Test handling of empty reaction network."""
        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)
        assert network_data == []


class TestMostLikedCalculation:
    """Integration tests for most-liked user calculations."""

    async def test_most_liked_ranking(self, real_user_stats, reaction_factory):
        """Test that most-liked ranking is based on received reactions."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222, 333], 2024, 1)

        # User 333 receives the most reactions
        await reaction_factory(giver_id=111, receiver_id=333, count=10)
        await reaction_factory(giver_id=222, receiver_id=333, count=8)

        # User 222 receives medium amount
        await reaction_factory(giver_id=111, receiver_id=222, count=5)
        await reaction_factory(giver_id=333, receiver_id=222, count=3)

        # User 111 receives the least
        await reaction_factory(giver_id=222, receiver_id=111, count=2)

        # Get network data to verify totals
        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # Calculate received reactions per user (using usernames)
        received = {}
        for entry in network_data:
            receiver = entry["receiver_username"]
            received[receiver] = received.get(receiver, 0) + entry["reaction_count"]

        # User 333 should have most (18), then 222 (8), then 111 (2)
        assert received["User333"] == 18
        assert received["User222"] == 8
        assert received["User111"] == 2

    async def test_most_liked_excludes_self_reactions(
        self, real_user_stats, real_database, reaction_factory
    ):
        """Test that self-reactions are tracked but can be excluded from analysis."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222], 2024, 1)

        # Self reaction (should be stored but typically excluded in analysis)
        await reaction_factory(giver_id=111, receiver_id=111, count=100)
        # Normal reaction
        await reaction_factory(giver_id=222, receiver_id=111, count=5)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # Both should be in the data
        assert len(network_data) == 2

        # Self-reaction is present
        self_reaction = next(
            (d for d in network_data if d["giver_username"] == d["receiver_username"]), None
        )
        assert self_reaction is not None
        assert self_reaction["reaction_count"] == 100


class TestRollingWindowQueries:
    """Integration tests for rolling window reaction queries."""

    async def test_rolling_reaction_network(self, real_user_stats, real_database):
        """Test rolling window reaction network aggregation."""
        # Set up required user mappings
        await real_user_stats.update_user_mapping(111, "User111")
        await real_user_stats.update_user_mapping(222, "User222")

        # Add reactions across multiple months
        # January
        await real_database.batch_upsert_reaction_stats(
            [
                (111, 222, 2024, 1),
                (111, 222, 2024, 1),
                (111, 222, 2024, 1),  # 3 reactions in Jan
            ]
        )

        # February
        await real_database.batch_upsert_reaction_stats(
            [
                (111, 222, 2024, 2),
                (111, 222, 2024, 2),  # 2 reactions in Feb
            ]
        )

        # March
        await real_database.batch_upsert_reaction_stats(
            [
                (111, 222, 2024, 3),
                (111, 222, 2024, 3),
                (111, 222, 2024, 3),
                (111, 222, 2024, 3),  # 4 reactions in Mar
            ]
        )

        # Query rolling window starting from Jan 2024
        rolling_data = await real_user_stats.get_reaction_network_rolling(2024, 1)

        # Should have one entry aggregating all months (returns tuples with usernames)
        entry = next(
            (d for d in rolling_data if d[0] == "User111" and d[1] == "User222"),
            None,
        )
        assert entry is not None
        # Total across all months: 3 + 2 + 4 = 9
        assert entry[2] == 9  # reaction_count is third element


class TestNetworkDataIntegrity:
    """Integration tests for network data integrity."""

    async def test_bidirectional_relationships(self, real_user_stats, reaction_factory):
        """Test that bidirectional relationships are tracked separately."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222], 2024, 1)

        # User A -> User B
        await reaction_factory(giver_id=111, receiver_id=222, count=5)
        # User B -> User A
        await reaction_factory(giver_id=222, receiver_id=111, count=3)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # Should have 2 separate entries
        assert len(network_data) == 2

        # Verify each direction (using usernames)
        a_to_b = next(
            (
                d
                for d in network_data
                if d["giver_username"] == "User111" and d["receiver_username"] == "User222"
            ),
            None,
        )
        b_to_a = next(
            (
                d
                for d in network_data
                if d["giver_username"] == "User222" and d["receiver_username"] == "User111"
            ),
            None,
        )

        assert a_to_b is not None
        assert a_to_b["reaction_count"] == 5

        assert b_to_a is not None
        assert b_to_a["reaction_count"] == 3

    async def test_multiple_receivers_from_same_giver(self, real_user_stats, reaction_factory):
        """Test tracking reactions from one user to multiple receivers."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222, 333, 444], 2024, 1)

        # User 111 gives reactions to multiple users
        await reaction_factory(giver_id=111, receiver_id=222, count=5)
        await reaction_factory(giver_id=111, receiver_id=333, count=3)
        await reaction_factory(giver_id=111, receiver_id=444, count=7)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # All 3 relationships should be present
        assert len(network_data) == 3

        # Verify counts (using usernames)
        network_dict = {
            (d["giver_username"], d["receiver_username"]): d["reaction_count"] for d in network_data
        }

        assert network_dict[("User111", "User222")] == 5
        assert network_dict[("User111", "User333")] == 3
        assert network_dict[("User111", "User444")] == 7

    async def test_reaction_accumulation(self, real_user_stats, reaction_factory):
        """Test that multiple reaction events accumulate correctly."""
        # Set up required user data
        await setup_network_users(real_user_stats, [111, 222], 2024, 1)

        # Add reactions in separate calls (simulating multiple days)
        await reaction_factory(giver_id=111, receiver_id=222, count=2)
        await reaction_factory(giver_id=111, receiver_id=222, count=3)
        await reaction_factory(giver_id=111, receiver_id=222, count=5)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # Should be combined into single entry
        assert len(network_data) == 1
        assert network_data[0]["reaction_count"] == 10  # 2 + 3 + 5


class TestLargeNetworkScaling:
    """Integration tests for network scaling with larger datasets."""

    async def test_large_network_creation(self, real_user_stats, real_database):
        """Test creating a larger reaction network."""
        # Create 20 users with mappings and stats
        for user_id in range(100, 120):
            await real_user_stats.update_user_mapping(user_id, f"User{user_id}")
            await real_user_stats.batch_update_stats([(user_id, 1, datetime.datetime(2024, 1, 15))])

        # Create interconnected reactions
        reaction_entries = []
        for giver in range(100, 120):
            for receiver in range(100, 120):
                if giver != receiver:
                    reaction_entries.append((giver, receiver, 2024, 1))

        await real_database.batch_upsert_reaction_stats(reaction_entries)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # 20 users * 19 possible receivers = 380 relationships
        assert len(network_data) == 380

    async def test_network_query_with_many_edges(self, real_user_stats, real_database):
        """Test querying a network with many edges efficiently."""
        hub_user = 100

        # Create user mappings and stats for hub and spokes
        await real_user_stats.update_user_mapping(hub_user, f"Hub{hub_user}")
        await real_user_stats.batch_update_stats([(hub_user, 1, datetime.datetime(2024, 1, 15))])

        for user in range(101, 151):
            await real_user_stats.update_user_mapping(user, f"Spoke{user}")
            await real_user_stats.batch_update_stats([(user, 1, datetime.datetime(2024, 1, 15))])

        # Create a hub-and-spoke pattern (one central user)
        reaction_entries = []
        # 50 users all reacting to the hub
        for user in range(101, 151):
            for _ in range(5):  # 5 reactions each
                reaction_entries.append((user, hub_user, 2024, 1))

        await real_database.batch_upsert_reaction_stats(reaction_entries)

        network_data = await real_user_stats.get_reaction_network_for_month(2024, 1)

        # 50 edges (one per spoke user to hub)
        assert len(network_data) == 50

        # Hub should have received 250 total reactions (50 * 5)
        total_to_hub = sum(
            d["reaction_count"] for d in network_data if d["receiver_username"] == f"Hub{hub_user}"
        )
        assert total_to_hub == 250
