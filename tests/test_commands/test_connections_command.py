"""
Tests for the !connections command.
"""

import discord.ext.test as dpytest
import pytest

from strofkabot.utils import compute_top_connection_per_user


class TestConnectionsCommand:
    """Tests for the !connections command."""

    @pytest.mark.asyncio
    async def test_connections_returns_top_connections_per_user(self, bot_with_mocked_db):
        """Test that !connections returns top connection per active user by default."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Mock reaction network data with message counts for active users
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 35,
                "receiver_messages": 40,
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 40,
                "giver_messages": 40,
                "receiver_messages": 35,
            },
            {
                "giver_username": "Alice",
                "receiver_username": "Charlie",
                "reaction_count": 30,
                "giver_messages": 35,
                "receiver_messages": 50,
            },
            {
                "giver_username": "Charlie",
                "receiver_username": "Alice",
                "reaction_count": 20,
                "giver_messages": 50,
                "receiver_messages": 35,
            },
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        assert "Top Connections by User" in response.content
        assert "Users with 30+ messages" in response.content
        assert "Alice" in response.content
        assert "Bob" in response.content

    @pytest.mark.asyncio
    async def test_connections_top_flag_returns_mutual_relationships(self, bot_with_mocked_db):
        """Test that !connections --top returns top 10 mutual relationships."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 35,
                "receiver_messages": 40,
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 40,
                "giver_messages": 40,
                "receiver_messages": 35,
            },
        ]

        await dpytest.message("!connections --top")

        response = dpytest.get_message()
        assert "Top 10 Mutual Relationships" in response.content
        assert "<->" in response.content

    @pytest.mark.asyncio
    async def test_connections_handles_no_data(self, bot_with_mocked_db):
        """Test that !connections handles empty data gracefully."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_for_month.return_value = []

        await dpytest.message("!connections")

        response = dpytest.get_message()
        assert "No reaction data available for" in response.content

    @pytest.mark.asyncio
    async def test_connections_shows_current_month(self, bot_with_mocked_db):
        """Test that !connections shows current month by default."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 100,
                "giver_messages": 50,
                "receiver_messages": 50,
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 80,
                "giver_messages": 50,
                "receiver_messages": 50,
            },
        ]

        await dpytest.message("!connections")

        mock_user_stats.get_reaction_network_for_month.assert_called_once()
        response = dpytest.get_message()
        assert "Top Connections by User" in response.content

    @pytest.mark.asyncio
    async def test_connections_shows_affinity_scores(self, bot_with_mocked_db):
        """Test that !connections shows affinity scores."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 35,
                "receiver_messages": 40,
            },
            {
                "giver_username": "Alice",
                "receiver_username": "Charlie",
                "reaction_count": 30,
                "giver_messages": 35,
                "receiver_messages": 50,
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 40,
                "giver_messages": 40,
                "receiver_messages": 35,
            },
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        # Should show the affinity score with -> notation for per-user format
        assert "->" in response.content

    @pytest.mark.asyncio
    async def test_connections_handles_no_mutual_connections(self, bot_with_mocked_db):
        """Test that !connections handles case with no bidirectional edges."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Only unidirectional reactions - no mutual connections
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 35,
                "receiver_messages": 40,
            },
            {
                "giver_username": "Charlie",
                "receiver_username": "Dave",
                "reaction_count": 30,
                "giver_messages": 50,
                "receiver_messages": 60,
            },
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        assert "No mutual connections found for" in response.content

    @pytest.mark.asyncio
    async def test_connections_filters_inactive_users(self, bot_with_mocked_db):
        """Test that !connections filters out users with < 30 messages."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Alice and Bob have mutual connection but Alice has < 30 messages
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 20,  # Below threshold
                "receiver_messages": 40,
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 40,
                "giver_messages": 40,
                "receiver_messages": 20,
            },
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        # Bob is active, Alice is not - so Bob should still appear
        assert "Top Connections by User" in response.content
        assert "Bob" in response.content

    @pytest.mark.asyncio
    async def test_connections_handles_no_active_users(self, bot_with_mocked_db):
        """Test that !connections handles case with no active users."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_for_month.return_value = [
            {
                "giver_username": "Alice",
                "receiver_username": "Bob",
                "reaction_count": 50,
                "giver_messages": 20,  # Below threshold
                "receiver_messages": 25,  # Below threshold
            },
            {
                "giver_username": "Bob",
                "receiver_username": "Alice",
                "reaction_count": 40,
                "giver_messages": 25,
                "receiver_messages": 20,
            },
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        assert "No active users (30+ messages) found for" in response.content


class TestComputeTopConnectionPerUser:
    """Tests for the compute_top_connection_per_user function."""

    def test_returns_top_connection_for_each_active_user(self):
        """Test that each active user gets their top connection."""
        affinities = [
            ("Alice", "Bob", 0.5),
            ("Alice", "Charlie", 0.3),
            ("Bob", "Charlie", 0.2),
        ]
        active_users = {"Alice", "Bob", "Charlie"}

        result = compute_top_connection_per_user(affinities, active_users)

        # Should have one entry per active user
        users_in_result = {r[0] for r in result}
        assert users_in_result == active_users

        # Alice and Bob should both have the highest affinity (0.5)
        alice_entry = next(r for r in result if r[0] == "Alice")
        bob_entry = next(r for r in result if r[0] == "Bob")
        assert alice_entry[1] == "Bob"
        assert alice_entry[2] == 0.5
        assert bob_entry[1] == "Alice"
        assert bob_entry[2] == 0.5

        # Charlie's best is 0.3 with Alice (first seen in sorted order)
        charlie_entry = next(r for r in result if r[0] == "Charlie")
        assert charlie_entry[2] == 0.3

    def test_filters_inactive_users(self):
        """Test that inactive users are not included."""
        affinities = [
            ("Alice", "Bob", 0.5),
            ("Alice", "Charlie", 0.3),
        ]
        active_users = {"Alice"}  # Only Alice is active

        result = compute_top_connection_per_user(affinities, active_users)

        assert len(result) == 1
        assert result[0][0] == "Alice"

    def test_sorted_by_affinity_descending(self):
        """Test that results are sorted by affinity score."""
        affinities = [
            ("Alice", "Bob", 0.5),
            ("Charlie", "Dave", 0.8),
            ("Eve", "Frank", 0.3),
        ]
        active_users = {"Alice", "Charlie", "Dave", "Eve"}

        result = compute_top_connection_per_user(affinities, active_users)

        # Check that affinities are in descending order
        affinity_scores = [r[2] for r in result]
        assert affinity_scores == sorted(affinity_scores, reverse=True)

    def test_empty_affinities(self):
        """Test that empty affinities returns empty list."""
        result = compute_top_connection_per_user([], {"Alice", "Bob"})
        assert result == []

    def test_empty_active_users(self):
        """Test that empty active users returns empty list."""
        affinities = [("Alice", "Bob", 0.5)]
        result = compute_top_connection_per_user(affinities, set())
        assert result == []
