"""
Tests for the !connections command.
"""

import discord.ext.test as dpytest
import pytest


class TestConnectionsCommand:
    """Tests for the !connections command."""

    @pytest.mark.asyncio
    async def test_connections_returns_top_relationships(self, bot_with_mocked_db):
        """Test that !connections returns top mutual relationships."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Mock reaction network data (dict format from get_reaction_network_for_month)
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {"giver_username": "Alice", "receiver_username": "Bob", "reaction_count": 50},
            {"giver_username": "Bob", "receiver_username": "Alice", "reaction_count": 40},
            {"giver_username": "Alice", "receiver_username": "Charlie", "reaction_count": 30},
            {"giver_username": "Charlie", "receiver_username": "Alice", "reaction_count": 20},
        ]

        await dpytest.message("!connections")

        # Should return a message with top connections
        response = dpytest.get_message()
        assert "Top 10 Mutual Relationships" in response.content
        assert "Alice" in response.content
        assert "Bob" in response.content

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
            {"giver_username": "Alice", "receiver_username": "Bob", "reaction_count": 100},
            {"giver_username": "Bob", "receiver_username": "Alice", "reaction_count": 80},
        ]

        await dpytest.message("!connections")

        # Should call get_reaction_network_for_month (not rolling)
        mock_user_stats.get_reaction_network_for_month.assert_called_once()
        response = dpytest.get_message()
        assert "Top 10 Mutual Relationships" in response.content

    @pytest.mark.asyncio
    async def test_connections_shows_affinity_scores(self, bot_with_mocked_db):
        """Test that !connections shows affinity scores."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Set up data where we can calculate expected affinity
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {"giver_username": "Alice", "receiver_username": "Bob", "reaction_count": 50},
            {"giver_username": "Alice", "receiver_username": "Charlie", "reaction_count": 30},
            {"giver_username": "Bob", "receiver_username": "Alice", "reaction_count": 40},
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        # Should show the affinity score with <-> notation
        assert "<->" in response.content

    @pytest.mark.asyncio
    async def test_connections_handles_no_mutual_connections(self, bot_with_mocked_db):
        """Test that !connections handles case with no bidirectional edges."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Only unidirectional reactions - no mutual connections
        mock_user_stats.get_reaction_network_for_month.return_value = [
            {"giver_username": "Alice", "receiver_username": "Bob", "reaction_count": 50},
            {"giver_username": "Charlie", "receiver_username": "Dave", "reaction_count": 30},
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        assert "No mutual connections found for" in response.content
