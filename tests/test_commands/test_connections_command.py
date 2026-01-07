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

        # Mock reaction network data
        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 50),
            ("Bob", "Alice", 40),
            ("Alice", "Charlie", 30),
            ("Charlie", "Alice", 20),
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

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!connections")

        assert dpytest.verify().message().content("No reaction data available for this period.")

    @pytest.mark.asyncio
    async def test_connections_accepts_months_parameter(self, bot_with_mocked_db):
        """Test that !connections accepts a months parameter."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 100),
            ("Bob", "Alice", 80),
        ]

        await dpytest.message("!connections 3")

        # Should call with correct rolling window
        mock_user_stats.get_reaction_network_rolling.assert_called_once()
        response = dpytest.get_message()
        assert "Top 10 Mutual Relationships" in response.content

    @pytest.mark.asyncio
    async def test_connections_validates_months_minimum(self, bot_with_mocked_db):
        """Test that !connections validates months >= 1."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        await dpytest.message("!connections 0")

        assert dpytest.verify().message().content("Number of months must be at least 1.")

    @pytest.mark.asyncio
    async def test_connections_validates_months_maximum(self, bot_with_mocked_db):
        """Test that !connections validates months <= 12."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        await dpytest.message("!connections 13")

        assert dpytest.verify().message().content("Number of months must be at most 12.")

    @pytest.mark.asyncio
    async def test_connections_shows_affinity_scores(self, bot_with_mocked_db):
        """Test that !connections shows affinity scores."""
        bot, mock_db, mock_user_stats, _ = bot_with_mocked_db

        # Set up data where we can calculate expected affinity
        # Alice gives 50 to Bob out of 80 total (50/80 = 0.625)
        # Bob gives 40 to Alice out of 40 total (40/40 = 1.0)
        # Affinity = sqrt(0.625 * 1.0) = 0.791
        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 50),
            ("Alice", "Charlie", 30),
            ("Bob", "Alice", 40),
        ]

        await dpytest.message("!connections")

        response = dpytest.get_message()
        # Should show the affinity score
        assert "<->" in response.content or "↔" in response.content
