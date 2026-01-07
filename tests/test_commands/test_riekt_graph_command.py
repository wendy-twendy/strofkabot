"""Tests for the !riekt-graph command."""

import discord.ext.test as dpytest
import pytest


class TestRiektGraphCommand:
    """Tests for the !riekt-graph command."""

    @pytest.mark.asyncio
    async def test_riekt_graph_handles_no_data(self, bot_with_mocked_db):
        """Test that !riekt-graph handles empty database gracefully."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        # Mock the user_stats to return empty data
        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!riekt-graph")
        assert dpytest.verify().message().content("No reaction data available for this period.")

    @pytest.mark.asyncio
    async def test_riekt_graph_validates_months_min(self, bot_with_mocked_db):
        """Test that !riekt-graph validates months parameter (min)."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        await dpytest.message("!riekt-graph 0")
        assert dpytest.verify().message().content("Number of months must be at least 1.")

    @pytest.mark.asyncio
    async def test_riekt_graph_validates_months_max(self, bot_with_mocked_db):
        """Test that !riekt-graph validates months parameter (max)."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        await dpytest.message("!riekt-graph 13")
        assert dpytest.verify().message().content("Number of months must be at most 12.")

    @pytest.mark.asyncio
    async def test_riekt_graph_calls_user_stats(self, bot_with_mocked_db):
        """Test that !riekt-graph calls the user_stats method."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!riekt-graph")
        mock_user_stats.get_reaction_network_rolling.assert_called_once()

    @pytest.mark.asyncio
    async def test_riekt_graph_with_custom_months(self, bot_with_mocked_db):
        """Test that !riekt-graph accepts custom months parameter."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!riekt-graph 3")
        # Should have been called (exact args depend on current date)
        mock_user_stats.get_reaction_network_rolling.assert_called_once()

    @pytest.mark.asyncio
    async def test_riekt_graph_generates_image_with_data(self, bot_with_mocked_db):
        """Test that !riekt-graph generates an image when data is available."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        # Mock data with bidirectional reactions for community detection
        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 10),
            ("Bob", "Alice", 8),
            ("Charlie", "Diana", 5),
            ("Diana", "Charlie", 6),
            ("Alice", "Charlie", 2),
            ("Charlie", "Alice", 3),
        ]

        await dpytest.message("!riekt-graph")

        # Verify a message was sent (will contain an attachment)
        # dpytest doesn't easily verify file attachments, but we can check the message exists
        msg = dpytest.get_message()
        assert msg is not None
        assert "Reaction Network Graph" in msg.content or len(msg.attachments) > 0
