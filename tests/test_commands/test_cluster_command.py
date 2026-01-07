"""Tests for the !cluster command."""

import discord.ext.test as dpytest
import pytest


class TestClusterCommand:
    """Tests for the !cluster command."""

    @pytest.mark.asyncio
    async def test_cluster_handles_no_data(self, bot_with_mocked_db):
        """Test that !cluster handles empty database gracefully."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!cluster")
        msg = dpytest.get_message()
        assert msg is not None
        assert "No reaction data available" in msg.content

    @pytest.mark.asyncio
    async def test_cluster_validates_months_min(self, bot_with_mocked_db):
        """Test that !cluster validates months parameter (min)."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        await dpytest.message("!cluster 0")
        assert dpytest.verify().message().content("Number of months must be at least 1.")

    @pytest.mark.asyncio
    async def test_cluster_validates_months_max(self, bot_with_mocked_db):
        """Test that !cluster validates months parameter (max)."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        await dpytest.message("!cluster 13")
        assert dpytest.verify().message().content("Number of months must be at most 12.")

    @pytest.mark.asyncio
    async def test_cluster_calls_user_stats(self, bot_with_mocked_db):
        """Test that !cluster calls the user_stats method."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!cluster")
        mock_user_stats.get_reaction_network_rolling.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_with_custom_months(self, bot_with_mocked_db):
        """Test that !cluster accepts custom months parameter."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        mock_user_stats.get_reaction_network_rolling.return_value = []

        await dpytest.message("!cluster 3")
        mock_user_stats.get_reaction_network_rolling.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_generates_text_output_with_data(self, bot_with_mocked_db):
        """Test that !cluster generates text output when data is available."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        # High reaction counts to pass min_activity=20 threshold
        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 15),
            ("Bob", "Alice", 15),
            ("Charlie", "Diana", 12),
            ("Diana", "Charlie", 12),
            ("Alice", "Charlie", 5),
            ("Charlie", "Alice", 5),
        ]

        await dpytest.message("!cluster")

        msg = dpytest.get_message()
        assert msg is not None
        assert "Social Clusters" in msg.content
        assert "communities" in msg.content

    @pytest.mark.asyncio
    async def test_cluster_shows_member_names(self, bot_with_mocked_db):
        """Test that !cluster shows member names in groups."""
        bot, mock_db, mock_user_stats, cog = bot_with_mocked_db

        # High reaction counts to pass min_activity=20 threshold
        mock_user_stats.get_reaction_network_rolling.return_value = [
            ("Alice", "Bob", 15),
            ("Bob", "Alice", 15),
        ]

        await dpytest.message("!cluster")

        msg = dpytest.get_message()
        assert msg is not None
        assert "Alice" in msg.content
        assert "Bob" in msg.content
