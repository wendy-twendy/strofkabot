"""Tests for NetworkCog class."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.cogs.network import NetworkCog


class TestNetworkCogInit:
    """Tests for NetworkCog initialization."""

    def test_init_stores_dependencies(self):
        """Test that all dependencies are stored correctly."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        guild = MagicMock()
        logger = logging.getLogger("test")

        cog = NetworkCog(bot, db, user_stats, guild, logger)

        assert cog.bot is bot
        assert cog.db is db
        assert cog.user_stats is user_stats
        assert cog.guild is guild
        assert cog.logger is logger

    def test_set_guild(self):
        """Test that set_guild updates the guild."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        cog = NetworkCog(bot, db, user_stats, None, logger)
        assert cog.guild is None

        new_guild = MagicMock()
        cog.set_guild(new_guild)
        assert cog.guild is new_guild


class TestShowRiektGraph:
    """Tests for show_riekt_graph command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        guild = MagicMock()
        logger = logging.getLogger("test")
        return NetworkCog(bot, db, user_stats, guild, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    async def test_rejects_months_less_than_1(self, cog, mock_ctx):
        """Test that months < 1 is rejected."""
        await cog.show_riekt_graph.callback(cog, mock_ctx, months=0)
        mock_ctx.send.assert_called_with("Number of months must be at least 1.")

    async def test_rejects_months_greater_than_12(self, cog, mock_ctx):
        """Test that months > 12 is rejected."""
        await cog.show_riekt_graph.callback(cog, mock_ctx, months=13)
        mock_ctx.send.assert_called_with("Number of months must be at most 12.")

    async def test_handles_no_reaction_data(self, cog, mock_ctx):
        """Test handling when no reaction data is available."""
        cog.user_stats.get_reaction_network_rolling = AsyncMock(return_value=[])

        with patch("strofkabot.cogs.network.get_rolling_start_month", return_value=(2024, 1)):
            await cog.show_riekt_graph.callback(cog, mock_ctx, months=3)

        mock_ctx.send.assert_called_with("No reaction data available for this period.")

    async def test_handles_no_affinities(self, cog, mock_ctx):
        """Test handling when no affinities can be computed."""
        cog.user_stats.get_reaction_network_rolling = AsyncMock(return_value=[("A", "B", 1)])

        with (
            patch("strofkabot.cogs.network.get_rolling_start_month", return_value=(2024, 1)),
            patch("strofkabot.cogs.network.build_reaction_graph"),
            patch("strofkabot.cogs.network.compute_all_affinities", return_value=[]),
        ):
            await cog.show_riekt_graph.callback(cog, mock_ctx, months=3)

        mock_ctx.send.assert_called_with("Not enough mutual connections to build a graph.")

    async def test_handles_exception(self, cog, mock_ctx):
        """Test that exceptions are handled gracefully."""
        cog.user_stats.get_reaction_network_rolling = AsyncMock(
            side_effect=Exception("Database error")
        )

        with patch("strofkabot.cogs.network.get_rolling_start_month", return_value=(2024, 1)):
            await cog.show_riekt_graph.callback(cog, mock_ctx, months=3)

        mock_ctx.send.assert_called_with("An error occurred while generating the graph.")


class TestShowConnections:
    """Tests for show_connections command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        bot.wait_for = AsyncMock(side_effect=TimeoutError)
        db = MagicMock()
        user_stats = MagicMock()
        guild = MagicMock()
        logger = logging.getLogger("test")
        return NetworkCog(bot, db, user_stats, guild, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    async def test_default_shows_per_user_connections(self, cog, mock_ctx):
        """Test that default mode shows per-user connections."""
        with patch.object(cog, "_send_connections", new_callable=AsyncMock) as mock_send:
            await cog.show_connections.callback(cog, mock_ctx, args="")
            mock_send.assert_called_once_with(mock_ctx, month_offset=0, show_top_pairs=False)

    async def test_top_flag_shows_pairs(self, cog, mock_ctx):
        """Test that --top flag shows top pairs."""
        with patch.object(cog, "_send_connections", new_callable=AsyncMock) as mock_send:
            await cog.show_connections.callback(cog, mock_ctx, args="--top")
            mock_send.assert_called_once_with(mock_ctx, month_offset=0, show_top_pairs=True)

    async def test_handles_no_reaction_data(self, cog, mock_ctx):
        """Test handling when no reaction data is available."""
        cog.user_stats.get_reaction_network_for_month = AsyncMock(return_value=[])

        await cog._send_connections(mock_ctx, month_offset=0)

        assert "No reaction data" in mock_ctx.send.call_args[0][0]

    async def test_handles_no_affinities(self, cog, mock_ctx):
        """Test handling when no affinities can be computed."""
        cog.user_stats.get_reaction_network_for_month = AsyncMock(
            return_value=[{"giver_username": "A", "receiver_username": "B", "reaction_count": 1}]
        )

        with (
            patch("strofkabot.cogs.network.build_reaction_graph"),
            patch("strofkabot.cogs.network.compute_all_affinities", return_value=[]),
        ):
            await cog._send_connections(mock_ctx, month_offset=0)

        assert "No mutual connections" in mock_ctx.send.call_args[0][0]

    async def test_handles_no_active_users(self, cog, mock_ctx):
        """Test handling when no users meet message threshold."""
        cog.user_stats.get_reaction_network_for_month = AsyncMock(
            return_value=[
                {
                    "giver_username": "A",
                    "receiver_username": "B",
                    "reaction_count": 5,
                    "giver_messages": 10,  # Below 30 threshold
                    "receiver_messages": 10,
                }
            ]
        )

        with (
            patch("strofkabot.cogs.network.build_reaction_graph"),
            patch("strofkabot.cogs.network.compute_all_affinities", return_value=[("A", "B", 0.5)]),
        ):
            await cog._send_connections(mock_ctx, month_offset=0)

        assert "No active users" in mock_ctx.send.call_args[0][0]

    async def test_handles_exception(self, cog, mock_ctx):
        """Test that exceptions are handled gracefully."""
        cog.user_stats.get_reaction_network_for_month = AsyncMock(
            side_effect=Exception("Database error")
        )

        await cog._send_connections(mock_ctx, month_offset=0)

        mock_ctx.send.assert_called_with("An error occurred while generating the connections.")


class TestShowClusters:
    """Tests for show_clusters command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        bot.wait_for = AsyncMock(side_effect=TimeoutError)
        db = MagicMock()
        user_stats = MagicMock()
        guild = MagicMock()
        logger = logging.getLogger("test")
        return NetworkCog(bot, db, user_stats, guild, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    async def test_rejects_months_less_than_1(self, cog, mock_ctx):
        """Test that months < 1 is rejected."""
        await cog.show_clusters.callback(cog, mock_ctx, months=0)
        mock_ctx.send.assert_called_with("Number of months must be at least 1.")

    async def test_rejects_months_greater_than_12(self, cog, mock_ctx):
        """Test that months > 12 is rejected."""
        await cog.show_clusters.callback(cog, mock_ctx, months=13)
        mock_ctx.send.assert_called_with("Number of months must be at most 12.")

    async def test_handles_no_reply_data(self, cog, mock_ctx):
        """Test handling when no reply data is available."""
        cog.user_stats.get_reply_network_for_months = AsyncMock(return_value=[])

        await cog._send_clusters(mock_ctx, months=3, month_offset=0)

        assert "No reply data" in mock_ctx.send.call_args[0][0]

    async def test_handles_no_bidirectional_connections(self, cog, mock_ctx):
        """Test handling when no bidirectional connections exist."""
        # Only A->B, no B->A
        cog.user_stats.get_reply_network_for_months = AsyncMock(return_value=[("A", "B", 10)])

        await cog._send_clusters(mock_ctx, months=3, month_offset=0)

        assert "No strong mutual connections" in mock_ctx.send.call_args[0][0]

    async def test_handles_exception(self, cog, mock_ctx):
        """Test that exceptions are handled gracefully."""
        cog.user_stats.get_reply_network_for_months = AsyncMock(
            side_effect=Exception("Database error")
        )

        await cog._send_clusters(mock_ctx, months=3, month_offset=0)

        mock_ctx.send.assert_called_with("An error occurred while generating clusters.")


class TestShowEchoChamber:
    """Tests for show_echo_chamber command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        db = MagicMock()
        db.get_username_by_id = AsyncMock(return_value=None)
        user_stats = MagicMock()
        guild = MagicMock()
        guild.get_member = MagicMock(return_value=None)
        logger = logging.getLogger("test")
        return NetworkCog(bot, db, user_stats, guild, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.author.display_name = "TestUser"
        ctx.send = AsyncMock()
        return ctx

    async def test_uses_author_when_no_member_specified(self, cog, mock_ctx):
        """Test that author is used when no member is specified."""
        cog.user_stats.get_echo_chamber_data = AsyncMock(
            return_value={"outgoing": [], "incoming": []}
        )

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=None)

        # Should use ctx.author
        assert "TestUser" in mock_ctx.send.call_args[0][0]

    async def test_uses_specified_member(self, cog, mock_ctx):
        """Test that specified member is used."""
        target_member = MagicMock()
        target_member.id = 67890
        target_member.display_name = "TargetUser"

        cog.user_stats.get_echo_chamber_data = AsyncMock(
            return_value={"outgoing": [], "incoming": []}
        )

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=target_member)

        assert "TargetUser" in mock_ctx.send.call_args[0][0]

    async def test_handles_no_data(self, cog, mock_ctx):
        """Test handling when no echo chamber data is available."""
        cog.user_stats.get_echo_chamber_data = AsyncMock(
            return_value={"outgoing": [], "incoming": []}
        )

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=None)

        assert "No reaction data found" in mock_ctx.send.call_args[0][0]

    async def test_displays_metrics_with_data(self, cog, mock_ctx):
        """Test that metrics are displayed when data exists."""
        # Create member mocks for resolving names
        member1 = MagicMock()
        member1.display_name = "User1"
        member2 = MagicMock()
        member2.display_name = "User2"

        cog.guild.get_member = MagicMock(
            side_effect=lambda uid: {111: member1, 222: member2}.get(uid)
        )

        cog.user_stats.get_echo_chamber_data = AsyncMock(
            return_value={
                "outgoing": [(111, 10), (222, 5)],
                "incoming": [(111, 8), (222, 4)],
            }
        )

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=None)

        response = mock_ctx.send.call_args[0][0]
        assert "Echo Chamber Analysis" in response
        assert "Outgoing Reactions" in response
        assert "Incoming Reactions" in response
        assert "Echo Chamber Index" in response

    async def test_handles_exception(self, cog, mock_ctx):
        """Test that exceptions are handled gracefully."""
        cog.user_stats.get_echo_chamber_data = AsyncMock(side_effect=Exception("Database error"))

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=None)

        mock_ctx.send.assert_called_with(
            "An error occurred while generating the echo chamber analysis."
        )

    async def test_resolves_usernames_from_database(self, cog, mock_ctx):
        """Test that usernames are resolved from database when not in guild."""
        # guild.get_member returns None, so fall back to db
        cog.guild.get_member = MagicMock(return_value=None)
        cog.db.get_username_by_id = AsyncMock(return_value="DatabaseUser")

        cog.user_stats.get_echo_chamber_data = AsyncMock(
            return_value={
                "outgoing": [(111, 10)],
                "incoming": [(222, 5)],
            }
        )

        await cog.show_echo_chamber.callback(cog, mock_ctx, member=None)

        response = mock_ctx.send.call_args[0][0]
        assert "DatabaseUser" in response
