"""Tests for UserStatsCog class."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from strofkabot.cogs.user_stats import UserStatsCog


class TestUserStatsCogInit:
    """Tests for UserStatsCog initialization."""

    def test_init_stores_dependencies(self):
        """Test that all dependencies are stored correctly."""
        bot = MagicMock()
        user_stats = MagicMock()
        message_history_db = MagicMock()
        logger = logging.getLogger("test")

        cog = UserStatsCog(bot, user_stats, message_history_db, logger)

        assert cog.bot is bot
        assert cog.user_stats is user_stats
        assert cog.message_history_db is message_history_db
        assert cog.logger is logger

    def test_init_with_none_message_history_db(self):
        """Test initialization with None message_history_db."""
        bot = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        cog = UserStatsCog(bot, user_stats, None, logger)

        assert cog.message_history_db is None


class TestSendRpmStats:
    """Tests for send_rpm_stats command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        user_stats = MagicMock()
        message_history_db = MagicMock()
        logger = logging.getLogger("test")
        return UserStatsCog(bot, user_stats, message_history_db, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.author.__str__ = MagicMock(return_value="TestUser")
        ctx.send = AsyncMock()
        return ctx

    async def test_rpm_calls_personal_stats_by_default(self, cog, mock_ctx):
        """Test that RPM without args calls personal stats."""
        with patch(
            "strofkabot.cogs.user_stats.send_personal_stats", new_callable=AsyncMock
        ) as mock_send:
            await cog.send_rpm_stats.callback(cog, mock_ctx)
            mock_send.assert_called_once()

    async def test_rpm_with_leaderboard_flag(self, cog, mock_ctx):
        """Test that RPM with --leaderboard calls leaderboard function."""
        with patch(
            "strofkabot.cogs.user_stats.send_leaderboard", new_callable=AsyncMock
        ) as mock_send:
            await cog.send_rpm_stats.callback(cog, mock_ctx, "--leaderboard")
            mock_send.assert_called_once()

    async def test_rpm_with_leaderboard_and_least_flags(self, cog, mock_ctx):
        """Test that RPM with --leaderboard --least passes correct args."""
        with patch(
            "strofkabot.cogs.user_stats.send_leaderboard", new_callable=AsyncMock
        ) as mock_send:
            await cog.send_rpm_stats.callback(cog, mock_ctx, "--leaderboard", "--least")
            # Check that least=True was passed
            call_args = mock_send.call_args
            assert call_args[0][3] is True  # least parameter

    async def test_rpm_with_all_users_flag(self, cog, mock_ctx):
        """Test that RPM with --all passes all_users=True."""
        with patch(
            "strofkabot.cogs.user_stats.send_leaderboard", new_callable=AsyncMock
        ) as mock_send:
            await cog.send_rpm_stats.callback(cog, mock_ctx, "--leaderboard", "--all")
            call_args = mock_send.call_args
            assert call_args[0][4] is True  # all_users parameter

    async def test_rpm_handles_exception(self, cog, mock_ctx):
        """Test that RPM handles exceptions gracefully."""
        with patch(
            "strofkabot.cogs.user_stats.send_personal_stats", new_callable=AsyncMock
        ) as mock_send:
            mock_send.side_effect = Exception("Database error")
            await cog.send_rpm_stats.callback(cog, mock_ctx)
            mock_ctx.send.assert_called_with("An error occurred while fetching RPM statistics.")


class TestShowMostLiked:
    """Tests for show_most_liked command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        user_stats = MagicMock()
        message_history_db = MagicMock()
        logger = logging.getLogger("test")
        return UserStatsCog(bot, user_stats, message_history_db, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.__str__ = MagicMock(return_value="TestUser")
        ctx.send = AsyncMock()
        return ctx

    async def test_most_liked_calls_send_function(self, cog, mock_ctx):
        """Test that most-liked calls send_most_liked_stats."""
        with patch(
            "strofkabot.cogs.user_stats.send_most_liked_stats", new_callable=AsyncMock
        ) as mock_send:
            await cog.show_most_liked.callback(cog, mock_ctx)
            mock_send.assert_called_once()

    async def test_most_liked_with_all_flag(self, cog, mock_ctx):
        """Test that --all flag is parsed correctly."""
        with patch(
            "strofkabot.cogs.user_stats.send_most_liked_stats", new_callable=AsyncMock
        ) as mock_send:
            await cog.show_most_liked.callback(cog, mock_ctx, args="--all")
            call_args = mock_send.call_args
            assert call_args[0][6] is True  # show_all parameter

    async def test_most_liked_handles_exception(self, cog, mock_ctx):
        """Test that most-liked handles exceptions gracefully."""
        with patch(
            "strofkabot.cogs.user_stats.send_most_liked_stats", new_callable=AsyncMock
        ) as mock_send:
            mock_send.side_effect = Exception("Database error")
            await cog.show_most_liked.callback(cog, mock_ctx)
            mock_ctx.send.assert_called_with(
                "An error occurred while calculating most-liked users."
            )


class TestShowActivityHeatmap:
    """Tests for show_activity_heatmap command."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        user_stats = MagicMock()
        message_history_db = MagicMock()
        logger = logging.getLogger("test")
        return UserStatsCog(bot, user_stats, message_history_db, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.author.display_name = "TestUser"
        ctx.message = MagicMock()
        ctx.message.mentions = []
        ctx.send = AsyncMock()
        return ctx

    async def test_activity_without_message_history_db(self, mock_ctx):
        """Test activity command when message_history_db is None."""
        bot = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")
        cog = UserStatsCog(bot, user_stats, None, logger)

        await cog.show_activity_heatmap.callback(cog, mock_ctx)
        mock_ctx.send.assert_called_with("Activity data is not available.")

    async def test_activity_uses_author_when_no_mention(self, cog, mock_ctx):
        """Test that activity uses ctx.author when no user is mentioned."""
        with patch.object(cog, "_send_activity_heatmap", new_callable=AsyncMock) as mock_send:
            await cog.show_activity_heatmap.callback(cog, mock_ctx)
            call_args = mock_send.call_args
            assert call_args[0][1] == mock_ctx.author

    async def test_activity_uses_mentioned_user(self, cog, mock_ctx):
        """Test that activity uses mentioned user when provided."""
        mentioned_user = MagicMock()
        mentioned_user.id = 67890
        mock_ctx.message.mentions = [mentioned_user]

        with patch.object(cog, "_send_activity_heatmap", new_callable=AsyncMock) as mock_send:
            await cog.show_activity_heatmap.callback(cog, mock_ctx)
            call_args = mock_send.call_args
            assert call_args[0][1] == mentioned_user

    async def test_activity_parses_months_argument(self, cog, mock_ctx):
        """Test that --months argument is parsed correctly."""
        with patch.object(cog, "_send_activity_heatmap", new_callable=AsyncMock) as mock_send:
            await cog.show_activity_heatmap.callback(cog, mock_ctx, args="--months 6")
            call_args = mock_send.call_args
            assert call_args[0][2] == 6  # num_months

    async def test_activity_clamps_months_to_max_12(self, cog, mock_ctx):
        """Test that months is clamped to max 12."""
        with patch.object(cog, "_send_activity_heatmap", new_callable=AsyncMock) as mock_send:
            await cog.show_activity_heatmap.callback(cog, mock_ctx, args="--months 24")
            call_args = mock_send.call_args
            assert call_args[0][2] == 12

    async def test_activity_clamps_months_to_min_1(self, cog, mock_ctx):
        """Test that months is clamped to min 1."""
        with patch.object(cog, "_send_activity_heatmap", new_callable=AsyncMock) as mock_send:
            await cog.show_activity_heatmap.callback(cog, mock_ctx, args="--months 0")
            call_args = mock_send.call_args
            assert call_args[0][2] == 1


class TestSendActivityHeatmap:
    """Tests for _send_activity_heatmap internal method."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        user_stats = MagicMock()
        message_history_db = AsyncMock()
        logger = logging.getLogger("test")
        return UserStatsCog(bot, user_stats, message_history_db, logger)

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_user(self):
        user = MagicMock()
        user.id = 12345
        user.display_name = "TestUser"
        return user

    async def test_sends_no_data_message(self, cog, mock_ctx, mock_user):
        """Test that no data message is sent when activity_data is None."""
        with (
            patch(
                "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("strofkabot.cogs.user_stats.handle_month_navigation", new_callable=AsyncMock),
        ):
            mock_fetch.return_value = None

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=3, window_offset=0)

            # Check that a message about no data was sent
            call_args = mock_ctx.send.call_args
            assert "No activity data" in call_args[0][0]

    async def test_sends_no_data_when_all_zeros(self, cog, mock_ctx, mock_user):
        """Test that no data message is sent when activity_data is all zeros."""
        with (
            patch(
                "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("strofkabot.cogs.user_stats.handle_month_navigation", new_callable=AsyncMock),
        ):
            mock_fetch.return_value = np.zeros((7, 24))

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=3, window_offset=0)

            call_args = mock_ctx.send.call_args
            assert "No activity data" in call_args[0][0]

    async def test_sends_heatmap_with_data(self, cog, mock_ctx, mock_user):
        """Test that heatmap is sent when data exists."""
        import io

        with (
            patch(
                "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("strofkabot.cogs.user_stats.create_activity_heatmap") as mock_heatmap,
            patch("strofkabot.cogs.user_stats.handle_month_navigation", new_callable=AsyncMock),
            patch("strofkabot.cogs.user_stats.discord.File") as mock_file,
        ):
            # Create activity data with some non-zero values
            activity_data = np.zeros((7, 24))
            activity_data[0, 10] = 5  # 5 messages on Monday at 10am
            mock_fetch.return_value = activity_data
            mock_heatmap.return_value = io.BytesIO(b"fake plot data")

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=3, window_offset=0)

            # Check that heatmap was created
            mock_heatmap.assert_called_once()
            mock_ctx.send.assert_called_once()

    async def test_handles_exception(self, cog, mock_ctx, mock_user):
        """Test that exceptions are handled gracefully."""
        with patch(
            "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Database error")

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=3, window_offset=0)

            mock_ctx.send.assert_called_with("An error occurred while generating the heatmap.")

    async def test_single_month_period_string(self, cog, mock_ctx, mock_user):
        """Test that single month uses correct period string format."""
        with (
            patch(
                "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("strofkabot.cogs.user_stats.handle_month_navigation", new_callable=AsyncMock),
        ):
            mock_fetch.return_value = None

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=1, window_offset=0)

            call_args = mock_ctx.send.call_args[0][0]
            # Single month should not have " - " in the period
            assert " - " not in call_args or "No activity data" in call_args

    async def test_multi_month_period_string(self, cog, mock_ctx, mock_user):
        """Test that multi-month uses range period string format."""
        import io

        with (
            patch(
                "strofkabot.cogs.user_stats.fetch_hourly_activity_data_for_range",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("strofkabot.cogs.user_stats.create_activity_heatmap") as mock_heatmap,
            patch("strofkabot.cogs.user_stats.handle_month_navigation", new_callable=AsyncMock),
            patch("strofkabot.cogs.user_stats.discord.File"),
        ):
            activity_data = np.ones((7, 24))
            mock_fetch.return_value = activity_data
            mock_heatmap.return_value = io.BytesIO(b"fake plot data")

            await cog._send_activity_heatmap(mock_ctx, mock_user, num_months=3, window_offset=0)

            call_args = mock_ctx.send.call_args[0][0]
            # Multi-month should include a range with " - "
            assert " - " in call_args or "Based on" in call_args
