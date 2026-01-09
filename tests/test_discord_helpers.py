"""Tests for discord_helpers utility functions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strofkabot.utils.discord_helpers import (
    _get_trade_status,
    format_clusters_report,
    format_connections_report,
    format_monthly_trade_report,
    format_top_connections_by_user,
    format_yearly_trade_report,
    get_diversity_label,
    get_index_label,
    handle_month_navigation,
    send_leaderboard,
    send_most_liked_stats,
    send_personal_stats,
)


class TestGetDiversityLabel:
    """Tests for get_diversity_label function."""

    def test_very_high_diversity(self):
        """Test score >= 0.90 returns 'Very High'."""
        assert get_diversity_label(0.90) == "Very High"
        assert get_diversity_label(0.95) == "Very High"
        assert get_diversity_label(1.0) == "Very High"

    def test_high_diversity(self):
        """Test score >= 0.83 and < 0.90 returns 'High'."""
        assert get_diversity_label(0.83) == "High"
        assert get_diversity_label(0.85) == "High"
        assert get_diversity_label(0.89) == "High"

    def test_moderate_diversity(self):
        """Test score >= 0.75 and < 0.83 returns 'Moderate'."""
        assert get_diversity_label(0.75) == "Moderate"
        assert get_diversity_label(0.80) == "Moderate"
        assert get_diversity_label(0.82) == "Moderate"

    def test_low_diversity(self):
        """Test score < 0.75 returns 'Low'."""
        assert get_diversity_label(0.74) == "Low"
        assert get_diversity_label(0.50) == "Low"
        assert get_diversity_label(0.0) == "Low"


class TestGetIndexLabel:
    """Tests for get_index_label function."""

    def test_very_diverse(self):
        """Test index <= 12 returns 'Very Diverse'."""
        assert get_index_label(12) == "Very Diverse"
        assert get_index_label(10) == "Very Diverse"
        assert get_index_label(0) == "Very Diverse"

    def test_balanced(self):
        """Test index > 12 and <= 17 returns 'Balanced'."""
        assert get_index_label(13) == "Balanced"
        assert get_index_label(15) == "Balanced"
        assert get_index_label(17) == "Balanced"

    def test_concentrated(self):
        """Test index > 17 and <= 22 returns 'Concentrated'."""
        assert get_index_label(18) == "Concentrated"
        assert get_index_label(20) == "Concentrated"
        assert get_index_label(22) == "Concentrated"

    def test_echo_chamber(self):
        """Test index > 22 returns 'Echo Chamber'."""
        assert get_index_label(23) == "Echo Chamber"
        assert get_index_label(30) == "Echo Chamber"
        assert get_index_label(100) == "Echo Chamber"


class TestGetTradeStatus:
    """Tests for _get_trade_status function."""

    def test_surplus(self):
        """Test positive balance returns 'SURPLUS'."""
        assert _get_trade_status(1) == "SURPLUS"
        assert _get_trade_status(100) == "SURPLUS"

    def test_deficit(self):
        """Test negative balance returns 'DEFICIT'."""
        assert _get_trade_status(-1) == "DEFICIT"
        assert _get_trade_status(-100) == "DEFICIT"

    def test_neutral(self):
        """Test zero balance returns 'NEUTRAL'."""
        assert _get_trade_status(0) == "NEUTRAL"


class TestFormatYearlyTradeReport:
    """Tests for format_yearly_trade_report function."""

    def test_basic_report(self):
        """Test basic yearly trade report formatting."""
        trade_data = {
            "exports": [("Alice", 10), ("Bob", 5)],
            "imports": [("Charlie", 8), ("Diana", 3)],
            "total_given": 15,
            "total_received": 11,
            "trade_balance": 4,
        }
        result = format_yearly_trade_report(trade_data, "TestUser")

        assert "TestUser" in result
        assert "past 12 months" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result
        assert "Diana" in result
        assert "15" in result
        assert "11" in result
        assert "SURPLUS" in result

    def test_empty_exports(self):
        """Test report with no exports."""
        trade_data = {
            "exports": [],
            "imports": [("Charlie", 8)],
            "total_given": 0,
            "total_received": 8,
            "trade_balance": -8,
        }
        result = format_yearly_trade_report(trade_data, "TestUser")

        assert "No reactions given" in result
        assert "DEFICIT" in result

    def test_empty_imports(self):
        """Test report with no imports."""
        trade_data = {
            "exports": [("Alice", 10)],
            "imports": [],
            "total_given": 10,
            "total_received": 0,
            "trade_balance": 10,
        }
        result = format_yearly_trade_report(trade_data, "TestUser")

        assert "No reactions received" in result
        assert "SURPLUS" in result


class TestFormatMonthlyTradeReport:
    """Tests for format_monthly_trade_report function."""

    def test_basic_report(self):
        """Test basic monthly trade report formatting."""
        trade_data = {
            "exports": [("Alice", 10)],
            "imports": [("Bob", 5)],
            "total_given": 10,
            "total_received": 5,
            "trade_balance": 5,
        }
        result = format_monthly_trade_report(trade_data, "TestUser", "January 2025")

        assert "TestUser" in result
        assert "January 2025" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "SURPLUS" in result

    def test_no_activity(self):
        """Test report with no trading activity."""
        trade_data = {
            "exports": [],
            "imports": [],
            "total_given": 0,
            "total_received": 0,
            "trade_balance": 0,
        }
        result = format_monthly_trade_report(trade_data, "TestUser", "January 2025")

        assert "No trading activity" in result

    def test_no_exports_with_imports(self):
        """Test report with imports but no exports."""
        trade_data = {
            "exports": [],
            "imports": [("Bob", 5)],
            "total_given": 0,
            "total_received": 5,
            "trade_balance": -5,
        }
        result = format_monthly_trade_report(trade_data, "TestUser", "January 2025")

        assert "No reactions given" in result
        assert "Bob" in result

    def test_no_imports_with_exports(self):
        """Test report with exports but no imports."""
        trade_data = {
            "exports": [("Alice", 10)],
            "imports": [],
            "total_given": 10,
            "total_received": 0,
            "trade_balance": 10,
        }
        result = format_monthly_trade_report(trade_data, "TestUser", "January 2025")

        assert "Alice" in result
        assert "No reactions received" in result


class TestFormatConnectionsReport:
    """Tests for format_connections_report function."""

    def test_basic_report(self):
        """Test basic connections report formatting."""
        affinities = [
            ("Alice", "Bob", 0.85),
            ("Charlie", "Diana", 0.72),
            ("Eve", "Frank", 0.65),
        ]
        result = format_connections_report(affinities, "January 2025", limit=10)

        assert "Top 10 Mutual Relationships" in result
        assert "January 2025" in result
        assert "Alice <-> Bob" in result
        assert "0.85" in result

    def test_respects_limit(self):
        """Test that limit is respected."""
        affinities = [
            ("A", "B", 0.9),
            ("C", "D", 0.8),
            ("E", "F", 0.7),
            ("G", "H", 0.6),
            ("I", "J", 0.5),
        ]
        result = format_connections_report(affinities, "January 2025", limit=3)

        assert "Top 3" in result
        assert "A <-> B" in result
        assert "C <-> D" in result
        assert "E <-> F" in result
        assert "G <-> H" not in result


class TestFormatTopConnectionsByUser:
    """Tests for format_top_connections_by_user function."""

    def test_basic_report(self):
        """Test basic top connections by user report."""
        connections = [
            ("Alice", "Bob", 0.85),
            ("Charlie", "Diana", 0.72),
        ]
        result = format_top_connections_by_user(connections, "January 2025")

        assert "Top Connections by User" in result
        assert "January 2025" in result
        assert "Alice -> Bob" in result
        assert "30+ messages" in result


class TestFormatClustersReport:
    """Tests for format_clusters_report function."""

    def test_basic_report(self):
        """Test basic clusters report formatting."""
        sorted_groups = [
            (0, ["Alice", "Bob", "Charlie"]),
            (1, ["Diana", "Eve"]),
        ]
        result = format_clusters_report(sorted_groups, "Jan 2025 - Mar 2025", total_members=5)

        assert "Social Clusters" in result
        assert "Jan 2025 - Mar 2025" in result
        assert "2 groups" in result
        assert "5 members" in result
        assert "Group 1" in result
        assert "Group 2" in result

    def test_members_sorted_alphabetically(self):
        """Test that members within groups are sorted alphabetically."""
        sorted_groups = [
            (0, ["Charlie", "Alice", "Bob"]),
        ]
        result = format_clusters_report(sorted_groups, "Jan 2025", total_members=3)

        # Members should be sorted: Alice, Bob, Charlie
        assert "Alice, Bob, Charlie" in result


class TestHandleMonthNavigation:
    """Tests for handle_month_navigation function."""

    async def test_adds_reactions(self):
        """Test that left and right arrow reactions are added."""
        mock_bot = MagicMock()
        mock_message = AsyncMock()
        mock_callback = AsyncMock()

        # Make wait_for raise TimeoutError immediately
        mock_bot.wait_for = AsyncMock(side_effect=TimeoutError)

        await handle_month_navigation(mock_bot, mock_message, 0, mock_callback, timeout=0.1)

        # Check reactions were added
        mock_message.add_reaction.assert_any_call("⬅️")
        mock_message.add_reaction.assert_any_call("➡️")

    async def test_left_navigation_decrements_offset(self):
        """Test that left arrow decrements offset and calls callback."""
        mock_bot = MagicMock()
        mock_message = AsyncMock()
        mock_callback = AsyncMock()

        # Create mock reaction
        mock_reaction = MagicMock()
        mock_reaction.message.id = 123
        mock_reaction.emoji = "⬅️"
        mock_user = MagicMock()
        mock_user.bot = False

        mock_message.id = 123
        mock_bot.wait_for = AsyncMock(return_value=(mock_reaction, mock_user))

        await handle_month_navigation(mock_bot, mock_message, 0, mock_callback)

        mock_message.delete.assert_called_once()
        mock_callback.assert_called_once_with(-1)

    async def test_right_navigation_increments_offset(self):
        """Test that right arrow increments offset and calls callback."""
        mock_bot = MagicMock()
        mock_message = AsyncMock()
        mock_callback = AsyncMock()

        mock_reaction = MagicMock()
        mock_reaction.message.id = 123
        mock_reaction.emoji = "➡️"
        mock_user = MagicMock()
        mock_user.bot = False

        mock_message.id = 123
        mock_bot.wait_for = AsyncMock(return_value=(mock_reaction, mock_user))

        await handle_month_navigation(mock_bot, mock_message, 0, mock_callback)

        mock_message.delete.assert_called_once()
        mock_callback.assert_called_once_with(1)

    async def test_timeout_exits_loop(self):
        """Test that timeout exits the navigation loop."""
        mock_bot = MagicMock()
        mock_message = AsyncMock()
        mock_callback = AsyncMock()

        mock_bot.wait_for = AsyncMock(side_effect=TimeoutError)

        await handle_month_navigation(mock_bot, mock_message, 0, mock_callback, timeout=0.1)

        # Callback should not be called on timeout
        mock_callback.assert_not_called()
        # Message should not be deleted on timeout
        mock_message.delete.assert_not_called()


class TestSendLeaderboard:
    """Tests for send_leaderboard function."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_bot(self):
        bot = MagicMock()
        bot.wait_for = AsyncMock(side_effect=TimeoutError)
        return bot

    async def test_sends_leaderboard_message(self, mock_ctx, mock_bot):
        """Test that leaderboard message is sent."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_monthly_stats = AsyncMock(
            return_value=[
                {
                    "author_id": 111,
                    "username": "Alice",
                    "total_msgs": 50,
                    "total_reacts": 100,
                    "avg_reacts": 2.0,
                },
                {
                    "author_id": 222,
                    "username": "Bob",
                    "total_msgs": 40,
                    "total_reacts": 60,
                    "avg_reacts": 1.5,
                },
            ]
        )

        await send_leaderboard(
            mock_ctx,
            2025,
            1,
            least=False,
            all_users=False,
            user_stats=mock_user_stats,
            bot=mock_bot,
        )

        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "RPM Leaderboard" in call_args
        assert "Alice" in call_args
        assert "Bob" in call_args

    async def test_sends_no_stats_message(self, mock_ctx, mock_bot):
        """Test message when no stats available."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_monthly_stats = AsyncMock(return_value=[])

        await send_leaderboard(
            mock_ctx,
            2025,
            1,
            least=False,
            all_users=False,
            user_stats=mock_user_stats,
            bot=mock_bot,
        )

        mock_ctx.send.assert_called()
        call_args = mock_ctx.send.call_args[0][0]
        assert "No user statistics available" in call_args

    async def test_filters_users_with_low_messages(self, mock_ctx, mock_bot):
        """Test that users with fewer than 30 messages are filtered."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_monthly_stats = AsyncMock(
            return_value=[
                {
                    "author_id": 111,
                    "username": "Active",
                    "total_msgs": 50,
                    "total_reacts": 100,
                    "avg_reacts": 2.0,
                },
                {
                    "author_id": 222,
                    "username": "Inactive",
                    "total_msgs": 10,
                    "total_reacts": 20,
                    "avg_reacts": 2.0,
                },
            ]
        )

        await send_leaderboard(
            mock_ctx,
            2025,
            1,
            least=False,
            all_users=False,
            user_stats=mock_user_stats,
            bot=mock_bot,
        )

        call_args = mock_ctx.send.call_args[0][0]
        assert "Active" in call_args
        assert "Inactive" not in call_args

    async def test_least_flag_reverses_order(self, mock_ctx, mock_bot):
        """Test that --least flag reverses the sort order."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_monthly_stats = AsyncMock(
            return_value=[
                {
                    "author_id": 111,
                    "username": "High",
                    "total_msgs": 50,
                    "total_reacts": 100,
                    "avg_reacts": 2.0,
                },
                {
                    "author_id": 222,
                    "username": "Low",
                    "total_msgs": 50,
                    "total_reacts": 50,
                    "avg_reacts": 1.0,
                },
            ]
        )

        await send_leaderboard(
            mock_ctx, 2025, 1, least=True, all_users=False, user_stats=mock_user_stats, bot=mock_bot
        )

        call_args = mock_ctx.send.call_args[0][0]
        assert "Least Leaderboard" in call_args
        # Low RPM should appear first
        assert call_args.index("Low") < call_args.index("High")


class TestSendPersonalStats:
    """Tests for send_personal_stats function."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.author = MagicMock()
        ctx.author.id = 12345
        ctx.send = AsyncMock()
        return ctx

    async def test_sends_personal_stats(self, mock_ctx):
        """Test that personal stats are sent."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_user_monthly_stats = AsyncMock(
            return_value={"total_msgs": 50, "total_reacts": 100, "avg_reacts": 2.0}
        )

        await send_personal_stats(mock_ctx, 2025, 1, mock_user_stats)

        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "Your RPM Stats" in call_args
        assert "past 12 months" in call_args

    async def test_shows_zero_for_months_without_data(self, mock_ctx):
        """Test that months without data show zeros."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_user_monthly_stats = AsyncMock(return_value=None)

        await send_personal_stats(mock_ctx, 2025, 1, mock_user_stats)

        call_args = mock_ctx.send.call_args[0][0]
        assert "0.00" in call_args


class TestSendMostLikedStats:
    """Tests for send_most_liked_stats function."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_bot(self):
        bot = MagicMock()
        bot.wait_for = AsyncMock(side_effect=TimeoutError)
        return bot

    async def test_sends_most_liked_message(self, mock_ctx, mock_bot):
        """Test that most-liked message is sent."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_reaction_network_for_month = AsyncMock(
            return_value=[
                {"giver_username": "G1", "receiver_username": "Alice", "reaction_count": 10},
                {"giver_username": "G2", "receiver_username": "Alice", "reaction_count": 10},
                {"giver_username": "G3", "receiver_username": "Alice", "reaction_count": 10},
                {"giver_username": "G4", "receiver_username": "Alice", "reaction_count": 10},
                {"giver_username": "G5", "receiver_username": "Alice", "reaction_count": 10},
            ]
        )

        await send_most_liked_stats(mock_ctx, 2025, 1, 0, mock_user_stats, mock_bot, show_all=False)

        mock_ctx.send.assert_called_once()
        call_args = mock_ctx.send.call_args[0][0]
        assert "Most Liked Users" in call_args
        assert "Alice" in call_args

    async def test_no_data_message(self, mock_ctx, mock_bot):
        """Test message when no reaction data available."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_reaction_network_for_month = AsyncMock(return_value=[])

        await send_most_liked_stats(mock_ctx, 2025, 1, 0, mock_user_stats, mock_bot, show_all=False)

        call_args = mock_ctx.send.call_args[0][0]
        assert "No reaction data available" in call_args

    async def test_not_enough_users_message(self, mock_ctx, mock_bot):
        """Test message when not enough qualifying users."""
        mock_user_stats = MagicMock()
        # Only 2 unique reactors - not enough for default threshold of 5
        mock_user_stats.get_reaction_network_for_month = AsyncMock(
            return_value=[
                {"giver_username": "G1", "receiver_username": "Alice", "reaction_count": 10},
                {"giver_username": "G2", "receiver_username": "Alice", "reaction_count": 10},
            ]
        )

        await send_most_liked_stats(mock_ctx, 2025, 1, 0, mock_user_stats, mock_bot, show_all=False)

        call_args = mock_ctx.send.call_args[0][0]
        assert "Not enough users" in call_args

    async def test_show_all_flag(self, mock_ctx, mock_bot):
        """Test that show_all shows all users."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_reaction_network_for_month = AsyncMock(
            return_value=[
                {"giver_username": f"G{i}", "receiver_username": f"User{j}", "reaction_count": 10}
                for i in range(1, 11)  # 10 givers
                for j in range(1, 11)  # 10 receivers
            ]
        )

        await send_most_liked_stats(mock_ctx, 2025, 1, 0, mock_user_stats, mock_bot, show_all=True)

        call_args = mock_ctx.send.call_args[0][0]
        assert "(All)" in call_args
