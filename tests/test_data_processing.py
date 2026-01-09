"""Tests for data_processing utility functions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strofkabot.utils.data_processing import (
    calculate_echo_chamber_metrics,
    calculate_normalized_entropy,
    calculate_top_n_concentration,
    fetch_hourly_activity_data,
    fetch_hourly_activity_data_for_month,
    fetch_hourly_activity_data_for_range,
    get_reaction_trade_data_for_month,
)


class TestFetchHourlyActivityData:
    """Tests for fetch_hourly_activity_data function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(self):
        """Test that None is returned when no data exists."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user = AsyncMock(return_value=[])

        result = await fetch_hourly_activity_data(mock_db, user_id=12345)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_7x24_matrix(self):
        """Test that result is a 7x24 matrix."""
        mock_db = MagicMock()
        # SQLite returns (day_of_week, hour, count)
        # day 1 = Monday in SQLite, hour 10, count 5
        mock_db.get_hourly_activity_by_user = AsyncMock(return_value=[(1, 10, 5), (2, 14, 3)])

        result = await fetch_hourly_activity_data(mock_db, user_id=12345)

        assert result.shape == (7, 24)

    @pytest.mark.asyncio
    async def test_maps_sqlite_days_correctly(self):
        """Test that SQLite day mapping is correct (0=Sun -> 6, 1=Mon -> 0)."""
        mock_db = MagicMock()
        # SQLite: 0=Sunday, 1=Monday
        mock_db.get_hourly_activity_by_user = AsyncMock(
            return_value=[
                (0, 12, 10),  # Sunday hour 12 -> row 6
                (1, 12, 20),  # Monday hour 12 -> row 0
            ]
        )

        result = await fetch_hourly_activity_data(mock_db, user_id=12345)

        # Sunday (SQLite 0) should be at row 6 (our index)
        assert result[6, 12] == 10
        # Monday (SQLite 1) should be at row 0
        assert result[0, 12] == 20

    @pytest.mark.asyncio
    async def test_passes_timezone_offset(self):
        """Test that timezone offset is passed to database."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user = AsyncMock(return_value=[])

        await fetch_hourly_activity_data(mock_db, user_id=12345, timezone_offset=2)

        mock_db.get_hourly_activity_by_user.assert_called_once_with(12345, 2)


class TestFetchHourlyActivityDataForMonth:
    """Tests for fetch_hourly_activity_data_for_month function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(self):
        """Test that None is returned when no data exists."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(return_value=[])

        result = await fetch_hourly_activity_data_for_month(
            mock_db, user_id=12345, year=2024, month=1
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_7x24_matrix(self):
        """Test that result is a 7x24 matrix."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(return_value=[(1, 10, 5)])

        result = await fetch_hourly_activity_data_for_month(
            mock_db, user_id=12345, year=2024, month=1
        )

        assert result.shape == (7, 24)

    @pytest.mark.asyncio
    async def test_passes_correct_parameters(self):
        """Test that all parameters are passed correctly."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(return_value=[])

        await fetch_hourly_activity_data_for_month(
            mock_db, user_id=12345, year=2024, month=6, timezone_offset=3
        )

        mock_db.get_hourly_activity_by_user_for_month.assert_called_once_with(12345, 2024, 6, 3)


class TestFetchHourlyActivityDataForRange:
    """Tests for fetch_hourly_activity_data_for_range function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data_in_any_month(self):
        """Test that None is returned when no data exists across range."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(return_value=[])

        result = await fetch_hourly_activity_data_for_range(
            mock_db, user_id=12345, end_year=2024, end_month=3, num_months=3
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregates_data_from_multiple_months(self):
        """Test that data from multiple months is aggregated."""
        mock_db = MagicMock()
        # Return data for each month call
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(
            side_effect=[
                [(1, 10, 5)],  # Month 1: 5 messages
                [(1, 10, 3)],  # Month 2: 3 messages
                [(1, 10, 2)],  # Month 3: 2 messages
            ]
        )

        result = await fetch_hourly_activity_data_for_range(
            mock_db, user_id=12345, end_year=2024, end_month=3, num_months=3
        )

        # Should sum to 10
        assert result[0, 10] == 10

    @pytest.mark.asyncio
    async def test_returns_data_when_some_months_empty(self):
        """Test that data is returned even if some months are empty."""
        mock_db = MagicMock()
        mock_db.get_hourly_activity_by_user_for_month = AsyncMock(
            side_effect=[
                [(1, 10, 5)],  # Month 1: has data
                [],  # Month 2: no data
                [(1, 10, 3)],  # Month 3: has data
            ]
        )

        result = await fetch_hourly_activity_data_for_range(
            mock_db, user_id=12345, end_year=2024, end_month=3, num_months=3
        )

        assert result is not None
        assert result[0, 10] == 8


class TestGetReactionTradeDataForMonth:
    """Tests for get_reaction_trade_data_for_month function."""

    @pytest.mark.asyncio
    async def test_resolves_member_names(self):
        """Test that guild member names are resolved."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_reaction_trade_data_for_month = AsyncMock(
            return_value={
                "exports": [(111, 10), (222, 5)],
                "imports": [(333, 8)],
                "total_given": 15,
                "total_received": 8,
                "trade_balance": 7,
            }
        )

        member1 = MagicMock()
        member1.display_name = "Alice"
        member2 = MagicMock()
        member2.display_name = "Bob"
        member3 = MagicMock()
        member3.display_name = "Charlie"

        mock_guild = MagicMock()
        mock_guild.get_member = MagicMock(
            side_effect=lambda uid: {111: member1, 222: member2, 333: member3}.get(uid)
        )

        result = await get_reaction_trade_data_for_month(
            mock_user_stats, user_id=12345, year=2024, month=1, guild=mock_guild
        )

        assert result["exports"] == [("Alice", 10), ("Bob", 5)]
        assert result["imports"] == [("Charlie", 8)]

    @pytest.mark.asyncio
    async def test_falls_back_to_user_id_for_unknown_members(self):
        """Test fallback to 'User {id}' format for unknown members."""
        mock_user_stats = MagicMock()
        mock_user_stats.get_reaction_trade_data_for_month = AsyncMock(
            return_value={
                "exports": [(999, 10)],
                "imports": [(888, 5)],
                "total_given": 10,
                "total_received": 5,
                "trade_balance": 5,
            }
        )

        mock_guild = MagicMock()
        mock_guild.get_member = MagicMock(return_value=None)

        result = await get_reaction_trade_data_for_month(
            mock_user_stats, user_id=12345, year=2024, month=1, guild=mock_guild
        )

        assert result["exports"] == [("User 999", 10)]
        assert result["imports"] == [("User 888", 5)]


class TestCalculateNormalizedEntropy:
    """Tests for calculate_normalized_entropy function."""

    def test_returns_zero_for_empty_distribution(self):
        """Test that empty distribution returns 0."""
        result = calculate_normalized_entropy([])
        assert result == 0.0

    def test_returns_zero_for_single_element(self):
        """Test that single element returns 0."""
        result = calculate_normalized_entropy([100])
        assert result == 0.0

    def test_returns_zero_for_all_zeros(self):
        """Test that all zeros returns 0."""
        result = calculate_normalized_entropy([0, 0, 0])
        assert result == 0.0

    def test_returns_one_for_uniform_distribution(self):
        """Test that uniform distribution returns 1."""
        result = calculate_normalized_entropy([10, 10, 10, 10])
        assert result == pytest.approx(1.0, abs=0.001)

    def test_returns_low_for_concentrated_distribution(self):
        """Test that concentrated distribution returns low value."""
        # 90% to one, 10% split
        result = calculate_normalized_entropy([90, 5, 5])
        assert result < 0.5

    def test_returns_high_for_even_distribution(self):
        """Test that even distribution returns high value."""
        result = calculate_normalized_entropy([25, 25, 25, 25])
        assert result > 0.9


class TestCalculateTopNConcentration:
    """Tests for calculate_top_n_concentration function."""

    def test_returns_zero_for_empty_distribution(self):
        """Test that empty distribution returns 0."""
        result = calculate_top_n_concentration([])
        assert result == 0.0

    def test_returns_100_when_all_in_top_n(self):
        """Test that 100% is returned when all is in top N."""
        result = calculate_top_n_concentration([10, 5, 3], n=3)
        assert result == 100.0

    def test_calculates_correct_percentage(self):
        """Test correct percentage calculation."""
        # Top 2 of [50, 30, 20] = 80/100 = 80%
        result = calculate_top_n_concentration([50, 30, 20], n=2)
        assert result == 80.0

    def test_handles_n_greater_than_length(self):
        """Test when N is greater than distribution length."""
        result = calculate_top_n_concentration([10, 5], n=5)
        assert result == 100.0


class TestCalculateEchoChamberMetrics:
    """Tests for calculate_echo_chamber_metrics function."""

    def test_returns_100_index_for_no_data(self):
        """Test that no data returns max echo chamber index."""
        result = calculate_echo_chamber_metrics([], [])
        assert result["echo_chamber_index"] == 100

    def test_calculates_outgoing_metrics(self):
        """Test that outgoing metrics are calculated."""
        outgoing = [(1, 50), (2, 30), (3, 20)]
        incoming = []

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "outgoing_top3_pct" in result
        assert "outgoing_diversity" in result
        assert len(result["outgoing_top"]) == 3

    def test_calculates_incoming_metrics(self):
        """Test that incoming metrics are calculated."""
        outgoing = []
        incoming = [(1, 40), (2, 35), (3, 25)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "incoming_top3_pct" in result
        assert "incoming_diversity" in result
        assert len(result["incoming_top"]) == 3

    def test_includes_interpretation(self):
        """Test that interpretation string is included."""
        result = calculate_echo_chamber_metrics([(1, 10)], [(2, 10)])
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)

    def test_top_partners_sorted_by_count(self):
        """Test that top partners are sorted by count descending."""
        outgoing = [(1, 10), (2, 50), (3, 30)]
        incoming = []

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        # Should be sorted: 50, 30, 10
        assert result["outgoing_top"][0][0] == 2  # user_id with 50
        assert result["outgoing_top"][1][0] == 3  # user_id with 30
        assert result["outgoing_top"][2][0] == 1  # user_id with 10

    def test_calculates_percentages_for_top_partners(self):
        """Test that percentages are calculated for top partners."""
        outgoing = [(1, 50), (2, 30), (3, 20)]  # Total 100
        incoming = []

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        # First user has 50/100 = 50%
        assert result["outgoing_top"][0][2] == pytest.approx(50.0, abs=0.1)

    def test_diverse_pattern_has_low_index(self):
        """Test that diverse patterns have low echo chamber index."""
        # Even distribution across 10 users
        outgoing = [(i, 10) for i in range(10)]
        incoming = [(i, 10) for i in range(10)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["echo_chamber_index"] < 20

    def test_concentrated_pattern_has_high_index(self):
        """Test that concentrated patterns have high echo chamber index."""
        # 95% to one user
        outgoing = [(1, 95), (2, 3), (3, 2)]
        incoming = [(1, 90), (2, 7), (3, 3)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["echo_chamber_index"] > 50

    def test_interpretation_varies_by_index(self):
        """Test that interpretation changes based on index."""
        # Test different ranges
        diverse = calculate_echo_chamber_metrics(
            [(i, 10) for i in range(20)],
            [(i, 10) for i in range(20)],
        )
        assert "diverse" in diverse["interpretation"].lower()

        concentrated = calculate_echo_chamber_metrics(
            [(1, 95), (2, 3), (3, 2)],
            [(1, 95), (2, 3), (3, 2)],
        )
        # Could be "concentrated" or "echo chamber"
        assert (
            "concentrated" in concentrated["interpretation"].lower()
            or "echo" in concentrated["interpretation"].lower()
        )
