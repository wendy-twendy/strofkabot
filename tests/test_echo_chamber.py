"""Tests for echo chamber analysis functions."""

import pytest

from strofkabot.utils import (
    calculate_echo_chamber_metrics,
    calculate_normalized_entropy,
    calculate_top_n_concentration,
)


class TestCalculateNormalizedEntropy:
    """Tests for the calculate_normalized_entropy function."""

    def test_perfect_concentration_single_item(self):
        """All reactions to one person = 0 entropy (maximum echo chamber)."""
        distribution = [100]
        result = calculate_normalized_entropy(distribution)
        assert result == 0.0

    def test_perfect_distribution_two_items(self):
        """Equal distribution between two = 1.0 (maximum diversity)."""
        distribution = [50, 50]
        result = calculate_normalized_entropy(distribution)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_perfect_distribution_three_items(self):
        """Equal distribution between three = 1.0."""
        distribution = [33, 33, 34]
        result = calculate_normalized_entropy(distribution)
        assert result == pytest.approx(1.0, abs=0.02)

    def test_skewed_distribution(self):
        """Skewed distribution should be between 0 and 1."""
        distribution = [80, 10, 10]
        result = calculate_normalized_entropy(distribution)
        assert 0 < result < 1

    def test_empty_distribution(self):
        """Empty distribution returns 0."""
        distribution = []
        result = calculate_normalized_entropy(distribution)
        assert result == 0.0

    def test_all_zeros(self):
        """All zeros returns 0."""
        distribution = [0, 0, 0]
        result = calculate_normalized_entropy(distribution)
        assert result == 0.0

    def test_single_zero(self):
        """Distribution with zeros still calculates correctly."""
        distribution = [50, 50, 0]
        result = calculate_normalized_entropy(distribution)
        # Only 2 non-zero items, so entropy is less than max for 3 items
        assert 0 < result < 1

    def test_highly_concentrated(self):
        """Very concentrated distribution has low entropy."""
        distribution = [95, 3, 1, 1]
        result = calculate_normalized_entropy(distribution)
        assert result < 0.5  # Should be quite low

    def test_large_even_distribution(self):
        """Large even distribution approaches 1.0."""
        distribution = [10] * 10  # 10 people, each with 10 reactions
        result = calculate_normalized_entropy(distribution)
        assert result == pytest.approx(1.0, abs=0.01)


class TestCalculateTopNConcentration:
    """Tests for the calculate_top_n_concentration function."""

    def test_top_3_simple(self):
        """Simple top 3 concentration test."""
        distribution = [50, 30, 10, 5, 5]
        result = calculate_top_n_concentration(distribution, n=3)
        # Top 3: 50 + 30 + 10 = 90 out of 100 = 90%
        assert result == pytest.approx(90.0, abs=0.1)

    def test_top_3_all_equal(self):
        """Equal distribution, top 3 of 5."""
        distribution = [20, 20, 20, 20, 20]
        result = calculate_top_n_concentration(distribution, n=3)
        # Top 3: 60 out of 100 = 60%
        assert result == pytest.approx(60.0, abs=0.1)

    def test_fewer_than_n_items(self):
        """When there are fewer items than n, return 100%."""
        distribution = [50, 50]
        result = calculate_top_n_concentration(distribution, n=3)
        assert result == pytest.approx(100.0, abs=0.1)

    def test_exactly_n_items(self):
        """When there are exactly n items, return 100%."""
        distribution = [40, 35, 25]
        result = calculate_top_n_concentration(distribution, n=3)
        assert result == pytest.approx(100.0, abs=0.1)

    def test_empty_distribution(self):
        """Empty distribution returns 0."""
        distribution = []
        result = calculate_top_n_concentration(distribution, n=3)
        assert result == 0.0

    def test_all_zeros(self):
        """All zeros returns 0."""
        distribution = [0, 0, 0, 0, 0]
        result = calculate_top_n_concentration(distribution, n=3)
        assert result == 0.0

    def test_single_item_dominates(self):
        """Single item gets all reactions."""
        distribution = [100, 0, 0, 0, 0]
        result = calculate_top_n_concentration(distribution, n=3)
        assert result == pytest.approx(100.0, abs=0.1)

    def test_unsorted_input(self):
        """Function should sort internally."""
        distribution = [5, 50, 10, 30, 5]  # Not sorted
        result = calculate_top_n_concentration(distribution, n=3)
        # Top 3 after sort: 50 + 30 + 10 = 90%
        assert result == pytest.approx(90.0, abs=0.1)

    def test_custom_n_value(self):
        """Test with n=5."""
        distribution = [30, 25, 20, 15, 5, 3, 2]
        result = calculate_top_n_concentration(distribution, n=5)
        # Top 5: 30 + 25 + 20 + 15 + 5 = 95 out of 100 = 95%
        assert result == pytest.approx(95.0, abs=0.1)


class TestCalculateEchoChamberMetrics:
    """Tests for the calculate_echo_chamber_metrics function."""

    def test_basic_metrics(self):
        """Test that all expected keys are returned."""
        outgoing = [(111, 50), (222, 30), (333, 20)]
        incoming = [(444, 40), (555, 35), (666, 25)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "outgoing_top3_pct" in result
        assert "incoming_top3_pct" in result
        assert "outgoing_diversity" in result
        assert "incoming_diversity" in result
        assert "echo_chamber_index" in result
        assert "interpretation" in result

    def test_high_echo_chamber(self):
        """Very concentrated patterns should have high echo chamber index."""
        outgoing = [(111, 95), (222, 5)]  # 95% to one person
        incoming = [(333, 90), (444, 10)]  # 90% from one person

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["outgoing_top3_pct"] == pytest.approx(100.0, abs=0.1)
        assert result["incoming_top3_pct"] == pytest.approx(100.0, abs=0.1)
        # With only 2 people, even 95/5 split has some entropy
        # Index should still be elevated (concentrated)
        assert result["echo_chamber_index"] > 50

    def test_low_echo_chamber(self):
        """Evenly distributed patterns should have low echo chamber index."""
        # 10 people, each with ~10 reactions
        outgoing = [(i, 10) for i in range(10)]
        incoming = [(i + 100, 10) for i in range(10)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["outgoing_diversity"] > 0.9  # High diversity
        assert result["incoming_diversity"] > 0.9
        assert result["echo_chamber_index"] < 30  # Low index

    def test_asymmetric_patterns(self):
        """Different outgoing and incoming patterns."""
        outgoing = [(111, 90), (222, 10)]  # Concentrated giving
        incoming = [(i, 10) for i in range(10)]  # Diverse receiving

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["outgoing_diversity"] < result["incoming_diversity"]

    def test_empty_outgoing(self):
        """Handle empty outgoing gracefully."""
        outgoing = []
        incoming = [(111, 50), (222, 50)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["outgoing_top3_pct"] == 0.0
        assert result["outgoing_diversity"] == 0.0

    def test_empty_incoming(self):
        """Handle empty incoming gracefully."""
        outgoing = [(111, 50), (222, 50)]
        incoming = []

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert result["incoming_top3_pct"] == 0.0
        assert result["incoming_diversity"] == 0.0

    def test_both_empty(self):
        """Handle both empty gracefully."""
        result = calculate_echo_chamber_metrics([], [])

        assert result["outgoing_top3_pct"] == 0.0
        assert result["incoming_top3_pct"] == 0.0
        assert result["echo_chamber_index"] == 100  # No data = maximum echo chamber

    def test_interpretation_very_diverse(self):
        """Index 0-25 should say diverse."""
        # Very even distribution
        outgoing = [(i, 5) for i in range(20)]
        incoming = [(i + 100, 5) for i in range(20)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "diverse" in result["interpretation"].lower()

    def test_interpretation_echo_chamber(self):
        """Index 76-100 should say echo chamber."""
        outgoing = [(111, 100)]  # All to one person
        incoming = [(222, 100)]  # All from one person

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "echo chamber" in result["interpretation"].lower()

    def test_top_partners_returned(self):
        """Top partners should be returned with names and percentages."""
        outgoing = [(111, 50), (222, 30), (333, 20)]
        incoming = [(444, 60), (555, 40)]

        result = calculate_echo_chamber_metrics(outgoing, incoming)

        assert "outgoing_top" in result
        assert "incoming_top" in result
        assert len(result["outgoing_top"]) <= 3
        assert len(result["incoming_top"]) <= 3
        # Each should be (user_id, count, percentage)
        assert result["outgoing_top"][0] == (111, 50, pytest.approx(50.0, abs=0.1))
