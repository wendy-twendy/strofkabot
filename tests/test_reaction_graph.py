"""Tests for reaction graph functions."""

from datetime import datetime
from unittest.mock import patch

import networkx as nx
import pytest

from strofkabot.utils.reaction_graph import (
    build_affinity_graph,
    build_reaction_graph,
    compute_all_affinities,
    compute_community_layout,
    compute_node_activity,
    detect_communities,
    format_period_string,
    get_rolling_start_month,
    mutual_affinity,
)


class TestBuildReactionGraph:
    """Tests for build_reaction_graph function."""

    def test_builds_directed_graph(self):
        """Test that function returns a directed graph."""
        data = [("Alice", "Bob", 10)]
        G = build_reaction_graph(data)
        assert isinstance(G, nx.DiGraph)

    def test_creates_edge_with_weight(self):
        """Test that edges are created with correct weights."""
        data = [("Alice", "Bob", 10), ("Bob", "Alice", 5)]
        G = build_reaction_graph(data)

        assert G.has_edge("Alice", "Bob")
        assert G.has_edge("Bob", "Alice")
        assert G["Alice"]["Bob"]["weight"] == 10
        assert G["Bob"]["Alice"]["weight"] == 5

    def test_aggregates_duplicate_edges(self):
        """Test that duplicate edges are aggregated."""
        data = [
            ("Alice", "Bob", 10),
            ("Alice", "Bob", 5),
        ]
        G = build_reaction_graph(data)

        assert G["Alice"]["Bob"]["weight"] == 15

    def test_empty_data(self):
        """Test with empty data returns empty graph."""
        G = build_reaction_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestMutualAffinity:
    """Tests for mutual_affinity function."""

    def test_basic_calculation(self):
        """Test mutual affinity calculation with known values."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("A", "C", weight=10)  # A gives 50% to B, 50% to C
        G.add_edge("B", "A", weight=20)
        G.add_edge("B", "C", weight=20)  # B gives 50% to A, 50% to C

        # share_A_to_B = 10/20 = 0.5
        # share_B_to_A = 20/40 = 0.5
        # affinity = sqrt(0.5 * 0.5) = 0.5
        affinity = mutual_affinity(G, "A", "B")
        assert affinity == pytest.approx(0.5, abs=0.01)

    def test_unidirectional_returns_zero(self):
        """Test that unidirectional edges return zero affinity."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        # B does not react to A

        affinity = mutual_affinity(G, "A", "B")
        assert affinity == 0.0

    def test_no_outgoing_edges_returns_zero(self):
        """Test that users with no outgoing edges return zero."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_node("C")  # No outgoing edges

        affinity = mutual_affinity(G, "A", "C")
        assert affinity == 0.0

    def test_high_affinity_pair(self):
        """Test pair that predominantly reacts to each other."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=90)
        G.add_edge("A", "C", weight=10)  # A gives 90% to B
        G.add_edge("B", "A", weight=80)
        G.add_edge("B", "C", weight=20)  # B gives 80% to A

        # share_A_to_B = 0.9
        # share_B_to_A = 0.8
        # affinity = sqrt(0.9 * 0.8) = 0.849
        affinity = mutual_affinity(G, "A", "B")
        assert affinity == pytest.approx(0.849, abs=0.01)


class TestComputeAllAffinities:
    """Tests for compute_all_affinities function."""

    def test_returns_sorted_list(self):
        """Test that results are sorted by affinity descending."""
        G = nx.DiGraph()
        # High affinity pair
        G.add_edge("A", "B", weight=90)
        G.add_edge("B", "A", weight=90)
        # Low affinity pair
        G.add_edge("C", "D", weight=10)
        G.add_edge("D", "C", weight=10)
        G.add_edge("C", "A", weight=90)
        G.add_edge("D", "B", weight=90)

        affinities = compute_all_affinities(G)

        # Should be sorted descending by affinity
        assert affinities[0][2] >= affinities[-1][2]

    def test_excludes_unidirectional_edges(self):
        """Test that unidirectional edges are not included."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)  # Unidirectional
        G.add_edge("C", "D", weight=10)
        G.add_edge("D", "C", weight=10)  # Bidirectional

        affinities = compute_all_affinities(G)

        pairs = [(a, b) for a, b, _ in affinities]
        assert ("A", "B") not in pairs and ("B", "A") not in pairs
        assert ("C", "D") in pairs or ("D", "C") in pairs

    def test_no_duplicate_pairs(self):
        """Test that (A,B) and (B,A) are not both in results."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)

        affinities = compute_all_affinities(G)

        # Should only have one entry for the pair
        assert len(affinities) == 1


class TestBuildAffinityGraph:
    """Tests for build_affinity_graph function."""

    def test_creates_undirected_graph(self):
        """Test that result is an undirected graph."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        affinities = [("A", "B", 0.5)]

        affinity_G = build_affinity_graph(G, affinities, min_affinity=0.0)

        assert isinstance(affinity_G, nx.Graph)

    def test_filters_by_min_affinity(self):
        """Test that edges below threshold are excluded."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        affinities = [("A", "B", 0.01), ("C", "D", 0.1)]

        affinity_G = build_affinity_graph(G, affinities, min_affinity=0.05)

        # Only C-D should be included (0.1 >= 0.05)
        assert not affinity_G.has_edge("A", "B")

    def test_removes_isolated_nodes_by_default(self):
        """Test that isolated nodes are removed by default."""
        G = nx.DiGraph()
        G.add_node("Isolated")
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        affinities = [("A", "B", 0.5)]

        affinity_G = build_affinity_graph(G, affinities, min_affinity=0.0, remove_isolated=True)

        assert "Isolated" not in affinity_G.nodes()

    def test_keeps_isolated_when_specified(self):
        """Test that isolated nodes are kept when remove_isolated=False."""
        G = nx.DiGraph()
        G.add_node("Isolated")
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        affinities = [("A", "B", 0.5)]

        affinity_G = build_affinity_graph(G, affinities, min_affinity=0.0, remove_isolated=False)

        assert "Isolated" in affinity_G.nodes()


class TestDetectCommunities:
    """Tests for detect_communities function."""

    def test_returns_dict(self):
        """Test that result is a dictionary."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)

        communities = detect_communities(G)

        assert isinstance(communities, dict)

    def test_all_nodes_assigned(self):
        """Test that all nodes are assigned to a community."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        G.add_edge("C", "D", weight=10)
        G.add_edge("D", "C", weight=10)

        communities = detect_communities(G)

        assert set(communities.keys()) == {"A", "B", "C", "D"}

    def test_deterministic_with_seed(self):
        """Test that same seed produces same communities."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)
        G.add_edge("C", "D", weight=10)
        G.add_edge("D", "C", weight=10)
        G.add_edge("A", "C", weight=1)
        G.add_edge("C", "A", weight=1)

        communities1 = detect_communities(G, seed=42)
        communities2 = detect_communities(G, seed=42)

        assert communities1 == communities2

    def test_community_ids_are_integers(self):
        """Test that community IDs are integers."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("B", "A", weight=10)

        communities = detect_communities(G)

        for comm_id in communities.values():
            assert isinstance(comm_id, int)


class TestComputeNodeActivity:
    """Tests for compute_node_activity function."""

    def test_sums_in_and_out(self):
        """Test that activity is sum of in and out weights."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=10)
        G.add_edge("C", "A", weight=5)

        activity = compute_node_activity(G)

        # A: gives 10, receives 5 = 15
        assert activity["A"] == 15
        # B: gives 0, receives 10 = 10
        assert activity["B"] == 10
        # C: gives 5, receives 0 = 5
        assert activity["C"] == 5

    def test_empty_graph(self):
        """Test with empty graph returns empty dict."""
        G = nx.DiGraph()

        activity = compute_node_activity(G)

        assert activity == {}


class TestComputeCommunityLayout:
    """Tests for compute_community_layout function."""

    def test_returns_positions_for_all_nodes(self):
        """Test that all nodes have positions."""
        G = nx.Graph()
        G.add_edge("A", "B", affinity=0.5)

        positions = compute_community_layout(G)

        assert "A" in positions
        assert "B" in positions

    def test_positions_have_two_coordinates(self):
        """Test that positions have (x, y) coordinates."""
        G = nx.Graph()
        G.add_edge("A", "B", affinity=0.5)

        positions = compute_community_layout(G)

        for pos in positions.values():
            # networkx may return numpy arrays or tuples
            assert len(pos) == 2

    def test_includes_isolated_nodes(self):
        """Test that isolated nodes are positioned."""
        G = nx.Graph()
        G.add_edge("A", "B", affinity=0.5)

        isolated = ["C", "D"]
        positions = compute_community_layout(G, isolated_nodes=isolated)

        assert "C" in positions
        assert "D" in positions


class TestGetRollingStartMonth:
    """Tests for get_rolling_start_month function."""

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_one_month_back(self, mock_datetime):
        """Test getting 1 month (current month only)."""
        mock_datetime.now.return_value = datetime(2024, 3, 15)

        year, month = get_rolling_start_month(1)

        # 1 month including current = just March
        assert year == 2024
        assert month == 3

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_three_months_back(self, mock_datetime):
        """Test getting 3 months including current."""
        mock_datetime.now.return_value = datetime(2024, 5, 10)

        year, month = get_rolling_start_month(3)

        # 3 months including May: May, April, March
        # Start month is March
        assert year == 2024
        assert month == 3

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_crosses_year_boundary(self, mock_datetime):
        """Test when rolling back crosses year boundary."""
        mock_datetime.now.return_value = datetime(2024, 2, 15)

        year, month = get_rolling_start_month(3)

        # 3 months including Feb: Feb, Jan, Dec
        # Start month is December of previous year
        assert year == 2023
        assert month == 12

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_january_one_month(self, mock_datetime):
        """Test 1 month from January (current month only)."""
        mock_datetime.now.return_value = datetime(2024, 1, 15)

        year, month = get_rolling_start_month(1)

        # 1 month including current = just January
        assert year == 2024
        assert month == 1

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_large_offset_multiple_years(self, mock_datetime):
        """Test large offset spanning multiple years."""
        mock_datetime.now.return_value = datetime(2024, 3, 15)

        year, month = get_rolling_start_month(14)

        # 14 months including March 2024:
        # Go back 13 months from March: Feb 2023
        assert year == 2023
        assert month == 2


class TestFormatPeriodString:
    """Tests for format_period_string function."""

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_single_month(self, mock_datetime):
        """Test period string for single month."""
        mock_datetime.now.return_value = datetime(2024, 3, 15)

        result = format_period_string(2024, 2, 1)

        assert "Feb 2024" in result

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_multiple_months(self, mock_datetime):
        """Test period string for multiple months."""
        mock_datetime.now.return_value = datetime(2024, 5, 10)

        result = format_period_string(2024, 3, 3)

        # Should show range: Mar 2024 - May 2024 (current month included)
        assert "Mar 2024" in result
        assert "May 2024" in result

    @patch("strofkabot.utils.reaction_graph.datetime")
    def test_crosses_year_boundary(self, mock_datetime):
        """Test period string crossing year boundary."""
        mock_datetime.now.return_value = datetime(2024, 2, 15)

        result = format_period_string(2023, 12, 3)

        # Should show range: Dec 2023 - Feb 2024 (current month included)
        assert "Dec 2023" in result
        assert "Feb 2024" in result
