"""
Tests for visualization functions.
Focuses on verifying that functions return valid BytesIO objects with PNG data.
"""

import io

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from strofkabot.utils import (
    calculate_monthly_inflation,
    calculate_yearly_inflation,
    create_gdp_plot,
    create_hdi_plot,
    create_monthly_inflation_plot,
    create_yearly_inflation_plot,
    generate_cluster_plot,
    generate_reaction_matrix_plot,
)


class TestInflationPlots:
    """Tests for inflation visualization functions."""

    def test_create_monthly_inflation_plot_returns_bytesio(self, sample_monthly_data):
        """Test that monthly inflation plot returns BytesIO."""
        inflation_data = calculate_monthly_inflation(sample_monthly_data)

        if not inflation_data:
            pytest.skip("Not enough data for inflation calculation")

        result = create_monthly_inflation_plot(inflation_data)

        assert isinstance(result, io.BytesIO)
        # Verify it's a PNG by checking magic bytes
        result.seek(0)
        header = result.read(8)
        assert header[:4] == b"\x89PNG"

    def test_create_yearly_inflation_plot_returns_bytesio(self, sample_yearly_data):
        """Test that yearly inflation plot returns BytesIO."""
        inflation_data = calculate_yearly_inflation(sample_yearly_data)

        if not inflation_data:
            pytest.skip("Not enough data for inflation calculation")

        result = create_yearly_inflation_plot(inflation_data)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    def test_monthly_plot_position_is_zero(self, sample_monthly_data):
        """Test that the BytesIO position is at start (ready to read)."""
        inflation_data = calculate_monthly_inflation(sample_monthly_data)

        if not inflation_data:
            pytest.skip("Not enough data")

        result = create_monthly_inflation_plot(inflation_data)

        assert result.tell() == 0  # Position should be at start


class TestGDPPlot:
    """Tests for GDP plot visualization."""

    def test_create_gdp_plot_returns_bytesio(self, sample_gdp_data):
        """Test that GDP plot returns BytesIO with PNG data."""
        result = create_gdp_plot(sample_gdp_data)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    def test_gdp_plot_handles_single_datapoint(self):
        """Test GDP plot with single data point."""
        data = [{"year": 2024, "month": 1, "total_messages": 1000}]

        result = create_gdp_plot(data)

        assert isinstance(result, io.BytesIO)

    def test_gdp_plot_ready_for_reading(self, sample_gdp_data):
        """Test that GDP plot BytesIO is seeked to start."""
        result = create_gdp_plot(sample_gdp_data)

        assert result.tell() == 0

    def test_gdp_plot_handles_large_dataset(self):
        """Test GDP plot with 84+ months of data (7 years)."""
        # Generate 84 months of data
        data = []
        for year in range(2018, 2025):
            for month in range(1, 13):
                if year == 2024 and month > 12:
                    break
                data.append(
                    {
                        "year": year,
                        "month": month,
                        "total_messages": 1000 + (year - 2018) * 100 + month * 10,
                    }
                )

        result = create_gdp_plot(data)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"


class TestHDIPlot:
    """Tests for HDI plot visualization."""

    def test_create_hdi_plot_returns_bytesio(self, sample_hdi_data):
        """Test that HDI plot returns BytesIO with PNG data."""
        result = create_hdi_plot(sample_hdi_data)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    def test_hdi_plot_ready_for_reading(self, sample_hdi_data):
        """Test that HDI plot BytesIO is seeked to start."""
        result = create_hdi_plot(sample_hdi_data)

        assert result.tell() == 0

    def test_hdi_plot_single_datapoint(self):
        """Test HDI plot with single data point."""
        data = [
            {"year": 2024, "month": 1, "quality_count": 100, "total_count": 1000, "hdi_ratio": 0.1}
        ]

        result = create_hdi_plot(data)

        assert isinstance(result, io.BytesIO)


class TestReactionMatrixPlot:
    """Tests for reaction matrix heatmap visualization."""

    def test_generate_reaction_matrix_plot_returns_bytesio(self):
        """Test that reaction matrix plot returns BytesIO."""
        data = np.array([[0, 50, 30], [40, 0, 60], [20, 30, 0]])
        labels = ["User1", "User2", "User3"]
        fig_size = 24

        result = generate_reaction_matrix_plot(data, labels, fig_size)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    def test_reaction_matrix_handles_larger_data(self):
        """Test reaction matrix with larger dataset."""
        size = 10
        data = np.random.rand(size, size) * 100
        labels = [f"User{i}" for i in range(size)]
        fig_size = 24

        result = generate_reaction_matrix_plot(data, labels, fig_size)

        assert isinstance(result, io.BytesIO)

    def test_reaction_matrix_ready_for_reading(self):
        """Test that reaction matrix plot is seeked to start."""
        data = np.array([[0, 50], [50, 0]])
        labels = ["A", "B"]

        result = generate_reaction_matrix_plot(data, labels, 24)

        assert result.tell() == 0


class TestClusterPlot:
    """Tests for clustering visualization."""

    @pytest.mark.xfail(reason="Known seaborn compatibility issue with ListedColormap")
    def test_generate_cluster_plot_returns_bytesio(self):
        """Test that cluster plot returns BytesIO."""
        data = np.array([[10, 20], [15, 25], [12, 22], [50, 60], [55, 65], [52, 62]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        member_names = ["A", "B", "C", "D", "E", "F"]

        result = generate_cluster_plot(data, labels, member_names)

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    @pytest.mark.xfail(reason="Known seaborn compatibility issue with ListedColormap")
    def test_cluster_plot_ready_for_reading(self):
        """Test that cluster plot is seeked to start."""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        labels = np.array([0, 0, 1, 1])
        names = ["A", "B", "C", "D"]

        result = generate_cluster_plot(data, labels, names)

        assert result.tell() == 0

    @pytest.mark.xfail(reason="Known seaborn compatibility issue with ListedColormap")
    def test_cluster_plot_handles_multiple_clusters(self):
        """Test cluster plot with 4 clusters."""
        data = np.array([[1, 1], [2, 2], [10, 1], [11, 2], [1, 10], [2, 11], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        result = generate_cluster_plot(data, labels, names)

        assert isinstance(result, io.BytesIO)


class TestActivityHeatmap:
    """Tests for activity heatmap visualization."""

    def test_create_activity_heatmap_returns_bytesio(self):
        """Test that heatmap returns BytesIO with PNG data."""
        from strofkabot.utils import create_activity_heatmap

        data = np.random.randint(0, 100, size=(7, 24))

        result = create_activity_heatmap(data, "TestUser")

        assert isinstance(result, io.BytesIO)
        result.seek(0)
        assert result.read(4) == b"\x89PNG"

    def test_create_activity_heatmap_zero_data(self):
        """Test heatmap handles all-zero data without error."""
        from strofkabot.utils import create_activity_heatmap

        data = np.zeros((7, 24), dtype=int)

        result = create_activity_heatmap(data, "InactiveUser")

        assert isinstance(result, io.BytesIO)

    def test_create_activity_heatmap_with_timezone_label(self):
        """Test custom timezone label is accepted."""
        from strofkabot.utils import create_activity_heatmap

        data = np.random.randint(0, 50, size=(7, 24))

        result = create_activity_heatmap(data, "User", timezone_label="CET")

        assert isinstance(result, io.BytesIO)

    def test_create_activity_heatmap_ready_for_reading(self):
        """Test that buffer position is at start."""
        from strofkabot.utils import create_activity_heatmap

        data = np.ones((7, 24), dtype=int) * 10

        result = create_activity_heatmap(data, "User")

        assert result.tell() == 0
