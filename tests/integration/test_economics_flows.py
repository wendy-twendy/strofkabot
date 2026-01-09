"""
Integration tests for economics data flows.

Tests the full flow from data insertion through statistics calculation
for GDP, HDI, and inflation metrics.
"""

import datetime

import pytest


class TestGDPDataFlow:
    """Integration tests for GDP (message count) data flows."""

    async def test_gdp_data_accuracy(self, real_user_stats, real_database):
        """Test that GDP data accurately reflects total messages per month."""
        # Insert message stats for multiple months
        stats_data = [
            # January 2024: 3 messages
            (111, 5, datetime.datetime(2024, 1, 10)),
            (111, 3, datetime.datetime(2024, 1, 15)),
            (222, 7, datetime.datetime(2024, 1, 20)),
            # February 2024: 2 messages
            (111, 10, datetime.datetime(2024, 2, 5)),
            (333, 2, datetime.datetime(2024, 2, 10)),
            # March 2024: 4 messages
            (111, 4, datetime.datetime(2024, 3, 1)),
            (222, 6, datetime.datetime(2024, 3, 10)),
            (333, 8, datetime.datetime(2024, 3, 15)),
            (444, 1, datetime.datetime(2024, 3, 20)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch GDP data
        gdp_data = await real_user_stats.get_gdp_data(limit=None)

        # Convert to dict for easier assertion
        gdp_by_month = {(d["year"], d["month"]): d["total_messages"] for d in gdp_data}

        assert gdp_by_month[(2024, 1)] == 3  # 3 messages in January
        assert gdp_by_month[(2024, 2)] == 2  # 2 messages in February
        assert gdp_by_month[(2024, 3)] == 4  # 4 messages in March

    async def test_gdp_respects_limit(self, real_user_stats):
        """Test that GDP data limit is respected."""
        # Add stats for many months
        stats_data = []
        for month in range(1, 13):
            stats_data.append((111, 5, datetime.datetime(2024, month, 15)))

        await real_user_stats.batch_update_stats(stats_data)

        # Request only 6 months
        gdp_data = await real_user_stats.get_gdp_data(limit=6)
        assert len(gdp_data) == 6

    async def test_gdp_empty_database(self, real_user_stats):
        """Test GDP data returns empty list for empty database."""
        gdp_data = await real_user_stats.get_gdp_data()
        assert gdp_data == []


class TestHDIDataFlow:
    """Integration tests for HDI (quality ratio) data flows."""

    async def test_hdi_data_quality_ratio(self, real_user_stats, real_database, message_factory):
        """Test that HDI correctly calculates quality message ratio."""
        # Add user stats (represents all messages)
        stats_data = [
            # 10 messages in January 2024
            (111, 5, datetime.datetime(2024, 1, i))
            for i in range(1, 11)
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Add quality messages (messages with 4+ reactions stored in messages table)
        for i in range(1, 5):  # 4 quality messages
            await message_factory(
                message_id=1000 + i,
                content=f"Quality message {i}",
                author_id=111,
                reaction_count=10,
                timestamp=datetime.datetime(2024, 1, i),
            )

        # Fetch HDI data
        hdi_data = await real_user_stats.get_hdi_data(limit=24)

        # Find January 2024 data
        jan_data = next((d for d in hdi_data if d["year"] == 2024 and d["month"] == 1), None)

        assert jan_data is not None
        assert jan_data["total_count"] == 10  # Total messages
        assert jan_data["quality_count"] == 4  # Quality messages
        assert jan_data["hdi_ratio"] == pytest.approx(0.4, rel=0.01)  # 4/10 = 0.4

    async def test_hdi_handles_zero_total(self, real_user_stats):
        """Test that HDI handles months with no messages gracefully."""
        hdi_data = await real_user_stats.get_hdi_data()
        assert hdi_data == []


class TestInflationDataFlow:
    """Integration tests for inflation (reactions per message) data flows."""

    async def test_monthly_inflation_data(self, real_user_stats):
        """Test monthly inflation data aggregation."""
        # Add stats with varying reaction counts per message
        stats_data = [
            # January: 3 messages, 15 total reactions (avg 5.0)
            (111, 5, datetime.datetime(2024, 1, 1)),
            (222, 5, datetime.datetime(2024, 1, 10)),
            (333, 5, datetime.datetime(2024, 1, 20)),
            # February: 2 messages, 16 total reactions (avg 8.0)
            (111, 8, datetime.datetime(2024, 2, 5)),
            (222, 8, datetime.datetime(2024, 2, 15)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch monthly inflation data
        monthly_data = await real_user_stats.get_reaction_inflation_raw(monthly=True)

        jan_data = next((d for d in monthly_data if d["year"] == 2024 and d["month"] == 1), None)
        feb_data = next((d for d in monthly_data if d["year"] == 2024 and d["month"] == 2), None)

        assert jan_data is not None
        assert jan_data["total_messages"] == 3
        assert jan_data["total_reactions"] == 15
        assert jan_data["average_rpm"] == pytest.approx(5.0, rel=0.01)

        assert feb_data is not None
        assert feb_data["total_messages"] == 2
        assert feb_data["total_reactions"] == 16
        assert feb_data["average_rpm"] == pytest.approx(8.0, rel=0.01)

    async def test_yearly_inflation_data(self, real_user_stats):
        """Test yearly inflation data aggregation."""
        # Add stats across two years
        stats_data = [
            # 2023: 4 messages, 20 reactions (avg 5.0)
            (111, 5, datetime.datetime(2023, 6, 1)),
            (111, 5, datetime.datetime(2023, 7, 1)),
            (111, 5, datetime.datetime(2023, 8, 1)),
            (111, 5, datetime.datetime(2023, 9, 1)),
            # 2024: 3 messages, 24 reactions (avg 8.0)
            (111, 8, datetime.datetime(2024, 1, 1)),
            (111, 8, datetime.datetime(2024, 2, 1)),
            (111, 8, datetime.datetime(2024, 3, 1)),
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Fetch yearly inflation data
        yearly_data = await real_user_stats.get_reaction_inflation_raw(monthly=False)

        year_2023 = next((d for d in yearly_data if d["year"] == 2023), None)
        year_2024 = next((d for d in yearly_data if d["year"] == 2024), None)

        assert year_2023 is not None
        assert year_2023["total_messages"] == 4
        assert year_2023["total_reactions"] == 20
        assert year_2023["average_rpm"] == pytest.approx(5.0, rel=0.01)

        assert year_2024 is not None
        assert year_2024["total_messages"] == 3
        assert year_2024["total_reactions"] == 24
        assert year_2024["average_rpm"] == pytest.approx(8.0, rel=0.01)


class TestTradeDataFlow:
    """Integration tests for reaction trade data flows."""

    async def test_reaction_exports_imports(self, real_user_stats, reaction_factory):
        """Test that reaction exports/imports are correctly calculated."""
        # User 111 gives reactions to others (exports)
        await reaction_factory(giver_id=111, receiver_id=222, count=5)
        await reaction_factory(giver_id=111, receiver_id=333, count=3)

        # User 111 receives reactions from others (imports)
        await reaction_factory(giver_id=222, receiver_id=111, count=2)
        await reaction_factory(giver_id=333, receiver_id=111, count=4)

        # Query the trade data for user 111 for the specific month
        trade_data = await real_user_stats.get_reaction_trade_data_for_month(111, 2024, 1)

        assert trade_data is not None
        assert trade_data["total_given"] == 8  # 5 + 3 = 8 reactions given
        assert trade_data["total_received"] == 6  # 2 + 4 = 6 reactions received

    async def test_trade_balance_calculation(self, real_user_stats, reaction_factory):
        """Test trade balance (received - given) calculation."""
        # User with negative balance (gives more than receives)
        await reaction_factory(giver_id=111, receiver_id=222, count=10)
        await reaction_factory(giver_id=222, receiver_id=111, count=3)

        trade_data = await real_user_stats.get_reaction_trade_data_for_month(111, 2024, 1)

        # Trade balance = received - given = 3 - 10 = -7
        expected_balance = 3 - 10  # -7
        assert trade_data["trade_balance"] == expected_balance

    async def test_trade_top_partners(self, real_user_stats, reaction_factory):
        """Test top trade partners are correctly identified."""
        # User 111's reaction relationships
        await reaction_factory(giver_id=111, receiver_id=222, count=10)  # Most exports
        await reaction_factory(giver_id=111, receiver_id=333, count=5)
        await reaction_factory(giver_id=111, receiver_id=444, count=2)

        await reaction_factory(giver_id=222, receiver_id=111, count=8)  # Most imports
        await reaction_factory(giver_id=333, receiver_id=111, count=3)

        trade_data = await real_user_stats.get_reaction_trade_data_for_month(111, 2024, 1)

        # Exports list should have 222 first (10 reactions) - exports is list of (receiver_id, count)
        assert len(trade_data["exports"]) > 0
        top_export = trade_data["exports"][0]
        assert top_export[0] == 222  # receiver_id
        assert top_export[1] == 10  # count

        # Imports list should have 222 first (8 reactions) - imports is list of (giver_id, count)
        assert len(trade_data["imports"]) > 0
        top_import = trade_data["imports"][0]
        assert top_import[0] == 222  # giver_id
        assert top_import[1] == 8  # count


class TestCrossMonthDataIntegrity:
    """Integration tests for data integrity across months."""

    async def test_month_isolation(self, real_user_stats, reaction_factory):
        """Test that data from different months is properly isolated."""
        # January reactions
        await reaction_factory(giver_id=111, receiver_id=222, count=5, year=2024, month=1)

        # February reactions
        await reaction_factory(giver_id=111, receiver_id=222, count=10, year=2024, month=2)

        # Query each month separately
        jan_data = await real_user_stats.get_reaction_trade_data_for_month(111, 2024, 1)
        feb_data = await real_user_stats.get_reaction_trade_data_for_month(111, 2024, 2)

        assert jan_data["total_given"] == 5
        assert feb_data["total_given"] == 10

    async def test_year_boundary_handling(self, real_user_stats):
        """Test data handling across year boundaries."""
        stats_data = [
            (111, 5, datetime.datetime(2023, 12, 15)),  # December 2023
            (111, 8, datetime.datetime(2024, 1, 15)),  # January 2024
        ]
        await real_user_stats.batch_update_stats(stats_data)

        # Get yearly data
        yearly_data = await real_user_stats.get_reaction_inflation_raw(monthly=False)

        years = {d["year"] for d in yearly_data}
        assert 2023 in years
        assert 2024 in years
