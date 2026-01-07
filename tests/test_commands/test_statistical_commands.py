"""
Tests for statistical Discord commands (!rpm, !inflation, !trade, !gdp, !hdi, !most-liked).
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import discord.ext.test as dpytest
import pytest


class TestRpmCommand:
    """Tests for the !rpm command."""

    @pytest.mark.asyncio
    async def test_rpm_error_handling(self, bot_with_mocked_db):
        """Test !rpm handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        # Simulate an error by making send_personal_stats raise
        with patch('strofkabot.llumi.send_personal_stats', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Database error")
            await dpytest.message("!rpm")
            response = dpytest.get_message()
            assert "error" in response.content.lower()

    @pytest.mark.asyncio
    async def test_rpm_leaderboard_error(self, bot_with_mocked_db):
        """Test !rpm --leaderboard handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.send_leaderboard', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Leaderboard error")
            await dpytest.message("!rpm --leaderboard")
            response = dpytest.get_message()
            assert "error" in response.content.lower()


class TestInflationCommand:
    """Tests for the !inflation command."""

    @pytest.mark.asyncio
    async def test_inflation_calls_correct_functions(self, bot_with_mocked_db):
        """Test !inflation calls the expected data functions."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        monthly_data = [
            {'year': 2024, 'month': 1, 'average_rpm': 2.0},
            {'year': 2024, 'month': 2, 'average_rpm': 2.3},
        ]
        yearly_data = [
            {'year': 2023, 'average_rpm': 1.8},
            {'year': 2024, 'average_rpm': 2.0},
        ]

        with patch('strofkabot.llumi.fetch_inflation_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (monthly_data, yearly_data)

            with patch('strofkabot.llumi.create_monthly_inflation_plot') as mock_monthly_plot:
                with patch('strofkabot.llumi.create_yearly_inflation_plot') as mock_yearly_plot:
                    mock_monthly_plot.return_value = io.BytesIO(b'fake png data')
                    mock_yearly_plot.return_value = io.BytesIO(b'fake png data')

                    await dpytest.message("!inflation")

                    # Verify functions were called
                    assert mock_fetch.called
                    assert mock_monthly_plot.called
                    assert mock_yearly_plot.called

    @pytest.mark.asyncio
    async def test_inflation_error_handling(self, bot_with_mocked_db):
        """Test !inflation handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.fetch_inflation_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Inflation calculation error")
            await dpytest.message("!inflation")
            response = dpytest.get_message()
            assert "error" in response.content.lower()


class TestTradeCommand:
    """Tests for the !trade command."""

    @pytest.mark.asyncio
    async def test_trade_self(self, bot_with_mocked_db):
        """Test !trade shows own stats when no user mentioned."""
        bot, _, mock_user_stats, cog = bot_with_mocked_db

        # Mock guild for get_member calls
        cog.guild = MagicMock()
        cog.guild.get_member = MagicMock(return_value=None)

        trade_data = {
            'exports': [('Alice', 50), ('Bob', 30)],
            'imports': [('Charlie', 40), ('Diana', 20)],
            'total_given': 100,
            'total_received': 80,
            'trade_balance': -20
        }

        with patch('strofkabot.llumi.get_reaction_trade_data', new_callable=AsyncMock) as mock_trade:
            mock_trade.return_value = trade_data
            await dpytest.message("!trade")
            response = dpytest.get_message()

            assert "Trade Report" in response.content
            assert "Export" in response.content
            assert "Import" in response.content
            assert "DEFICIT" in response.content

    @pytest.mark.asyncio
    async def test_trade_surplus(self, bot_with_mocked_db):
        """Test !trade shows SURPLUS when receiving more than giving."""
        bot, _, mock_user_stats, cog = bot_with_mocked_db

        cog.guild = MagicMock()
        cog.guild.get_member = MagicMock(return_value=None)

        trade_data = {
            'exports': [('Alice', 30)],
            'imports': [('Bob', 50)],
            'total_given': 30,
            'total_received': 100,
            'trade_balance': 70
        }

        with patch('strofkabot.llumi.get_reaction_trade_data', new_callable=AsyncMock) as mock_trade:
            mock_trade.return_value = trade_data
            await dpytest.message("!trade")
            response = dpytest.get_message()

            assert "SURPLUS" in response.content

    @pytest.mark.asyncio
    async def test_trade_neutral(self, bot_with_mocked_db):
        """Test !trade shows NEUTRAL when balanced."""
        bot, _, mock_user_stats, cog = bot_with_mocked_db

        cog.guild = MagicMock()
        cog.guild.get_member = MagicMock(return_value=None)

        trade_data = {
            'exports': [('Alice', 50)],
            'imports': [('Bob', 50)],
            'total_given': 50,
            'total_received': 50,
            'trade_balance': 0
        }

        with patch('strofkabot.llumi.get_reaction_trade_data', new_callable=AsyncMock) as mock_trade:
            mock_trade.return_value = trade_data
            await dpytest.message("!trade")
            response = dpytest.get_message()

            assert "NEUTRAL" in response.content

    @pytest.mark.asyncio
    async def test_trade_no_data(self, bot_with_mocked_db):
        """Test !trade handles no export/import data."""
        bot, _, mock_user_stats, cog = bot_with_mocked_db

        cog.guild = MagicMock()
        cog.guild.get_member = MagicMock(return_value=None)

        trade_data = {
            'exports': [],
            'imports': [],
            'total_given': 0,
            'total_received': 0,
            'trade_balance': 0
        }

        with patch('strofkabot.llumi.get_reaction_trade_data', new_callable=AsyncMock) as mock_trade:
            mock_trade.return_value = trade_data
            await dpytest.message("!trade")
            response = dpytest.get_message()

            assert "No reactions given" in response.content
            assert "No reactions received" in response.content

    @pytest.mark.asyncio
    async def test_trade_error_handling(self, bot_with_mocked_db):
        """Test !trade handles errors gracefully."""
        bot, _, mock_user_stats, cog = bot_with_mocked_db

        cog.guild = MagicMock()

        with patch('strofkabot.llumi.get_reaction_trade_data', new_callable=AsyncMock) as mock_trade:
            mock_trade.side_effect = Exception("Trade data error")
            await dpytest.message("!trade")
            response = dpytest.get_message()
            assert "error" in response.content.lower()


class TestGdpCommand:
    """Tests for the !gdp command."""

    @pytest.mark.asyncio
    async def test_gdp_calls_correct_functions(self, bot_with_mocked_db):
        """Test !gdp calls fetch and plot functions."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        gdp_data = [
            {'year': 2024, 'month': 1, 'total_messages': 1000},
            {'year': 2024, 'month': 2, 'total_messages': 1200},
        ]

        with patch('strofkabot.llumi.fetch_gdp_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = gdp_data

            with patch('strofkabot.llumi.create_gdp_plot') as mock_plot:
                mock_plot.return_value = io.BytesIO(b'fake png data')

                await dpytest.message("!gdp")

                assert mock_fetch.called
                assert mock_plot.called

    @pytest.mark.asyncio
    async def test_gdp_no_data(self, bot_with_mocked_db):
        """Test !gdp handles empty data."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.fetch_gdp_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            await dpytest.message("!gdp")
            response = dpytest.get_message()

            assert "No message data available" in response.content

    @pytest.mark.asyncio
    async def test_gdp_error_handling(self, bot_with_mocked_db):
        """Test !gdp handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.fetch_gdp_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("GDP calculation error")
            await dpytest.message("!gdp")
            response = dpytest.get_message()
            assert "error" in response.content.lower()


class TestHdiCommand:
    """Tests for the !hdi command."""

    @pytest.mark.asyncio
    async def test_hdi_calls_correct_functions(self, bot_with_mocked_db):
        """Test !hdi calls fetch and plot functions."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        hdi_data = [
            {'year': 2024, 'month': 1, 'quality_count': 100, 'total_count': 1000, 'hdi_ratio': 0.1},
            {'year': 2024, 'month': 2, 'quality_count': 150, 'total_count': 1200, 'hdi_ratio': 0.125},
        ]

        with patch('strofkabot.llumi.fetch_hdi_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = hdi_data

            with patch('strofkabot.llumi.create_hdi_plot') as mock_plot:
                mock_plot.return_value = io.BytesIO(b'fake png data')

                await dpytest.message("!hdi")

                assert mock_fetch.called
                assert mock_plot.called

    @pytest.mark.asyncio
    async def test_hdi_no_data(self, bot_with_mocked_db):
        """Test !hdi handles empty data."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.fetch_hdi_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []

            await dpytest.message("!hdi")
            response = dpytest.get_message()

            assert "No message data available" in response.content

    @pytest.mark.asyncio
    async def test_hdi_error_handling(self, bot_with_mocked_db):
        """Test !hdi handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.fetch_hdi_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("HDI calculation error")
            await dpytest.message("!hdi")
            response = dpytest.get_message()
            assert "error" in response.content.lower()


class TestMostLikedCommand:
    """Tests for the !most-liked command."""

    @pytest.mark.asyncio
    async def test_most_liked_calls_utility(self, bot_with_mocked_db):
        """Test !most-liked calls the send_most_liked_stats utility."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.send_most_liked_stats', new_callable=AsyncMock) as mock_send:
            await dpytest.message("!most-liked")
            assert mock_send.called

    @pytest.mark.asyncio
    async def test_most_liked_error_handling(self, bot_with_mocked_db):
        """Test !most-liked handles errors gracefully."""
        bot, _, mock_user_stats, _ = bot_with_mocked_db

        with patch('strofkabot.llumi.send_most_liked_stats', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Network analysis error")
            await dpytest.message("!most-liked")
            response = dpytest.get_message()
            assert "error" in response.content.lower()
