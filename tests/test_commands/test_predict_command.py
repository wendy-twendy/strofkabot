"""
Tests for the !predict command.
"""

import datetime

import discord.ext.test as dpytest
import pytest


class TestPredictCommand:
    """Tests for the !predict command."""

    @pytest.mark.asyncio
    async def test_predict_no_args_shows_usage(self, bot_with_mocked_db):
        """Test !predict with no args shows usage."""
        bot, mock_db, _, _ = bot_with_mocked_db

        await dpytest.message("!predict")
        response = dpytest.get_message()
        assert "Usage:" in response.content
        assert "DD-MM-YYYY" in response.content

    @pytest.mark.asyncio
    async def test_predict_invalid_date_shows_error(self, bot_with_mocked_db):
        """Test !predict with invalid date shows error."""
        bot, mock_db, _, _ = bot_with_mocked_db

        await dpytest.message("!predict asdfghjkl some prediction")
        response = dpytest.get_message()
        assert "couldn't understand" in response.content.lower()
        assert "DD-MM-YYYY" in response.content

    @pytest.mark.asyncio
    async def test_predict_past_date_shows_error(self, bot_with_mocked_db):
        """Test !predict with past date shows error."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Use an explicit past date that will always be in the past
        await dpytest.message("!predict 01-01-2020 something will happen")

        response = dpytest.get_message()
        assert "past" in response.content.lower()

    @pytest.mark.asyncio
    async def test_predict_no_text_shows_error(self, bot_with_mocked_db):
        """Test !predict with date but no text shows error."""
        bot, mock_db, _, _ = bot_with_mocked_db

        await dpytest.message("!predict tomorrow")

        response = dpytest.get_message()
        assert "prediction text" in response.content.lower()

    @pytest.mark.asyncio
    async def test_predict_success_stores_and_confirms(self, bot_with_mocked_db):
        """Test successful prediction is stored and confirmed."""
        bot, mock_db, _, _ = bot_with_mocked_db
        mock_db.add_prediction.return_value = 1

        await dpytest.message("!predict tomorrow The sun will rise")

        response = dpytest.get_message()
        # Check for embed response
        assert response.embeds
        assert "Prediction Recorded" in response.embeds[0].title
        mock_db.add_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_date_too_far_in_future(self, bot_with_mocked_db):
        """Test !predict with date > 5 years in future shows error."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Use a date that's definitely more than 5 years in the future
        await dpytest.message("!predict 01-01-2035 Something in the far future")

        response = dpytest.get_message()
        assert "far in the future" in response.content.lower()

    @pytest.mark.asyncio
    async def test_predict_dmy_format_works(self, bot_with_mocked_db):
        """Test !predict with DD-MM-YYYY format works."""
        bot, mock_db, _, _ = bot_with_mocked_db
        mock_db.add_prediction.return_value = 1

        # Use a future date in DD-MM-YYYY format
        await dpytest.message("!predict 25-12-2027 Christmas will be white")

        response = dpytest.get_message()
        assert response.embeds
        assert "Prediction Recorded" in response.embeds[0].title
        # Verify the date was parsed correctly (December 25, 2027)
        call_args = mock_db.add_prediction.call_args
        assert call_args.kwargs["target_date"] == datetime.date(2027, 12, 25)

    @pytest.mark.asyncio
    async def test_predict_natural_language_date(self, bot_with_mocked_db):
        """Test !predict with natural language date like 'next week'."""
        bot, mock_db, _, _ = bot_with_mocked_db
        mock_db.add_prediction.return_value = 1

        await dpytest.message("!predict next week Something will happen")

        response = dpytest.get_message()
        assert response.embeds
        assert "Prediction Recorded" in response.embeds[0].title
        mock_db.add_prediction.assert_called_once()
