"""
Tests for the !llumi command.
"""

import datetime
from unittest.mock import patch

import discord.ext.test as dpytest
import pytest

from strofkabot.discord_db import Attachment, Message


class TestLlumiCommand:
    """Tests for the !llumi command."""

    @pytest.mark.asyncio
    async def test_llumi_returns_random_message(self, bot_with_mocked_db):
        """Test that !llumi returns a message from the database."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Mock the count methods to return integers
        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 0

        mock_db.get_random_message.return_value = Message(
            id=12345,
            content="This is a test message from the database",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=99999,
        )

        await dpytest.message("!llumi")
        assert dpytest.verify().message().content("This is a test message from the database")

    @pytest.mark.asyncio
    async def test_llumi_handles_empty_database(self, bot_with_mocked_db):
        """Test that !llumi handles empty database gracefully."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Mock the count methods to return 0
        mock_db.get_message_count.return_value = 0
        mock_db.get_attachment_count.return_value = 0

        mock_db.get_random_message.return_value = None

        await dpytest.message("!llumi")
        assert dpytest.verify().message().content("No messages available at the moment.")

    @pytest.mark.asyncio
    async def test_llumi_calls_get_random_message(self, bot_with_mocked_db):
        """Test that !llumi calls get_random_message on the database."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Mock the count methods to return integers (all messages, no attachments)
        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 0

        mock_db.get_random_message.return_value = Message(
            id=1, content="Test", timestamp=datetime.datetime.now(), reaction_count=5, author_id=1
        )

        await dpytest.message("!llumi")
        mock_db.get_random_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_llumi_image_flag_handles_no_images(self, bot_with_mocked_db):
        """Test that !llumi -i handles case with no images."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 0

        await dpytest.message("!llumi -i")
        assert dpytest.verify().message().content("No images available.")

    @pytest.mark.asyncio
    async def test_llumi_easter_egg_when_random_is_1(self, bot_with_mocked_db):
        """Test that !llumi returns Easter egg 'Ik qiu Jordi' when random is 1."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 5

        with patch("strofkabot.cogs.entertainment.random.randint", return_value=1):
            await dpytest.message("!llumi")
            assert dpytest.verify().message().content("Ik qiu Jordi")

    @pytest.mark.asyncio
    async def test_llumi_no_easter_egg_when_random_not_1(self, bot_with_mocked_db):
        """Test that !llumi does not return Easter egg when random is not 1."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 0

        mock_db.get_random_message.return_value = Message(
            id=1,
            content="Regular message",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=1,
        )

        with patch("strofkabot.cogs.entertainment.random.randint", return_value=2):
            await dpytest.message("!llumi")
            assert dpytest.verify().message().content("Regular message")

    @pytest.mark.asyncio
    async def test_llumi_50_percent_image_when_random_below_half(
        self, bot_with_mocked_db, tmp_path
    ):
        """Test that !llumi returns image when random < 0.5 and attachments exist."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 5

        # Create a temporary image file
        test_image = tmp_path / "test_image.jpg"
        test_image.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header

        mock_db.get_random_attachment.return_value = Attachment(
            id=1,
            message_id=123,
            message_content=None,
            author_id=1,
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            original_filename="test.jpg",
            local_path=str(test_image),
        )

        with patch(
            "strofkabot.cogs.entertainment.random.randint", return_value=2
        ):  # Skip Easter egg
            with patch("strofkabot.cogs.entertainment.random.random", return_value=0.3):  # < 0.5
                with patch("strofkabot.cogs.entertainment.ATTACHMENTS_DIR", tmp_path):
                    await dpytest.message("!llumi")
                    # Should send an attachment, not a text message
                    mock_db.get_random_attachment.assert_called_once()

    @pytest.mark.asyncio
    async def test_llumi_message_when_random_above_half(self, bot_with_mocked_db):
        """Test that !llumi returns message when random >= 0.5."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 5

        mock_db.get_random_message.return_value = Message(
            id=1,
            content="Text message",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=1,
        )

        with patch(
            "strofkabot.cogs.entertainment.random.randint", return_value=2
        ):  # Skip Easter egg
            with patch("strofkabot.cogs.entertainment.random.random", return_value=0.7):  # >= 0.5
                await dpytest.message("!llumi")
                assert dpytest.verify().message().content("Text message")
                mock_db.get_random_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_llumi_falls_back_to_message_when_no_attachments(self, bot_with_mocked_db):
        """Test that !llumi falls back to message when no attachments exist."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_message_count.return_value = 10
        mock_db.get_attachment_count.return_value = 0  # No attachments

        mock_db.get_random_message.return_value = Message(
            id=1,
            content="Fallback message",
            timestamp=datetime.datetime.now(),
            reaction_count=5,
            author_id=1,
        )

        with patch(
            "strofkabot.cogs.entertainment.random.randint", return_value=2
        ):  # Skip Easter egg
            with patch(
                "strofkabot.cogs.entertainment.random.random", return_value=0.3
            ):  # Would be image, but none exist
                await dpytest.message("!llumi")
                assert dpytest.verify().message().content("Fallback message")
                mock_db.get_random_message.assert_called_once()


class TestOnThisDayCommand:
    """Tests for the !on-this-day and !otd commands."""

    @pytest.mark.asyncio
    async def test_otd_returns_historical_message(self, bot_with_mocked_db):
        """Test that !otd returns a message from a previous year."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_on_this_day_years.return_value = [2022, 2023]
        mock_db.get_top_message_on_this_day.return_value = (
            Message(
                id=1,
                content="Historical message from 2023",
                timestamp=datetime.datetime(2023, 1, 7, 10, 0, 0),
                reaction_count=15,
                author_id=123,
            ),
            None,
        )
        mock_db.get_username_by_id.return_value = "TestUser"

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 1, 7, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            with patch("strofkabot.cogs.entertainment.random.choice", return_value=2023):
                await dpytest.message("!otd")

        response = dpytest.get_message()
        assert "On This Day" in response.content
        assert "1 year ago" in response.content
        assert "15 reactions" in response.content
        assert "**TestUser**" in response.content
        assert "Historical message from 2023" in response.content

    @pytest.mark.asyncio
    async def test_on_this_day_alias_works(self, bot_with_mocked_db):
        """Test that !on-this-day command works (alias of !otd)."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_on_this_day_years.return_value = [2022]
        mock_db.get_top_message_on_this_day.return_value = (
            Message(
                id=1,
                content="Test message",
                timestamp=datetime.datetime(2022, 6, 15, 10, 0, 0),
                reaction_count=10,
                author_id=123,
            ),
            None,
        )
        mock_db.get_username_by_id.return_value = "AnotherUser"

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 6, 15, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            with patch("strofkabot.cogs.entertainment.random.choice", return_value=2022):
                await dpytest.message("!on-this-day")

        response = dpytest.get_message()
        assert "On This Day" in response.content
        assert "2 years ago" in response.content

    @pytest.mark.asyncio
    async def test_otd_handles_no_history(self, bot_with_mocked_db):
        """Test that !otd handles case with no historical messages."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_on_this_day_years.return_value = []

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 1, 7, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            await dpytest.message("!otd")

        response = dpytest.get_message()
        assert "No historical messages found" in response.content
        assert "January 07" in response.content

    @pytest.mark.asyncio
    async def test_otd_filters_current_year(self, bot_with_mocked_db):
        """Test that current year is filtered out."""
        bot, mock_db, _, _ = bot_with_mocked_db

        # Only current year has data
        mock_db.get_on_this_day_years.return_value = [2024]

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 1, 7, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            await dpytest.message("!otd")

        response = dpytest.get_message()
        assert "No historical messages found" in response.content

    @pytest.mark.asyncio
    async def test_otd_shows_years_plural_for_multiple_years(self, bot_with_mocked_db):
        """Test that 'years' is used for 2+ years ago."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_on_this_day_years.return_value = [2020]
        mock_db.get_top_message_on_this_day.return_value = (
            Message(
                id=1,
                content="Old message",
                timestamp=datetime.datetime(2020, 3, 10, 10, 0, 0),
                reaction_count=8,
                author_id=123,
            ),
            None,
        )
        mock_db.get_username_by_id.return_value = "OldUser"

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 3, 10, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            with patch("strofkabot.cogs.entertainment.random.choice", return_value=2020):
                await dpytest.message("!otd")

        response = dpytest.get_message()
        assert "4 years ago" in response.content

    @pytest.mark.asyncio
    async def test_otd_randomly_selects_year(self, bot_with_mocked_db):
        """Test that !otd randomly selects from available years."""
        bot, mock_db, _, _ = bot_with_mocked_db

        mock_db.get_on_this_day_years.return_value = [2020, 2021, 2022, 2023]
        mock_db.get_top_message_on_this_day.return_value = (
            Message(
                id=1,
                content="Message from 2021",
                timestamp=datetime.datetime(2021, 5, 5, 10, 0, 0),
                reaction_count=12,
                author_id=123,
            ),
            None,
        )
        mock_db.get_username_by_id.return_value = "RandomUser"

        with patch("strofkabot.cogs.entertainment.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 5, 5, 12, 0, 0, tzinfo=datetime.UTC
            )
            mock_datetime.UTC = datetime.UTC
            with patch(
                "strofkabot.cogs.entertainment.random.choice", return_value=2021
            ) as mock_choice:
                await dpytest.message("!otd")
                # Verify random.choice was called with the past years list
                mock_choice.assert_called_once_with([2020, 2021, 2022, 2023])
