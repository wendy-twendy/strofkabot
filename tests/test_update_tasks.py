"""Tests for BackgroundTaskManager class."""

import datetime
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.tasks.update_tasks import BackgroundTaskManager


class TestBackgroundTaskManagerInit:
    """Tests for BackgroundTaskManager initialization."""

    def test_init_stores_dependencies(self):
        """Test that all dependencies are stored correctly."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

        assert manager.bot is bot
        assert manager.db is db
        assert manager.user_stats is user_stats
        assert manager.message_filter is message_filter
        assert manager.logger is logger
        assert manager.guild is None

    def test_init_creates_image_processor(self):
        """Test that image processor is created."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor") as MockImageProcessor:
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)
            MockImageProcessor.assert_called_once()
            assert manager.image_processor is not None


class TestSetGuild:
    """Tests for set_guild method."""

    def test_sets_guild(self):
        """Test that guild is set correctly."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

        mock_guild = MagicMock()
        manager.set_guild(mock_guild)

        assert manager.guild is mock_guild


class TestClose:
    """Tests for close method."""

    async def test_close_is_noop(self):
        """Test that close method is a no-op (database closed by LlumiBot)."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

        # Should not raise
        await manager.close()


class TestUpdateDb:
    """Tests for update_db method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        db.get_last_scanned_timestamp = AsyncMock(return_value=None)
        db.update_last_scanned_timestamp = AsyncMock()
        db.add_history_messages = AsyncMock()
        db.update_scrape_progress = AsyncMock()
        user_stats = MagicMock()
        user_stats.batch_update_stats = AsyncMock()
        user_stats.batch_update_reaction_stats = AsyncMock()
        message_filter = MagicMock()
        message_filter.is_valid_message = MagicMock(return_value=True)
        logger = logging.getLogger("test")
        mock_image_processor = MagicMock()

        manager = BackgroundTaskManager(
            bot,
            db,
            user_stats,
            message_filter,
            logger,
            image_processor=mock_image_processor,
        )
        return manager

    async def test_skips_when_guild_not_set(self, manager):
        """Test that update is skipped when guild is not set."""
        manager.guild = None

        with patch.object(manager.logger, "warning") as mock_warning:
            await manager.update_db()
            mock_warning.assert_called_once()
            assert "Guild not set" in mock_warning.call_args[0][0]

    async def test_processes_accessible_channels(self, manager):
        """Test that only accessible channels are processed."""
        mock_channel1 = MagicMock()
        mock_channel1.permissions_for = MagicMock(return_value=MagicMock(read_messages=True))
        mock_channel2 = MagicMock()
        mock_channel2.permissions_for = MagicMock(return_value=MagicMock(read_messages=False))

        mock_guild = MagicMock()
        mock_guild.text_channels = [mock_channel1, mock_channel2]
        mock_guild.me = MagicMock()
        manager.guild = mock_guild

        with patch.object(manager, "_process_channel", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = (0, 0)
            await manager.update_db()

            # Should only process accessible channel
            assert mock_process.call_count == 1


class TestProcessChannel:
    """Tests for _process_channel method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        bot.user = MagicMock()
        bot.user.id = 12345
        db = MagicMock()
        db.get_last_scanned_timestamp = AsyncMock(return_value=None)
        db.update_last_scanned_timestamp = AsyncMock()
        db.add_messages = AsyncMock()
        db.add_attachments = AsyncMock()
        db.batch_upsert_reply_stats = AsyncMock()
        db.attachment_exists = AsyncMock(return_value=False)
        db.add_history_messages = AsyncMock()
        db.update_scrape_progress = AsyncMock()
        user_stats = MagicMock()
        user_stats.batch_update_stats = AsyncMock()
        user_stats.batch_update_reaction_stats = AsyncMock()
        message_filter = MagicMock()
        message_filter.is_valid_message = MagicMock(return_value=True)
        logger = logging.getLogger("test")
        mock_image_processor = MagicMock()
        mock_image_processor.is_image = MagicMock(return_value=False)

        manager = BackgroundTaskManager(
            bot,
            db,
            user_stats,
            message_filter,
            logger,
            image_processor=mock_image_processor,
        )
        manager.react_count_threshold = 4
        return manager

    async def test_skips_bot_messages(self, manager):
        """Test that bot messages are skipped."""
        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = manager.bot.user.id  # Bot's own message
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = []

        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(return_value=AsyncIterator([mock_message]))

        scan_until = datetime.datetime.now(datetime.UTC)
        msg_count, react_count = await manager._process_channel(mock_channel, scan_until)

        # Should not have processed the message
        assert msg_count == 0

    async def test_records_history_messages(self, manager):
        """Test that all messages are recorded to history."""
        mock_author = MagicMock()
        mock_author.id = 67890
        mock_author.display_name = "TestUser"

        mock_message = MagicMock()
        mock_message.id = 100
        mock_message.author = mock_author
        mock_message.content = "Test message"
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = []
        mock_message.reference = None
        mock_message.attachments = []

        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(return_value=AsyncIterator([mock_message]))

        scan_until = datetime.datetime.now(datetime.UTC)
        await manager._process_channel(mock_channel, scan_until)

        # Should have added to history
        manager.db.add_history_messages.assert_called()

    async def test_processes_high_reaction_messages(self, manager):
        """Test that high reaction messages are processed for quality."""
        mock_author = MagicMock()
        mock_author.id = 67890
        mock_author.display_name = "TestUser"

        mock_reaction = MagicMock()
        mock_reaction.count = 5  # Above threshold of 4
        mock_reaction.emoji = "👍"
        mock_reaction.users = MagicMock(return_value=AsyncIterator([]))

        mock_message = MagicMock()
        mock_message.id = 100
        mock_message.author = mock_author
        mock_message.content = "This is a quality message with enough content"
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = [mock_reaction]
        mock_message.reference = None
        mock_message.attachments = []

        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(return_value=AsyncIterator([mock_message]))

        scan_until = datetime.datetime.now(datetime.UTC)
        msg_count, react_count = await manager._process_channel(mock_channel, scan_until)

        assert msg_count == 1
        assert react_count == 5

    async def test_collects_reply_stats(self, manager):
        """Test that reply statistics are collected."""
        mock_author = MagicMock()
        mock_author.id = 67890

        mock_replied_to = MagicMock()
        mock_replied_to.author = MagicMock()
        mock_replied_to.author.id = 11111
        mock_replied_to.author.display_name = "RepliedUser"
        mock_replied_to.content = "Original message"

        mock_message = MagicMock()
        mock_message.id = 100
        mock_message.author = mock_author
        mock_message.author.display_name = "TestUser"
        mock_message.content = "Reply message"
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = []
        mock_message.reference = MagicMock()
        mock_message.reference.resolved = mock_replied_to
        mock_message.attachments = []

        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(return_value=AsyncIterator([mock_message]))

        scan_until = datetime.datetime.now(datetime.UTC)
        await manager._process_channel(mock_channel, scan_until)

        # Should have upserted reply stats
        manager.db.batch_upsert_reply_stats.assert_called()

    async def test_handles_channel_processing_error(self, manager):
        """Test that channel processing errors are handled."""
        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(side_effect=Exception("Discord API error"))

        scan_until = datetime.datetime.now(datetime.UTC)

        with patch.object(manager.logger, "exception") as mock_exception:
            msg_count, react_count = await manager._process_channel(mock_channel, scan_until)
            mock_exception.assert_called()

        assert msg_count == 0
        assert react_count == 0

    async def test_processes_attachments(self, manager):
        """Test that image attachments are processed."""
        manager.image_processor.is_image = MagicMock(return_value=True)
        manager.image_processor.process_attachment = AsyncMock(return_value=(MagicMock(), 1000))

        mock_author = MagicMock()
        mock_author.id = 67890
        mock_author.display_name = "TestUser"

        mock_reaction = MagicMock()
        mock_reaction.count = 5
        mock_reaction.emoji = "👍"
        mock_reaction.users = MagicMock(return_value=AsyncIterator([]))

        mock_attachment = MagicMock()
        mock_attachment.id = 999
        mock_attachment.filename = "image.jpg"
        mock_attachment.url = "http://example.com/image.jpg"

        mock_message = MagicMock()
        mock_message.id = 100
        mock_message.author = mock_author
        mock_message.content = "Check out this image!"
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = [mock_reaction]
        mock_message.reference = None
        mock_message.attachments = [mock_attachment]

        mock_channel = MagicMock()
        mock_channel.id = 123
        mock_channel.name = "test-channel"
        mock_channel.history = MagicMock(return_value=AsyncIterator([mock_message]))

        scan_until = datetime.datetime.now(datetime.UTC)
        await manager._process_channel(mock_channel, scan_until)

        manager.image_processor.process_attachment.assert_called_once()


class TestCollectReactions:
    """Tests for _collect_reactions method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            return BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

    async def test_collects_user_reactions(self, manager):
        """Test that user reactions are collected."""
        mock_user = MagicMock()
        mock_user.id = 11111
        mock_user.bot = False

        mock_reaction = MagicMock()
        mock_reaction.users = MagicMock(return_value=AsyncIterator([mock_user]))

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 22222
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = [mock_reaction]

        reactions = await manager._collect_reactions(mock_message)

        assert len(reactions) == 1
        assert reactions[0][0] == 11111  # giver_id
        assert reactions[0][1] == 22222  # receiver_id

    async def test_skips_bot_reactions(self, manager):
        """Test that bot reactions are skipped."""
        mock_bot_user = MagicMock()
        mock_bot_user.id = 11111
        mock_bot_user.bot = True

        mock_reaction = MagicMock()
        mock_reaction.users = MagicMock(return_value=AsyncIterator([mock_bot_user]))

        mock_message = MagicMock()
        mock_message.author = MagicMock()
        mock_message.author.id = 22222
        mock_message.created_at = datetime.datetime.now(datetime.UTC)
        mock_message.reactions = [mock_reaction]

        reactions = await manager._collect_reactions(mock_message)

        assert len(reactions) == 0


class TestGetAllReacts:
    """Tests for _get_all_reacts method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            return BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

    def test_sums_reaction_counts(self, manager):
        """Test that all reaction counts are summed."""
        mock_reaction1 = MagicMock()
        mock_reaction1.count = 5
        mock_reaction2 = MagicMock()
        mock_reaction2.count = 3

        mock_message = MagicMock()
        mock_message.reactions = [mock_reaction1, mock_reaction2]

        total = manager._get_all_reacts(mock_message)

        assert total == 8

    def test_returns_zero_for_no_reactions(self, manager):
        """Test that zero is returned for no reactions."""
        mock_message = MagicMock()
        mock_message.reactions = []

        total = manager._get_all_reacts(mock_message)

        assert total == 0


class TestSerializeReactions:
    """Tests for _serialize_reactions method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            return BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

    def test_serializes_reactions_to_json(self, manager):
        """Test that reactions are serialized to JSON."""
        mock_reaction1 = MagicMock()
        mock_reaction1.emoji = "👍"
        mock_reaction1.count = 5
        mock_reaction2 = MagicMock()
        mock_reaction2.emoji = "❤️"
        mock_reaction2.count = 3

        result = manager._serialize_reactions([mock_reaction1, mock_reaction2])

        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["emoji"] == "👍"
        assert parsed[0]["count"] == 5
        assert parsed[1]["emoji"] == "❤️"
        assert parsed[1]["count"] == 3

    def test_handles_empty_reactions(self, manager):
        """Test that empty reactions list is handled."""
        result = manager._serialize_reactions([])

        parsed = json.loads(result)
        assert parsed == []


class TestUpdateUsernames:
    """Tests for update_usernames method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        user_stats.update_user_mapping = AsyncMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            return BackgroundTaskManager(bot, db, user_stats, message_filter, logger)

    async def test_skips_when_guild_not_set(self, manager):
        """Test that update is skipped when guild is not set."""
        manager.guild = None

        with patch.object(manager.logger, "warning") as mock_warning:
            await manager.update_usernames()
            mock_warning.assert_called_once()
            assert "Guild not set" in mock_warning.call_args[0][0]

    async def test_skips_when_updated_recently(self, manager):
        """Test that update is skipped if updated less than 24 hours ago."""
        manager.guild = MagicMock()
        manager.last_username_update = datetime.datetime.now(datetime.UTC)

        with patch.object(manager.logger, "info") as mock_info:
            await manager.update_usernames()
            # Should have logged skip message
            assert any("skipped" in str(call).lower() for call in mock_info.call_args_list)

    async def test_updates_all_members(self, manager):
        """Test that all guild members are updated."""
        mock_member1 = MagicMock()
        mock_member1.id = 111
        mock_member1.display_name = "User1"
        mock_member2 = MagicMock()
        mock_member2.id = 222
        mock_member2.display_name = "User2"

        mock_guild = MagicMock()
        mock_guild.id = 999
        mock_guild.name = "Test Guild"
        mock_guild.member_count = 2
        mock_guild.members = [mock_member1, mock_member2]

        manager.guild = mock_guild
        manager.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        await manager.update_usernames()

        assert manager.user_stats.update_user_mapping.call_count == 2

    async def test_strips_emojis_from_names(self, manager):
        """Test that emojis are stripped from display names."""
        mock_member = MagicMock()
        mock_member.id = 111
        mock_member.display_name = "🎮 Gamer 🎮"

        mock_guild = MagicMock()
        mock_guild.id = 999
        mock_guild.name = "Test Guild"
        mock_guild.member_count = 1
        mock_guild.members = [mock_member]

        manager.guild = mock_guild
        manager.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        await manager.update_usernames()

        # Check that emoji was stripped
        call_args = manager.user_stats.update_user_mapping.call_args
        assert "🎮" not in call_args[0][1]


class TestCheckPredictions:
    """Tests for check_predictions method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        bot.get_channel = MagicMock()
        db = MagicMock()
        db.get_due_predictions = AsyncMock(return_value=[])
        db.mark_prediction_posted = AsyncMock()
        db.increment_prediction_retry = AsyncMock(return_value=1)
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)
            manager.guild = MagicMock()
            return manager

    async def test_skips_when_guild_not_set(self, manager):
        """Test that check is skipped when guild is not set."""
        manager.guild = None

        with patch.object(manager.logger, "warning") as mock_warning:
            await manager.check_predictions()
            mock_warning.assert_called_once()
            assert "Guild not set" in mock_warning.call_args[0][0]

    async def test_returns_early_when_no_predictions(self, manager):
        """Test that method returns early when no predictions are due."""
        manager.db.get_due_predictions = AsyncMock(return_value=[])

        await manager.check_predictions()

        # Should not have tried to post anything
        manager.db.mark_prediction_posted.assert_not_called()

    async def test_posts_overdue_predictions(self, manager):
        """Test that overdue predictions are posted."""
        from strofkabot.discord_db import Prediction

        yesterday = datetime.datetime.now(datetime.UTC).date() - datetime.timedelta(days=1)
        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=yesterday,
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        manager.db.get_due_predictions = AsyncMock(return_value=[mock_prediction])

        with patch.object(manager, "_post_prediction", new_callable=AsyncMock) as mock_post:
            await manager.check_predictions()
            mock_post.assert_called_once_with(mock_prediction)

    async def test_posts_midday_predictions(self, manager):
        """Test that today's predictions are posted at midday."""
        from strofkabot.discord_db import Prediction

        today = datetime.date.today()
        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=today,
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        manager.db.get_due_predictions = AsyncMock(return_value=[mock_prediction])

        # Mock time to be midday (12:00)
        mock_now = datetime.datetime.now(datetime.UTC).replace(hour=12)
        with patch("strofkabot.tasks.update_tasks.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.UTC = datetime.UTC
            mock_datetime.timedelta = datetime.timedelta

            with patch.object(manager, "_post_prediction", new_callable=AsyncMock) as mock_post:
                await manager.check_predictions()
                mock_post.assert_called_once()

    async def test_increments_retry_on_failure(self, manager):
        """Test that retry count is incremented on posting failure."""
        from strofkabot.discord_db import Prediction

        yesterday = datetime.datetime.now(datetime.UTC).date() - datetime.timedelta(days=1)
        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=yesterday,
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        manager.db.get_due_predictions = AsyncMock(return_value=[mock_prediction])
        manager.db.increment_prediction_retry = AsyncMock(return_value=1)

        with patch.object(manager, "_post_prediction", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Failed to post")
            await manager.check_predictions()

        manager.db.increment_prediction_retry.assert_called_once_with(1)

    async def test_gives_up_after_max_retries(self, manager):
        """Test that prediction is given up after max retries."""
        from strofkabot.discord_db import Prediction

        yesterday = datetime.datetime.now(datetime.UTC).date() - datetime.timedelta(days=1)
        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=yesterday,
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=4,
        )

        manager.db.get_due_predictions = AsyncMock(return_value=[mock_prediction])
        manager.db.increment_prediction_retry = AsyncMock(return_value=5)

        with patch.object(manager, "_post_prediction", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Failed to post")

            with patch.object(manager.logger, "error") as mock_error:
                await manager.check_predictions(max_retries=5)
                # Should log error about giving up
                assert any("giving up" in str(call).lower() for call in mock_error.call_args_list)


class TestPostPrediction:
    """Tests for _post_prediction method."""

    @pytest.fixture
    def manager(self):
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        message_filter = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.tasks.update_tasks.ImageProcessor"):
            manager = BackgroundTaskManager(bot, db, user_stats, message_filter, logger)
            manager.guild = MagicMock()
            return manager

    async def test_raises_type_error_for_non_prediction(self, manager):
        """Test that TypeError is raised for non-Prediction object."""
        with pytest.raises(TypeError):
            await manager._post_prediction("not a prediction")

    async def test_logs_warning_when_channel_not_found(self, manager):
        """Test that warning is logged when channel is not found."""
        from strofkabot.discord_db import Prediction

        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date.today(),
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        manager.bot.get_channel = MagicMock(return_value=None)

        with patch.object(manager.logger, "warning") as mock_warning:
            await manager._post_prediction(mock_prediction)
            mock_warning.assert_called_once()
            assert "not found" in mock_warning.call_args[0][0]

    async def test_posts_embed_with_reactions(self, manager):
        """Test that prediction is posted with voting reactions."""
        from strofkabot.discord_db import Prediction

        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date.today(),
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        mock_message = MagicMock()
        mock_message.add_reaction = AsyncMock()

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=mock_message)

        manager.bot.get_channel = MagicMock(return_value=mock_channel)
        manager.guild.get_member = MagicMock(return_value=None)

        await manager._post_prediction(mock_prediction)

        mock_channel.send.assert_called_once()
        assert mock_message.add_reaction.call_count == 2

    async def test_uses_member_avatar_when_available(self, manager):
        """Test that member avatar is used when available."""
        from strofkabot.discord_db import Prediction

        mock_prediction = Prediction(
            id=1,
            author_id=12345,
            author_name="TestUser",
            channel_id=67890,
            target_date=datetime.date.today(),
            prediction_text="This will happen",
            created_at=datetime.datetime.now(datetime.UTC),
            posted=False,
            retry_count=0,
        )

        mock_member = MagicMock()
        mock_member.display_name = "CurrentName"
        mock_member.display_avatar = MagicMock()
        mock_member.display_avatar.url = "http://example.com/avatar.png"

        mock_message = MagicMock()
        mock_message.add_reaction = AsyncMock()

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=mock_message)

        manager.bot.get_channel = MagicMock(return_value=mock_channel)
        manager.guild.get_member = MagicMock(return_value=mock_member)

        await manager._post_prediction(mock_prediction)

        # Should have sent embed with member info
        mock_channel.send.assert_called_once()


class AsyncIterator:
    """Helper class to create async iterators for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
