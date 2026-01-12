"""Tests for LlumiBot main module."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.llumi import LlumiBot, setup_logging


class TestLlumiBotInit:
    """Tests for LlumiBot initialization."""

    def test_init_stores_dependencies(self):
        """Test that all dependencies are stored correctly."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        artan_quotes = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager"):
            cog = LlumiBot(bot, db, user_stats, artan_quotes, None, logger)

        assert cog.bot is bot
        assert cog.db is db
        assert cog.user_stats is user_stats
        assert cog.artan_quotes is artan_quotes
        assert cog.logger is logger
        assert cog.guild is None

    def test_init_creates_task_manager(self):
        """Test that task manager is created."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            MockTaskManager.assert_called_once()
            assert cog.task_manager is not None


class TestCogLoad:
    """Tests for cog_load method."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        bot.add_cog = AsyncMock()
        db = MagicMock()
        db.initialize = AsyncMock()
        user_stats = MagicMock()
        artan_quotes = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager"):
            return LlumiBot(bot, db, user_stats, artan_quotes, None, logger)

    async def test_initializes_database(self, cog):
        """Test that database is initialized."""
        with (
            patch("strofkabot.llumi.EntertainmentCog"),
            patch("strofkabot.llumi.UserStatsCog"),
            patch("strofkabot.llumi.EconomicsCog"),
            patch("strofkabot.llumi.NetworkCog"),
            patch("strofkabot.llumi.AICog"),
        ):
            await cog.cog_load()

        cog.db.initialize.assert_called_once()

    async def test_loads_all_cogs(self, cog):
        """Test that all cogs are loaded."""
        with (
            patch("strofkabot.llumi.EntertainmentCog"),
            patch("strofkabot.llumi.UserStatsCog"),
            patch("strofkabot.llumi.EconomicsCog"),
            patch("strofkabot.llumi.NetworkCog"),
            patch("strofkabot.llumi.AICog"),
            patch("strofkabot.llumi.RAGCog"),
        ):
            await cog.cog_load()

        # Should have added 6 cogs (including RAGCog)
        assert cog.bot.add_cog.call_count == 6


class TestCogUnload:
    """Tests for cog_unload method."""

    async def test_closes_task_manager(self):
        """Test that task manager is closed."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.close = AsyncMock()
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            await cog.cog_unload()

            mock_task_manager.close.assert_called_once()


class TestOnReady:
    """Tests for on_ready listener."""

    @pytest.fixture
    def cog(self):
        bot = MagicMock()
        bot.user = MagicMock()
        bot.user.id = 12345
        bot.get_guild = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            MockTaskManager.return_value = mock_task_manager
            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            # Mock the task loops
            cog.update_db_task = MagicMock()
            cog.update_usernames_task = MagicMock()
            cog.check_predictions_task = MagicMock()
            return cog

    async def test_logs_error_when_guild_not_found(self, cog):
        """Test that error is logged when guild is not found."""
        cog.bot.get_guild = MagicMock(return_value=None)

        with patch.object(cog.logger, "error") as mock_error:
            await cog.on_ready()
            mock_error.assert_called_once()
            assert "not found" in mock_error.call_args[0][0]

    async def test_sets_guild_on_task_manager(self, cog):
        """Test that guild is set on task manager."""
        mock_guild = MagicMock()
        mock_guild.name = "Test Guild"
        cog.bot.get_guild = MagicMock(return_value=mock_guild)

        await cog.on_ready()

        cog.task_manager.set_guild.assert_called_once_with(mock_guild)

    async def test_sets_guild_on_economics_cog(self, cog):
        """Test that guild is set on economics cog."""
        mock_guild = MagicMock()
        mock_guild.name = "Test Guild"
        cog.bot.get_guild = MagicMock(return_value=mock_guild)
        cog._economics_cog = MagicMock()

        await cog.on_ready()

        cog._economics_cog.set_guild.assert_called_once_with(mock_guild)

    async def test_sets_guild_on_network_cog(self, cog):
        """Test that guild is set on network cog."""
        mock_guild = MagicMock()
        mock_guild.name = "Test Guild"
        cog.bot.get_guild = MagicMock(return_value=mock_guild)
        cog._network_cog = MagicMock()

        await cog.on_ready()

        cog._network_cog.set_guild.assert_called_once_with(mock_guild)

    async def test_starts_all_tasks(self, cog):
        """Test that all background tasks are started."""
        mock_guild = MagicMock()
        mock_guild.name = "Test Guild"
        cog.bot.get_guild = MagicMock(return_value=mock_guild)

        await cog.on_ready()

        cog.update_db_task.start.assert_called_once()
        cog.update_usernames_task.start.assert_called_once()
        cog.check_predictions_task.start.assert_called_once()

    async def test_does_not_start_tasks_when_guild_not_found(self, cog):
        """Test that tasks are not started when guild is not found."""
        cog.bot.get_guild = MagicMock(return_value=None)

        await cog.on_ready()

        cog.update_db_task.start.assert_not_called()
        cog.update_usernames_task.start.assert_not_called()


class TestUpdateDbTask:
    """Tests for update_db_task."""

    async def test_calls_task_manager_update_db(self):
        """Test that update_db is called on task manager."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.update_db = AsyncMock()
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            # Call the underlying coroutine directly
            await cog.update_db_task.coro(cog)

            mock_task_manager.update_db.assert_called_once()

    async def test_logs_exception_on_error(self):
        """Test that exceptions are logged."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.update_db = AsyncMock(side_effect=Exception("DB error"))
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)

            with patch.object(cog.logger, "exception") as mock_exception:
                await cog.update_db_task.coro(cog)
                mock_exception.assert_called_once()


class TestUpdateUsernamesTask:
    """Tests for update_usernames_task."""

    async def test_calls_task_manager_update_usernames(self):
        """Test that update_usernames is called on task manager."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.update_usernames = AsyncMock()
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            await cog.update_usernames_task.coro(cog)

            mock_task_manager.update_usernames.assert_called_once()

    async def test_logs_exception_on_error(self):
        """Test that exceptions are logged."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.update_usernames = AsyncMock(side_effect=Exception("Error"))
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)

            with patch.object(cog.logger, "exception") as mock_exception:
                await cog.update_usernames_task.coro(cog)
                mock_exception.assert_called_once()


class TestCheckPredictionsTask:
    """Tests for check_predictions_task."""

    async def test_calls_task_manager_check_predictions(self):
        """Test that check_predictions is called on task manager."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.check_predictions = AsyncMock()
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)
            await cog.check_predictions_task.coro(cog)

            mock_task_manager.check_predictions.assert_called_once()

    async def test_logs_exception_on_error(self):
        """Test that exceptions are logged."""
        bot = MagicMock()
        db = MagicMock()
        user_stats = MagicMock()
        logger = logging.getLogger("test")

        with patch("strofkabot.llumi.BackgroundTaskManager") as MockTaskManager:
            mock_task_manager = MagicMock()
            mock_task_manager.check_predictions = AsyncMock(side_effect=Exception("Error"))
            MockTaskManager.return_value = mock_task_manager

            cog = LlumiBot(bot, db, user_stats, None, None, logger)

            with patch.object(cog.logger, "exception") as mock_exception:
                await cog.check_predictions_task.coro(cog)
                mock_exception.assert_called_once()


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_logger_with_correct_name(self):
        """Test that logger has correct name."""
        logger = setup_logging("INFO")
        assert logger.name == "LlumiBot"

    def test_sets_debug_level(self):
        """Test that DEBUG level is set correctly."""
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_sets_info_level(self):
        """Test that INFO level is set correctly."""
        logger = setup_logging("INFO")
        assert logger.level == logging.INFO

    def test_sets_warning_level(self):
        """Test that WARNING level is set correctly."""
        logger = setup_logging("WARNING")
        assert logger.level == logging.WARNING

    def test_sets_error_level(self):
        """Test that ERROR level is set correctly."""
        logger = setup_logging("ERROR")
        assert logger.level == logging.ERROR

    def test_case_insensitive(self):
        """Test that log level is case insensitive."""
        logger = setup_logging("debug")
        assert logger.level == logging.DEBUG

    def test_has_stream_handler(self):
        """Test that stream handler is added."""
        logger = setup_logging("INFO")
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) >= 1

    def test_defaults_to_info_for_invalid_level(self):
        """Test that invalid level defaults to INFO."""
        logger = setup_logging("INVALID_LEVEL")
        assert logger.level == logging.INFO
