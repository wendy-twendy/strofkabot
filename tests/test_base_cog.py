"""Tests for BaseCog class."""

import logging
from unittest.mock import Mock

from strofkabot.cogs.base import BaseCog


class TestBaseCog:
    """Test cases for BaseCog initialization."""

    def test_init_stores_bot(self):
        """Test that bot is stored correctly."""
        bot = Mock()
        db = Mock()
        user_stats = Mock()
        logger = logging.getLogger("test")

        cog = BaseCog(bot, db, user_stats, logger)

        assert cog.bot is bot

    def test_init_stores_db(self):
        """Test that database is stored correctly."""
        bot = Mock()
        db = Mock()
        user_stats = Mock()
        logger = logging.getLogger("test")

        cog = BaseCog(bot, db, user_stats, logger)

        assert cog.db is db

    def test_init_stores_user_stats(self):
        """Test that user_stats is stored correctly."""
        bot = Mock()
        db = Mock()
        user_stats = Mock()
        logger = logging.getLogger("test")

        cog = BaseCog(bot, db, user_stats, logger)

        assert cog.user_stats is user_stats

    def test_init_stores_logger(self):
        """Test that logger is stored correctly."""
        bot = Mock()
        db = Mock()
        user_stats = Mock()
        logger = logging.getLogger("test")

        cog = BaseCog(bot, db, user_stats, logger)

        assert cog.logger is logger
