"""Base cog class with shared dependencies."""

import logging

from discord.ext import commands

from strofkabot.discord_db import Database
from strofkabot.user_stats import UserStats


class BaseCog(commands.Cog):
    """Base cog providing shared dependencies."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        user_stats: UserStats,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.logger = logger
