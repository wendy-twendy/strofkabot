"""LlumiBot - Discord bot for community analytics on the Strofka server."""

import argparse
import asyncio
import logging
import os
import signal
import sys

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.cogs import (
    AICog,
    EconomicsCog,
    EntertainmentCog,
    NetworkCog,
    UserStatsCog,
)
from strofkabot.config import (
    ARTAN_QUOTES_PATH,
    DATABASE_FILE_LOCATION,
    GUILD_ID,
    MEMORIES_DIR,
    UPDATE_INTERVAL_SECONDS,
)
from strofkabot.discord_db import Database
from strofkabot.memory_store import MemoryStore
from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager
from strofkabot.user_stats import UserStats

load_dotenv()


class LlumiBot(commands.Cog):
    """Main cog for LlumiBot - orchestrates other cogs and background tasks."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        user_stats: UserStats,
        artan_quotes: ArtanQuotes | None,
        memory_store: MemoryStore | None,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.artan_quotes = artan_quotes
        self.memory_store = memory_store
        self.logger = logger
        self.guild = None

        self.task_manager = BackgroundTaskManager(
            bot=bot, db=db, user_stats=user_stats, message_filter=MessageFilter(), logger=logger
        )

        # Store references to cogs that need guild updates
        self._economics_cog: EconomicsCog | None = None
        self._network_cog: NetworkCog | None = None

    async def cog_load(self):
        await self.db.initialize()

        # Initialize memory store
        if self.memory_store:
            await self.memory_store.initialize()
            self.logger.info("Memory store initialized")

        # Load all cogs
        entertainment_cog = EntertainmentCog(self.bot, self.db, self.artan_quotes, self.logger)
        await self.bot.add_cog(entertainment_cog)

        user_stats_cog = UserStatsCog(
            self.bot,
            self.user_stats,
            self.db,
            self.logger,
        )
        await self.bot.add_cog(user_stats_cog)

        self._economics_cog = EconomicsCog(self.bot, self.user_stats, None, self.logger)
        await self.bot.add_cog(self._economics_cog)

        self._network_cog = NetworkCog(self.bot, self.db, self.user_stats, None, self.logger)
        await self.bot.add_cog(self._network_cog)

        ai_cog = AICog(self.bot, self.db, self.logger, memory_store=self.memory_store)
        await self.bot.add_cog(ai_cog)

        self.logger.info("All cogs loaded successfully")

    async def cog_unload(self):
        """Clean up resources when cog is unloaded."""
        await self.task_manager.close()

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")
        self.guild = self.bot.get_guild(GUILD_ID)
        if not self.guild:
            self.logger.error(f"Guild with ID {GUILD_ID} not found.")
            return

        self.logger.info(f"Connected to guild: {self.guild.name}")
        self.task_manager.set_guild(self.guild)

        # Update cogs that need guild reference
        if self._economics_cog:
            self._economics_cog.set_guild(self.guild)
        if self._network_cog:
            self._network_cog.set_guild(self.guild)

        self.update_db_task.start()
        self.update_usernames_task.start()
        self.check_predictions_task.start()

    @tasks.loop(seconds=UPDATE_INTERVAL_SECONDS)
    async def update_db_task(self):
        try:
            await self.task_manager.update_db()
        except Exception:
            self.logger.exception("Error during periodic database update.")

    @tasks.loop(hours=24)
    async def update_usernames_task(self):
        try:
            await self.task_manager.update_usernames()
        except Exception:
            self.logger.exception("Error during periodic username update.")

    @tasks.loop(hours=1)
    async def check_predictions_task(self):
        try:
            await self.task_manager.check_predictions()
        except Exception:
            self.logger.exception("Error during prediction check.")


def setup_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger("LlumiBot")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


async def main():
    parser = argparse.ArgumentParser(description="Run the LlumiBot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Starting LlumiBot...")
    logger.info(f"Database file location: {DATABASE_FILE_LOCATION}")

    if not os.getenv("LLUMI_BOT_TOKEN"):
        logger.error("LLUMI_BOT_TOKEN not found in environment variables.")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.reactions = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    db = Database(DATABASE_FILE_LOCATION)
    user_stats = UserStats(db)
    memory_store = MemoryStore(MEMORIES_DIR)

    try:
        artan_quotes = ArtanQuotes(ARTAN_QUOTES_PATH)
        logger.info("Artan quotes initialized successfully.")
    except ValueError as error:
        logger.error(f"Error during the initialization of Artan quotes: {error}")
        artan_quotes = None

    await bot.add_cog(LlumiBot(bot, db, user_stats, artan_quotes, memory_store, logger))

    token = os.getenv("LLUMI_BOT_TOKEN")
    shutdown_event = asyncio.Event()

    async def shutdown(signal_received=None):
        if shutdown_event.is_set():
            return
        shutdown_event.set()

        if signal_received:
            logger.info(f"Received exit signal {signal_received}...")
        logger.info("Closing the bot...")
        await bot.close()
        await db.close()
        logger.info("Bot shutdown gracefully.")

    def signal_handler(sig, frame):
        asyncio.create_task(shutdown(sig))

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))

        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception:
        logger.exception("Failed to run the bot.")
    finally:
        await shutdown()
        pending_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending_tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    finally:
        sys.exit(0)
