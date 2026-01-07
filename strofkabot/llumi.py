"""LlumiBot - Discord bot for community analytics on the Strofka server."""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.config import (
    ARTAN_QUOTES_PATH,
    DATABASE_FILE_LOCATION,
    GUILD_ID,
    UPDATE_INTERVAL_SECONDS,
)
from strofkabot.discord_db import MessageDatabase
from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager
from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    calculate_monthly_inflation,
    calculate_yearly_inflation,
    create_gdp_plot,
    create_hdi_plot,
    create_monthly_inflation_plot,
    create_yearly_inflation_plot,
    fetch_gdp_data,
    fetch_hdi_data,
    fetch_inflation_data,
    get_reaction_trade_data,
    parse_rpm_args,
    send_leaderboard,
    send_most_liked_stats,
    send_personal_stats,
)

load_dotenv()


class LlumiBot(commands.Cog):
    """Main cog for LlumiBot with Discord commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db: MessageDatabase,
        user_stats: UserStats,
        artan_quotes: ArtanQuotes,
        logger: logging.Logger
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.artan_quotes = artan_quotes
        self.logger = logger
        self.guild = None

        self.task_manager = BackgroundTaskManager(
            bot=bot,
            db=db,
            user_stats=user_stats,
            message_filter=MessageFilter(),
            logger=logger
        )

    async def cog_load(self):
        await self.db.initialize()
        await self.user_stats.initialize()

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info(f'Logged in as {self.bot.user} (ID: {self.bot.user.id})')
        self.guild = self.bot.get_guild(GUILD_ID)
        if not self.guild:
            self.logger.error(f"Guild with ID {GUILD_ID} not found.")
            return

        self.logger.info(f"Connected to guild: {self.guild.name}")
        self.task_manager.set_guild(self.guild)
        self.update_db_task.start()
        self.update_usernames_task.start()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.content.strip():
            return
        await self.process_commands(message)

    @commands.command(name='llumi', help='Sends a random message from the database')
    async def send_random_message(self, ctx: commands.Context):
        random_message = await self.db.get_random_message()
        if random_message:
            await ctx.send(random_message.content)
            self.logger.info(f"Sent random message: {random_message.content[:50]}...")
        else:
            self.logger.warning("No random message found in the database.")
            await ctx.send("No messages available at the moment.")

    @commands.command(name='unsubscribe', help='Sends a special message about unsubscribing')
    async def send_unsubscribe_response(self, ctx: commands.Context):
        response = "dhe unsubscribe e ki, katolik i karit a orthodox i mutit a shka pidhsome je"
        await ctx.send(response)
        self.logger.info(f"Sent unsubscribe response: {response}")

    @commands.command(name='artan', help='Sends a random quote from Artan\'s collection')
    async def send_artan_quote(self, ctx: commands.Context):
        if self.artan_quotes:
            quote = self.artan_quotes.get_random_quote()
            await ctx.send(quote)
            self.logger.info(f"Sent Artan quote: {quote[:50]}...")
        else:
            self.logger.warning("Artan quotes not initialized.")
            await ctx.send("Quote feature is currently unavailable.")

    @commands.command(name='rpm', help='Shows reaction statistics for users')
    async def send_rpm_stats(self, ctx: commands.Context, *args):
        self.logger.info(f"RPM command called by {ctx.author} with args: {args}")
        parsed_args = parse_rpm_args(args)

        try:
            current_date = datetime.datetime.now(datetime.UTC)
            year, month = current_date.year, current_date.month

            if parsed_args.get('leaderboard', False):
                await send_leaderboard(
                    ctx, year, month,
                    parsed_args.get('least', False),
                    parsed_args.get('all_users', False),
                    self.user_stats, self.bot
                )
            else:
                await send_personal_stats(ctx, year, month, self.user_stats)
        except Exception as e:
            self.logger.exception(f"Error in send_rpm_stats: {str(e)}")
            await ctx.send("An error occurred while fetching RPM statistics.")

    @commands.command(name='inflation', help='Displays reaction inflation statistics')
    async def send_inflation_stats(self, ctx: commands.Context):
        try:
            self.logger.info("Fetching reaction inflation data")
            monthly_data, yearly_data = await fetch_inflation_data(self.user_stats)

            monthly_inflation = calculate_monthly_inflation(monthly_data)
            yearly_inflation = calculate_yearly_inflation(yearly_data)

            monthly_plot = create_monthly_inflation_plot(monthly_inflation)
            yearly_plot = create_yearly_inflation_plot(yearly_inflation)

            file_monthly = discord.File(fp=monthly_plot, filename='monthly_inflation.png')
            file_yearly = discord.File(fp=yearly_plot, filename='yearly_inflation.png')

            await ctx.send("**Reaction Inflation Overviews:**", files=[file_monthly, file_yearly])
            self.logger.info("Sent reaction inflation plots.")
        except Exception as e:
            self.logger.exception(f"Error in send_inflation_stats: {str(e)}")
            await ctx.send("An error occurred while fetching inflation statistics.")

    @commands.command(name='trade', help='Shows reaction trading statistics for a user')
    async def reaction_trade_report(self, ctx: commands.Context, member: discord.Member = None):
        try:
            target_user = member or ctx.author
            self.logger.info(f"Generating trade report for user {target_user.id}")
            trade_data = await get_reaction_trade_data(self.user_stats, target_user.id, self.guild)

            report = f"**Reaction Trade Report for {target_user.display_name}**\n"
            report += "*Data from the past 12 months*\n```\n"

            report += "Top Export Partners (Reactions Given):\n"
            if trade_data['exports']:
                for partner, count in trade_data['exports']:
                    report += f"  {partner:<20} {count:>6}\n"
            else:
                report += "  No reactions given\n"

            report += "\nTop Import Partners (Reactions Received):\n"
            if trade_data['imports']:
                for partner, count in trade_data['imports']:
                    report += f"  {partner:<20} {count:>6}\n"
            else:
                report += "  No reactions received\n"

            report += "\nTrade Summary:\n"
            report += f"  Total Reactions Given:    {trade_data['total_given']:>6}\n"
            report += f"  Total Reactions Received: {trade_data['total_received']:>6}\n"
            report += f"  Trade Balance:            {trade_data['trade_balance']:>6}\n"

            status = "SURPLUS" if trade_data['trade_balance'] > 0 else \
                     "DEFICIT" if trade_data['trade_balance'] < 0 else "NEUTRAL"
            report += f"\nTrade Status: {status}```"

            await ctx.send(report)
            self.logger.info(f"Trade report sent for user {target_user.id}")
        except Exception as e:
            self.logger.exception(f"Error generating trade report: {e}")
            await ctx.send("An error occurred while generating the trade report.")

    @commands.command(name='gdp', help='Displays the server\'s GDP over time')
    async def show_server_gdp(self, ctx: commands.Context):
        try:
            self.logger.info("Generating GDP plot")
            gdp_data = await fetch_gdp_data(self.user_stats)

            if not gdp_data:
                await ctx.send("No message data available for GDP calculation.")
                return

            plot = create_gdp_plot(gdp_data)
            file = discord.File(fp=plot, filename='server_gdp.png')
            await ctx.send("**Server GDP (Total Messages per Month)**", file=file)
            self.logger.info("GDP plot sent successfully")
        except Exception:
            self.logger.exception("Error generating GDP plot")
            await ctx.send("An error occurred while generating the GDP plot.")

    @commands.command(name='hdi', help='Shows the server\'s HDI over time')
    async def show_server_hdi(self, ctx: commands.Context):
        try:
            self.logger.info("Generating HDI plot")
            hdi_data = await fetch_hdi_data(self.user_stats)

            if not hdi_data:
                await ctx.send("No message data available for HDI calculation.")
                return

            plot = create_hdi_plot(hdi_data)
            file = discord.File(fp=plot, filename='server_hdi.png')
            await ctx.send("**Server HDI (Quality Messages per Month)**", file=file)
            self.logger.info("HDI plot sent successfully")
        except Exception:
            self.logger.exception("Error generating HDI plot")
            await ctx.send("An error occurred while generating the HDI plot.")

    @commands.command(name='most-liked', help='Shows the most influential users')
    async def show_most_liked(self, ctx: commands.Context):
        try:
            current_date = datetime.datetime.now(datetime.UTC)
            await send_most_liked_stats(
                ctx, current_date.year, current_date.month, 0,
                self.user_stats, self.bot
            )
            self.logger.info(f"Most-liked stats requested by {ctx.author}")
        except Exception:
            self.logger.exception("Error in most-liked command")
            await ctx.send("An error occurred while calculating most-liked users.")

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


def setup_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger('LlumiBot')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


async def main():
    parser = argparse.ArgumentParser(description='Run the LlumiBot')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
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
    bot = commands.Bot(command_prefix='!', intents=intents)

    db = MessageDatabase(DATABASE_FILE_LOCATION)
    user_stats = UserStats(DATABASE_FILE_LOCATION)

    try:
        artan_quotes = ArtanQuotes(ARTAN_QUOTES_PATH)
        logger.info("Artan quotes initialized successfully.")
    except ValueError as error:
        logger.error(f"Error during the initialization of Artan quotes: {error}")
        artan_quotes = None

    await bot.add_cog(LlumiBot(bot, db, user_stats, artan_quotes, logger))

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
        await user_stats.close()
        logger.info("Bot shutdown gracefully.")

    def signal_handler(sig, frame):
        asyncio.create_task(shutdown(sig))

    try:
        if os.name != 'nt':
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        else:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

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
