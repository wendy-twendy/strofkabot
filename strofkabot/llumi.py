# llumi.py

import os
import asyncio
import logging
import argparse
from pathlib import Path
import datetime

import discord
from discord.ext import commands, tasks

from discord_db import MessageDatabase, Message
from message_filter import MessageFilter
from artan_quotes import ArtanQuotes
from dotenv import load_dotenv
from user_stats import UserStats

import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np

# Load environment variables
load_dotenv()

# Constants
GUILD_ID = 413619835096924160
UPDATE_INTERVAL_SECONDS = 3600  # Update every hour
DATABASE_FILE_LOCATION = Path(__file__).parent.parent / "data" / "db.sqlite3"
ARTAN_QUOTES_PATH = Path(__file__).parent.parent / "data" / "artan_quotes.yaml"

class LlumiBot(commands.Cog):
    def __init__(self, bot: commands.Bot, db: MessageDatabase, user_stats: UserStats, artan_quotes: ArtanQuotes, logger: logging.Logger):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.artan_quotes = artan_quotes
        self.logger = logger
        self.guild = None
        self.channel = None
        self.react_count_threshold = 4
        self.message_filter = MessageFilter()
        self.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        self.update_interval = UPDATE_INTERVAL_SECONDS

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

        # Start tasks
        self.update_db_task.start()
        self.update_usernames_task.start()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not message.content.strip():
            return

        await self.process_commands(message)

    @commands.command(name='llumi')
    async def send_random_message(self, ctx: commands.Context):
        random_message = await self.db.get_random_message()
        if random_message:
            await ctx.send(random_message.content)
            self.logger.info(f"Sent random message: {random_message.content[:50]}...")
        else:
            self.logger.warning("No random message found in the database.")
            await ctx.send("No messages available at the moment.")

    @commands.command(name='unsubscribe')
    async def send_unsubscribe_response(self, ctx: commands.Context):
        response = "dhe unsubscribe e ki, katolik i karit a orthodox i mutit a shka pidhsome je"
        await ctx.send(response)
        self.logger.info(f"Sent unsubscribe response: {response}")

    @commands.command(name='artan')
    async def send_artan_quote(self, ctx: commands.Context):
        if self.artan_quotes:
            quote = self.artan_quotes.get_random_quote()
            await ctx.send(quote)
            self.logger.info(f"Sent Artan quote: {quote[:50]}...")
        else:
            self.logger.warning("Artan quotes not initialized, couldn't send a quote.")
            await ctx.send("Quote feature is currently unavailable.")

    @commands.command(name='rpm')
    async def send_rpm_stats(self, ctx: commands.Context):
        try:
            end_date = datetime.datetime.now(datetime.timezone.utc)
            start_date = end_date - datetime.timedelta(days=30)
            stats = await self.user_stats.get_stats(start_date, end_date)
            if not stats:
                await ctx.send("No user statistics available for the last 30 days.")
                return

            response = "**Reactions per Message (RPM) Statistics (Last 30 days):**\n```\n"
            response += f"{'User':<20} {'RPM':>5} {'Msgs':>5} {'Reacts':>7}\n"
            response += "-" * 40 + "\n"
            for stat in stats[:10]:  # Show top 10 users
                display_name = stat['username'] or f"User {stat['author_id']}"
                response += f"{display_name[:20]:<20} {stat['avg_reacts']:5.2f} {stat['total_msgs']:5d} {stat['total_reacts']:7d}\n"
            response += "```"
            await ctx.send(response)
            self.logger.info("Sent RPM statistics for the last 30 days.")
        except Exception as e:
            self.logger.exception("Error while sending RPM statistics.")
            await ctx.send("An error occurred while fetching RPM statistics.")
    
    @commands.command(name='ping')
    async def ping(self, ctx: commands.Context):
        await ctx.send('Pong!')
        self.logger.info(f"Responded to ping command from {ctx.author}")

    @commands.command(name='riekt-graph')
    async def send_reaction_graph(self, ctx: commands.Context):
        try:
            reaction_graph = await self.user_stats.get_reaction_graph()
            member_names = [member.display_name for member in self.guild.members]

            # Generate heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(reaction_graph, xticklabels=member_names, yticklabels=member_names, cmap="YlGnBu")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title("Reaction Matrix")

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            file = discord.File(fp=buf, filename='reaction_matrix.png')
            await ctx.send("Reaction Matrix:", file=file)
            self.logger.info("Sent reaction matrix.")
        except Exception as e:
            self.logger.exception("Error while generating or sending reaction graph.")
            await ctx.send("An error occurred while generating the reaction graph.")

    @tasks.loop(seconds=UPDATE_INTERVAL_SECONDS)
    async def update_db_task(self):
        try:
            await self.update_db()
        except Exception as e:
            self.logger.exception("Error during periodic database update.")

    @tasks.loop(hours=24)
    async def update_usernames_task(self):
        try:
            await self.update_usernames()
        except Exception as e:
            self.logger.exception("Error during periodic username update.")

    async def update_db(self):
        if not self.guild:
            self.logger.warning("Guild not set. Skipping database update.")
            return

        current_time = datetime.datetime.now(datetime.timezone.utc)
        scan_until = current_time - datetime.timedelta(days=1)
        total_message_count = 0
        total_reaction_count = 0

        async def process_channel(channel: discord.TextChannel):
            nonlocal total_message_count, total_reaction_count
            last_scanned = await self.db.get_last_scanned_timestamp(channel.id) or datetime.datetime(2017, 1, 1, tzinfo=datetime.timezone.utc)
            self.logger.info(f"Updating channel: {channel.name} (ID: {channel.id})")
            self.logger.info(f"Last scanned timestamp for this channel: {last_scanned}")

            messages_to_insert = []
            async for message in channel.history(after=last_scanned, before=scan_until, limit=None):
                if message.author.id == self.bot.user.id:
                    continue
                if self._get_all_reacts(message) < self.react_count_threshold:
                    continue
                if not self.message_filter.is_valid_message(message.content):
                    continue

                reply_info = self._get_reply_info(message)
                messages_to_insert.append(Message(
                    id=message.id,
                    content=message.content,
                    timestamp=message.created_at,
                    reaction_count=self._get_all_reacts(message),
                    author_id=message.author.id,
                    reply_to_id=reply_info[0],
                    reply_to_author=reply_info[1],
                    reply_to_content=reply_info[2]
                ))

                for reaction in message.reactions:
                    async for user in reaction.users():
                        if user.bot:
                            continue
                        await self.user_stats.update_reaction_stats(user.id, message.author.id)

                total_message_count += 1
                total_reaction_count += self._get_all_reacts(message)

                if len(messages_to_insert) >= 100:
                    await self.db.add_messages(messages_to_insert)
                    messages_to_insert.clear()

            if messages_to_insert:
                await self.db.add_messages(messages_to_insert)
            await self.db.update_last_scanned_timestamp(channel.id, message.created_at if 'message' in locals() else scan_until)

            self.logger.info(f"Channel {channel.name} updated. Added {len(messages_to_insert)} messages.")

        channels = [channel for channel in self.guild.text_channels if channel.permissions_for(self.guild.me).read_messages]
        await asyncio.gather(*(process_channel(channel) for channel in channels))

        self.logger.info(f"Database update completed. Total messages added: {total_message_count}")
        self.logger.info(f"Total reactions processed: {total_reaction_count}")

    def _get_all_reacts(self, message: discord.Message) -> int:
        return sum(reaction.count for reaction in message.reactions)

    def _get_reply_info(self, message: discord.Message):
        reply_to_id = None
        reply_to_author = None
        reply_to_content = None

        if message.reference and message.reference.resolved:
            replied_msg = message.reference.resolved
            reply_to_id = replied_msg.id

            if isinstance(replied_msg, discord.DeletedReferencedMessage):
                reply_to_author = "Deleted User"
                reply_to_content = "Message was deleted"
            else:
                reply_to_author = replied_msg.author.display_name if replied_msg.author else "Unknown User"
                reply_to_content = replied_msg.content if hasattr(replied_msg, 'content') else "Content unavailable"

        return reply_to_id, reply_to_author, reply_to_content

    async def update_usernames(self):
        if not self.guild:
            self.logger.warning("Guild not set. Skipping username update.")
            return

        current_time = datetime.datetime.now(datetime.timezone.utc)
        if (current_time - self.last_username_update).total_seconds() < 86400:
            self.logger.info("Username update skipped (less than 24 hours since last update).")
            return

        self.logger.info(f"Updating usernames for guild: {self.guild.name} (ID: {self.guild.id}). Total members: {self.guild.member_count}")
        for member in self.guild.members:
            await self.user_stats.update_user_mapping(member.id, member.display_name)
        self.last_username_update = current_time
        self.logger.info(f"Finished updating usernames at {current_time.isoformat()}.")

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
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info(f"Database file location: {DATABASE_FILE_LOCATION}")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.reactions = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    # Initialize database and other dependencies
    db = MessageDatabase(DATABASE_FILE_LOCATION)
    user_stats = UserStats(DATABASE_FILE_LOCATION)

    try:
        artan_quotes = ArtanQuotes(ARTAN_QUOTES_PATH)
        logger.info("Artan quotes initialized successfully.")
    except ValueError as error:
        logger.error(f"Error during the initialization of Artan quotes: {error}")
        artan_quotes = None

    # Add the bot cog
    await bot.add_cog(LlumiBot(bot, db, user_stats, artan_quotes, logger))

    # Run the bot
    token = os.getenv("LLUMI_BOT_TOKEN")
    if not token:
        logger.error("LLUMI_BOT_TOKEN not found in environment variables.")
        return

    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.exception("Failed to run the bot.")
    finally:
        await bot.close()
        await db.close()
        await user_stats.close()
        logger.info("Bot shutdown gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")