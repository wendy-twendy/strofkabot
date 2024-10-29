# llumi.py

import os
import asyncio
import logging
import argparse
from pathlib import Path
import datetime
import signal
import sys
import re

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
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

from utils import (
    parse_rpm_args,
    send_leaderboard,
    send_personal_stats,
    get_reply_info,
    fetch_inflation_data,
    calculate_monthly_inflation,
    calculate_yearly_inflation,
    create_monthly_inflation_plot,
    create_yearly_inflation_plot,
    adjust_month,
    get_member_names,
    calculate_reaction_percentage,
    determine_figure_size,
    generate_reaction_matrix_plot,
    get_non_bot_member_ids,
    prepare_clustering_data,
    perform_kmeans_clustering,
    generate_cluster_plot,
    get_reaction_trade_data,
    fetch_gdp_data,
    fetch_hdi_data,
    create_gdp_plot,
    create_hdi_plot,
    send_most_liked_stats
)

# Load environment variables
load_dotenv()

# Constants
GUILD_ID = 413619835096924160
UPDATE_INTERVAL_SECONDS = 3600*24  # Update every hour
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
    async def send_rpm_stats(self, ctx: commands.Context, *args):
        self.logger.info(f"RPM command called by {ctx.author} with args: {args}")

        # Parse options and flags
        parsed_args = parse_rpm_args(args)
        least = parsed_args.get('least', False)
        all_users = parsed_args.get('all_users', False)
        leaderboard = parsed_args.get('leaderboard', False)

        try:
            current_date = datetime.datetime.now(datetime.timezone.utc)
            year, month = current_date.year, current_date.month
            self.logger.debug(f"Current date: {current_date}, Year: {year}, Month: {month}")

            if leaderboard:
                await send_leaderboard(ctx, year, month, least, all_users, self.user_stats, self.bot)
            else:
                await send_personal_stats(ctx, year, month, self.user_stats)
        except Exception as e:
            self.logger.exception(f"Error in send_rpm_stats: {str(e)}")
            await ctx.send("An error occurred while fetching RPM statistics. Please try again later.")

    @commands.command(name='inflation')
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
            await ctx.send("An error occurred while fetching inflation statistics. Please try again later.")

    @commands.command(name='riekt-graph')
    async def send_reaction_graph(self, ctx: commands.Context):
        try:
            current_date = datetime.datetime.now(datetime.timezone.utc)
            year, month = current_date.year, current_date.month
            self.logger.info("Generating reaction graph with active users filter.")
            
            # Get active users: at least 30 messages in the past month
            active_users = await self.user_stats.get_active_users(year, month, min_messages=30)
            if not active_users:
                await ctx.send("No active users found for the reaction graph.")
                self.logger.warning("No active users to include in the reaction graph.")
                return

            member_names = await get_member_names(self.guild, active_users)
            if not member_names:
                await ctx.send("No valid members found for the reaction graph.")
                self.logger.warning("No valid member names found for the reaction graph.")
                return

            reaction_graph = await self.user_stats.get_reaction_graph(active_users)
            reaction_graph_percentage = calculate_reaction_percentage(reaction_graph)

            fig_size = determine_figure_size(len(member_names))
            reaction_matrix_plot = generate_reaction_matrix_plot(reaction_graph_percentage, member_names, fig_size)

            file = discord.File(fp=reaction_matrix_plot, filename='reaction_matrix_percentage.png')
            await ctx.send("Reaction Matrix (Active Users):", file=file)
            self.logger.info("Sent reaction matrix with active users.")
        except Exception as e:
            self.logger.exception("Error while generating or sending reaction graph.")
            await ctx.send("An error occurred while generating the reaction graph.")

    @commands.command(name='cluster')
    async def cluster_users(self, ctx: commands.Context):
        try:
            self.logger.info("Generating user clusters based on reactions given and received.")
            member_ids = await get_non_bot_member_ids(self.guild)
            if not member_ids:
                await ctx.send("No members found for clustering.")
                self.logger.warning("No members available for clustering.")
                return

            reactions_data = await self.user_stats.get_reactions_given_received(member_ids)
            if not reactions_data:
                await ctx.send("No reaction data available for clustering.")
                self.logger.warning("No reaction data available for clustering.")
                return

            data, member_names = prepare_clustering_data(self.guild, reactions_data)
            if data.size == 0:
                await ctx.send("Insufficient data for clustering.")
                self.logger.warning("Insufficient data for clustering.")
                return

            labels = perform_kmeans_clustering(data)
            cluster_plot = generate_cluster_plot(data, labels, member_names)

            file = discord.File(fp=cluster_plot, filename='user_clusters.png')
            await ctx.send("User Clusters Based on Reactions:", file=file)
            self.logger.info("Sent user clusters plot.")
        except Exception as e:
            self.logger.exception("Error while generating or sending user clusters.")
            await ctx.send("An error occurred while generating the user clusters.")

    @commands.command(name='trade')
    async def reaction_trade_report(self, ctx: commands.Context, member: discord.Member = None):
        """
        Generate a reaction trade report for a user.
        Usage: !trade [optional: @user]
        """
        try:
            target_user = member or ctx.author
            self.logger.info(f"Generating trade report for user {target_user.id}")
            trade_data = await get_reaction_trade_data(self.user_stats, target_user.id, self.guild)
            
            # Format the report
            report = f"**Reaction Trade Report for {target_user.display_name}**\n"
            report += "*Data from the past 12 months*\n```\n"
            
            # Top export partners (reactions given to others)
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
            
            # Add trade balance status
            status = "SURPLUS" if trade_data['trade_balance'] > 0 else "DEFICIT"
            if trade_data['trade_balance'] == 0:
                status = "NEUTRAL"
            report += f"\nTrade Status: {status}"
            
            report += "```"
            
            await ctx.send(report)
            self.logger.info(f"Trade report sent for user {target_user.id}")
            
        except Exception as e:
            self.logger.exception(f"Error generating trade report: {e}")
            await ctx.send("An error occurred while generating the trade report.")

    @commands.command(name='gdp')
    async def show_server_gdp(self, ctx: commands.Context):
        """Display the server's GDP (total messages per month)."""
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
        except Exception as e:
            self.logger.exception("Error generating GDP plot")
            await ctx.send("An error occurred while generating the GDP plot.")

    @commands.command(name='hdi')
    async def show_server_hdi(self, ctx: commands.Context):
        """Display the server's HDI (quality messages per month)."""
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
        except Exception as e:
            self.logger.exception("Error generating HDI plot")
            await ctx.send("An error occurred while generating the HDI plot.")

    @commands.command(name='most-liked')
    async def show_most_liked(self, ctx: commands.Context):
        """Show the most influential users based on reaction patterns."""
        try:
            current_date = datetime.datetime.now(datetime.timezone.utc)
            await send_most_liked_stats(ctx, current_date.year, current_date.month, 0, self.user_stats, self.bot)
        except Exception as e:
            self.logger.exception("Error in most-liked command")
            await ctx.send("An error occurred while calculating most-liked users.")

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

        channels = [channel for channel in self.guild.text_channels if channel.permissions_for(self.guild.me).read_messages]
        await asyncio.gather(*(self._process_channel(channel, scan_until, 
                                                     lambda msg_count, react_count: self._update_totals(msg_count, react_count)) 
                               for channel in channels))

        self.logger.info(f"Database update completed. Total messages added: {total_message_count}")
        self.logger.info(f"Total reactions processed: {total_reaction_count}")

    async def _process_channel(self, channel, scan_until, update_totals_callback):
        last_scanned = await self.db.get_last_scanned_timestamp(channel.id) or datetime.datetime(2017, 1, 1, tzinfo=datetime.timezone.utc)
        self.logger.info(f"Updating channel: {channel.name} (ID: {channel.id})")
        self.logger.info(f"Last scanned timestamp for this channel: {last_scanned}")

        messages_to_insert = []
        stats_to_update = []
        total_message_count = 0
        total_reaction_count = 0

        async for message in channel.history(after=last_scanned, before=scan_until, limit=None):
            if message.author.id == self.bot.user.id or not message.content.strip():
                continue

            reaction_count = self._get_all_reacts(message)
            stats_to_update.append((message.author.id, reaction_count, message.created_at))

            await self._process_reactions(message)

            if len(stats_to_update) >= 100:
                await self.user_stats.batch_update_stats(stats_to_update)
                stats_to_update.clear()

            if reaction_count < self.react_count_threshold or not self.message_filter.is_valid_message(message.content):
                continue

            reply_info = get_reply_info(message)
            messages_to_insert.append(Message(
                id=message.id,
                content=message.content,
                timestamp=message.created_at,
                reaction_count=reaction_count,
                author_id=message.author.id,
                reply_to_id=reply_info[0],
                reply_to_author=reply_info[1],
                reply_to_content=reply_info[2]
            ))

            total_message_count += 1
            total_reaction_count += reaction_count

            if len(messages_to_insert) >= 100:
                await self.db.add_messages(messages_to_insert)
                messages_to_insert.clear()

        if messages_to_insert:
            await self.db.add_messages(messages_to_insert)
        if stats_to_update:
            await self.user_stats.batch_update_stats(stats_to_update)
        await self.db.update_last_scanned_timestamp(channel.id, message.created_at if 'message' in locals() else scan_until)

        self.logger.info(f"Channel {channel.name} updated. Added {len(messages_to_insert)} messages.")
        update_totals_callback(total_message_count, total_reaction_count)

    async def _process_reactions(self, message):
        for reaction in message.reactions:
            async for user in reaction.users():
                if user.bot:
                    continue
                await self.user_stats.update_reaction_stats(user.id, message.author.id, message.created_at)

    def _update_totals(self, msg_count, react_count):
        self.total_message_count += msg_count
        self.total_reaction_count += react_count

    def _get_all_reacts(self, message: discord.Message) -> int:
        return sum(reaction.count for reaction in message.reactions)

    async def update_usernames(self):
        if not self.guild:
            self.logger.warning("Guild not set. Skipping username update.")
            return

        current_time = datetime.datetime.now(datetime.timezone.utc)
        if (current_time - self.last_username_update).total_seconds() < 86400:
            self.logger.info("Username update skipped (less than 24 hours since last update).")
            return

        self.logger.info(f"Updating usernames for guild: {self.guild.name} (ID: {self.guild.id}). Total members: {self.guild.member_count}")
        
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)

        for member in self.guild.members:
            clean_name = emoji_pattern.sub(r'', member.display_name)
            await self.user_stats.update_user_mapping(member.id, clean_name)
        
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
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
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
        # Set up signal handlers for graceful shutdown
        if os.name != 'nt':  # For Unix-like systems
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        else:  # For Windows
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.exception("Failed to run the bot.")
    finally:
        await shutdown()
        # Ensure all tasks are completed
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop the event loop if it's running
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.stop()
        except RuntimeError:
            # If there's no running event loop, we don't need to stop it
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    finally:
        # Force exit if the program is still running
        sys.exit(0)