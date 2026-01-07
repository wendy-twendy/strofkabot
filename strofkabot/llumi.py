"""LlumiBot - Discord bot for community analytics on the Strofka server."""

import argparse
import asyncio
import datetime
import logging
import os
import random
import re
import signal
import sys

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.config import (
    ARTAN_QUOTES_PATH,
    ATTACHMENTS_DIR,
    DATABASE_FILE_LOCATION,
    GUILD_ID,
    UPDATE_INTERVAL_SECONDS,
)
from strofkabot.discord_db import Database
from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager
from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    adjust_month,
    build_affinity_graph,
    build_reaction_graph,
    calculate_monthly_inflation,
    calculate_yearly_inflation,
    compute_all_affinities,
    compute_community_layout,
    compute_node_activity,
    create_activity_heatmap,
    create_gdp_plot,
    create_hdi_plot,
    create_monthly_inflation_plot,
    create_reaction_graph_plot,
    create_yearly_inflation_plot,
    detect_communities,
    fetch_gdp_data,
    fetch_hdi_data,
    fetch_hourly_activity_data,
    fetch_inflation_data,
    format_period_string,
    get_reaction_trade_data,
    get_reaction_trade_data_for_month,
    get_rolling_start_month,
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
        db: Database,
        user_stats: UserStats,
        artan_quotes: ArtanQuotes,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.artan_quotes = artan_quotes
        self.logger = logger
        self.guild = None

        self.task_manager = BackgroundTaskManager(
            bot=bot, db=db, user_stats=user_stats, message_filter=MessageFilter(), logger=logger
        )

    async def cog_load(self):
        await self.db.initialize()

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
        self.update_db_task.start()
        self.update_usernames_task.start()
        self.check_predictions_task.start()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.content.strip():
            return
        await self.process_commands(message)

    @commands.command(
        name="llumi", help="Sends a random message or image. Use -i or --image to force an image."
    )
    async def send_random_message(self, ctx: commands.Context, *, args: str = ""):
        force_image = args.strip() in ("-i", "--image")

        msg_count = await self.db.get_message_count()
        att_count = await self.db.get_attachment_count()

        if force_image:
            if att_count == 0:
                await ctx.send("No images available.")
                return
            attachment = await self.db.get_random_attachment()
            await self._send_attachment(ctx, attachment)
        else:
            total = msg_count + att_count
            if total == 0:
                self.logger.warning("No messages or attachments found in the database.")
                await ctx.send("No messages available at the moment.")
                return

            # Easter egg: 1/50 chance (2%)
            if random.randint(1, 50) == 1:
                await ctx.send("Ik qiu Jordi")
                self.logger.info("Sent Easter egg message: Ik qiu Jordi")
                return

            # 50% chance for image (if available), otherwise message
            if att_count > 0 and random.random() < 0.5:
                attachment = await self.db.get_random_attachment()
                await self._send_attachment(ctx, attachment)
            else:
                random_message = await self.db.get_random_message()
                if random_message:
                    await ctx.send(random_message.content)
                    self.logger.info(f"Sent random message: {random_message.content[:50]}...")

    async def _send_attachment(self, ctx: commands.Context, attachment):
        """Helper method to send an attachment with optional message content."""
        file_path = ATTACHMENTS_DIR / attachment.local_path
        if not file_path.exists():
            self.logger.warning(f"Attachment file not found: {file_path}")
            await ctx.send("Could not find the image file.")
            return

        file = discord.File(file_path)
        content = attachment.message_content if attachment.message_content else None
        await ctx.send(content=content, file=file)
        self.logger.info(f"Sent attachment: {attachment.original_filename}")

    @commands.command(name="unsubscribe", help="Sends a special message about unsubscribing")
    async def send_unsubscribe_response(self, ctx: commands.Context):
        response = "dhe unsubscribe e ki, katolik i karit a orthodox i mutit a shka pidhsome je"
        await ctx.send(response)
        self.logger.info(f"Sent unsubscribe response: {response}")

    @commands.command(name="artan", help="Sends a random quote from Artan's collection")
    async def send_artan_quote(self, ctx: commands.Context):
        if self.artan_quotes:
            quote = self.artan_quotes.get_random_quote()
            await ctx.send(quote)
            self.logger.info(f"Sent Artan quote: {quote[:50]}...")
        else:
            self.logger.warning("Artan quotes not initialized.")
            await ctx.send("Quote feature is currently unavailable.")

    @commands.command(
        name="rpm",
        help="Shows reaction stats. Use --leaderboard for rankings, --least for lowest, --all for all users.",
    )
    async def send_rpm_stats(self, ctx: commands.Context, *args):
        self.logger.info(f"RPM command called by {ctx.author} with args: {args}")
        parsed_args = parse_rpm_args(args)

        try:
            current_date = datetime.datetime.now(datetime.UTC)
            year, month = current_date.year, current_date.month

            if parsed_args.get("leaderboard", False):
                await send_leaderboard(
                    ctx,
                    year,
                    month,
                    parsed_args.get("least", False),
                    parsed_args.get("all_users", False),
                    self.user_stats,
                    self.bot,
                )
            else:
                await send_personal_stats(ctx, year, month, self.user_stats)
        except Exception as e:
            self.logger.exception(f"Error in send_rpm_stats: {str(e)}")
            await ctx.send("An error occurred while fetching RPM statistics.")

    @commands.command(name="inflation", help="Displays reaction inflation statistics")
    async def send_inflation_stats(self, ctx: commands.Context):
        try:
            self.logger.info("Fetching reaction inflation data")
            monthly_data, yearly_data = await fetch_inflation_data(self.user_stats)

            monthly_inflation = calculate_monthly_inflation(monthly_data)
            yearly_inflation = calculate_yearly_inflation(yearly_data)

            monthly_plot = create_monthly_inflation_plot(monthly_inflation)
            yearly_plot = create_yearly_inflation_plot(yearly_inflation)

            file_monthly = discord.File(fp=monthly_plot, filename="monthly_inflation.png")
            file_yearly = discord.File(fp=yearly_plot, filename="yearly_inflation.png")

            await ctx.send("**Reaction Inflation Overviews:**", files=[file_monthly, file_yearly])
            self.logger.info("Sent reaction inflation plots.")
        except Exception as e:
            self.logger.exception(f"Error in send_inflation_stats: {str(e)}")
            await ctx.send("An error occurred while fetching inflation statistics.")

    @commands.command(
        name="trade", help="Shows reaction trading statistics. Use --yearly for 12-month data."
    )
    async def reaction_trade_report(self, ctx: commands.Context, *, args: str = ""):
        """Show reaction trade report for a user.

        Default: Shows single-month data with arrow navigation.
        --yearly: Shows rolling 12-month data (original behavior).
        @member: Optionally specify a member.
        """
        # Parse args for --yearly flag and member mention
        yearly_mode = "--yearly" in args.lower()

        # Extract member mention from args
        member = ctx.message.mentions[0] if ctx.message.mentions else None
        target_user = member or ctx.author

        if yearly_mode:
            await self._send_yearly_trade(ctx, target_user)
        else:
            await self._send_monthly_trade(ctx, target_user, month_offset=0)

    async def _send_yearly_trade(self, ctx: commands.Context, target_user: discord.Member):
        """Send yearly (rolling 12-month) trade report."""
        try:
            self.logger.info(f"Generating yearly trade report for user {target_user.id}")
            trade_data = await get_reaction_trade_data(self.user_stats, target_user.id, self.guild)

            report = f"**Reaction Trade Report for {target_user.display_name}**\n"
            report += "*Data from the past 12 months*\n```\n"

            report += "Top Export Partners (Reactions Given):\n"
            if trade_data["exports"]:
                for partner, count in trade_data["exports"]:
                    report += f"  {partner:<20} {count:>6}\n"
            else:
                report += "  No reactions given\n"

            report += "\nTop Import Partners (Reactions Received):\n"
            if trade_data["imports"]:
                for partner, count in trade_data["imports"]:
                    report += f"  {partner:<20} {count:>6}\n"
            else:
                report += "  No reactions received\n"

            report += "\nTrade Summary:\n"
            report += f"  Total Reactions Given:    {trade_data['total_given']:>6}\n"
            report += f"  Total Reactions Received: {trade_data['total_received']:>6}\n"
            report += f"  Trade Balance:            {trade_data['trade_balance']:>6}\n"

            status = (
                "SURPLUS"
                if trade_data["trade_balance"] > 0
                else "DEFICIT"
                if trade_data["trade_balance"] < 0
                else "NEUTRAL"
            )
            report += f"\nTrade Status: {status}```"

            await ctx.send(report)
            self.logger.info(f"Yearly trade report sent for user {target_user.id}")
        except Exception as e:
            self.logger.exception(f"Error generating yearly trade report: {e}")
            await ctx.send("An error occurred while generating the trade report.")

    async def _send_monthly_trade(
        self, ctx: commands.Context, target_user: discord.Member, month_offset: int
    ):
        """Send monthly trade report with navigation reactions."""
        try:
            # Get target month
            now = datetime.datetime.now(datetime.UTC)
            target_year, target_month = adjust_month(now.year, now.month, month_offset)

            self.logger.info(
                f"Generating monthly trade report for user {target_user.id} "
                f"({target_year}-{target_month:02d})"
            )

            trade_data = await get_reaction_trade_data_for_month(
                self.user_stats, target_user.id, target_year, target_month, self.guild
            )

            # Format period string
            period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")

            report = f"**Reaction Trade Report for {target_user.display_name}**\n"
            report += f"*{period_str}*\n```\n"

            # Check if there's any data
            has_data = trade_data["total_given"] > 0 or trade_data["total_received"] > 0

            if not has_data:
                report += "No trading activity this month.\n"
            else:
                report += "Top Export Partners (Reactions Given):\n"
                if trade_data["exports"]:
                    for partner, count in trade_data["exports"]:
                        report += f"  {partner:<20} {count:>6}\n"
                else:
                    report += "  No reactions given\n"

                report += "\nTop Import Partners (Reactions Received):\n"
                if trade_data["imports"]:
                    for partner, count in trade_data["imports"]:
                        report += f"  {partner:<20} {count:>6}\n"
                else:
                    report += "  No reactions received\n"

                report += "\nTrade Summary:\n"
                report += f"  Total Reactions Given:    {trade_data['total_given']:>6}\n"
                report += f"  Total Reactions Received: {trade_data['total_received']:>6}\n"
                report += f"  Trade Balance:            {trade_data['trade_balance']:>6}\n"

                status = (
                    "SURPLUS"
                    if trade_data["trade_balance"] > 0
                    else "DEFICIT"
                    if trade_data["trade_balance"] < 0
                    else "NEUTRAL"
                )
                report += f"\nTrade Status: {status}"

            report += "```"

            # Send message and add navigation reactions
            message = await ctx.send(report)
            await message.add_reaction("⬅️")
            await message.add_reaction("➡️")

            self.logger.info(f"Monthly trade report sent for user {target_user.id}")

            # Wait for navigation reactions
            def check(reaction, user):
                return (
                    reaction.message.id == message.id
                    and str(reaction.emoji) in ["⬅️", "➡️"]
                    and not user.bot
                )

            while True:
                try:
                    reaction, user = await self.bot.wait_for(
                        "reaction_add", timeout=60.0, check=check
                    )
                    new_offset = month_offset
                    if str(reaction.emoji) == "⬅️":
                        new_offset -= 1
                    elif str(reaction.emoji) == "➡️":
                        new_offset += 1
                    await message.delete()
                    await self._send_monthly_trade(ctx, target_user, new_offset)
                    break
                except TimeoutError:
                    break
        except Exception as e:
            self.logger.exception(f"Error generating monthly trade report: {e}")
            await ctx.send("An error occurred while generating the trade report.")

    @commands.command(
        name="gdp", help="Displays the server's GDP over time. Use --all for full history."
    )
    async def show_server_gdp(self, ctx: commands.Context, *, args: str = ""):
        try:
            show_all = "--all" in args.lower()
            self.logger.info(f"Generating GDP plot (show_all={show_all})")
            gdp_data = await fetch_gdp_data(self.user_stats, show_all=show_all)

            if not gdp_data:
                await ctx.send("No message data available for GDP calculation.")
                return

            plot = create_gdp_plot(gdp_data)
            file = discord.File(fp=plot, filename="server_gdp.png")
            await ctx.send("**Server GDP (Total Messages per Month)**", file=file)
            self.logger.info("GDP plot sent successfully")
        except Exception:
            self.logger.exception("Error generating GDP plot")
            await ctx.send("An error occurred while generating the GDP plot.")

    @commands.command(name="hdi", help="Shows the server's HDI over time")
    async def show_server_hdi(self, ctx: commands.Context):
        try:
            self.logger.info("Generating HDI plot")
            hdi_data = await fetch_hdi_data(self.user_stats)

            if not hdi_data:
                await ctx.send("No message data available for HDI calculation.")
                return

            plot = create_hdi_plot(hdi_data)
            file = discord.File(fp=plot, filename="server_hdi.png")
            await ctx.send("**Server HDI (Quality Messages per Month)**", file=file)
            self.logger.info("HDI plot sent successfully")
        except Exception:
            self.logger.exception("Error generating HDI plot")
            await ctx.send("An error occurred while generating the HDI plot.")

    @commands.command(
        name="most-liked", help="Shows the most liked users. Use --all to show all users."
    )
    async def show_most_liked(self, ctx: commands.Context, *, args: str = ""):
        try:
            show_all = "--all" in args.lower()
            current_date = datetime.datetime.now(datetime.UTC)
            await send_most_liked_stats(
                ctx, current_date.year, current_date.month, 0, self.user_stats, self.bot, show_all
            )
            self.logger.info(f"Most-liked stats requested by {ctx.author} (show_all={show_all})")
        except Exception:
            self.logger.exception("Error in most-liked command")
            await ctx.send("An error occurred while calculating most-liked users.")

    @commands.command(
        name="riekt-graph", help="Shows reaction network graph. Usage: !riekt-graph [months]"
    )
    async def show_riekt_graph(self, ctx: commands.Context, months: int = 1):
        try:
            self.logger.info(f"Generating riekt graph for {months} month(s)")

            # Validate months parameter
            if months < 1:
                await ctx.send("Number of months must be at least 1.")
                return
            if months > 12:
                await ctx.send("Number of months must be at most 12.")
                return

            # Calculate rolling window start
            start_year, start_month = get_rolling_start_month(months)

            # Fetch data
            reaction_data = await self.user_stats.get_reaction_network_rolling(
                start_year, start_month
            )

            if not reaction_data:
                await ctx.send("No reaction data available for this period.")
                return

            # Build graph and compute affinities
            directed_graph = build_reaction_graph(reaction_data)
            affinities = compute_all_affinities(directed_graph)

            # Build affinity graph with threshold
            min_affinity_threshold = 0.03
            affinity_graph = build_affinity_graph(
                directed_graph,
                affinities,
                min_affinity=min_affinity_threshold,
                remove_isolated=True,
            )

            # Find isolated nodes
            all_nodes = set(directed_graph.nodes())
            connected_nodes = set(affinity_graph.nodes())
            isolated_nodes = sorted(all_nodes - connected_nodes)

            # Detect communities and compute layout
            communities = detect_communities(directed_graph)
            activity = compute_node_activity(directed_graph)
            positions = compute_community_layout(
                affinity_graph, communities, isolated_nodes=isolated_nodes
            )

            # Add isolated nodes back to graph for drawing
            for node in isolated_nodes:
                affinity_graph.add_node(node)

            # Generate plot
            period_str = format_period_string(start_year, start_month, months)
            plot = create_reaction_graph_plot(
                affinity_graph, positions, communities, activity, period_str
            )

            file = discord.File(fp=plot, filename="reaction_graph.png")
            await ctx.send("**Reaction Network Graph**", file=file)
            self.logger.info("Riekt graph sent successfully")
        except Exception:
            self.logger.exception("Error generating riekt graph")
            await ctx.send("An error occurred while generating the graph.")

    @commands.command(
        name="connections", help="Shows top 10 mutual relationships with month navigation."
    )
    async def show_connections(self, ctx: commands.Context):
        """Show top 10 mutual relationships for the current month with navigation."""
        await self._send_connections(ctx, month_offset=0)

    async def _send_connections(self, ctx: commands.Context, month_offset: int):
        """Send connections message with navigation reactions."""
        try:
            # Get target month
            now = datetime.datetime.now(datetime.UTC)
            target_year, target_month = adjust_month(now.year, now.month, month_offset)

            self.logger.info(f"Generating connections for {target_year}-{target_month:02d}")

            # Fetch single month data
            rows = await self.user_stats.get_reaction_network_for_month(target_year, target_month)

            if not rows:
                period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")
                await ctx.send(f"No reaction data available for {period_str}.")
                return

            # Convert to tuples for graph functions
            reaction_data = [
                (r["giver_username"], r["receiver_username"], r["reaction_count"]) for r in rows
            ]

            # Build graph and compute affinities
            directed_graph = build_reaction_graph(reaction_data)
            affinities = compute_all_affinities(directed_graph)

            if not affinities:
                period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")
                await ctx.send(f"No mutual connections found for {period_str}.")
                return

            # Format period string
            period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")

            # Build response with top 10
            response = f"**Top 10 Mutual Relationships** ({period_str})\n"
            response += "*Mutual Affinity Score*\n```\n"

            for i, (user_a, user_b, affinity) in enumerate(affinities[:10], 1):
                response += f"{i:2}. {user_a} <-> {user_b}: {affinity:.3f}\n"

            response += "```"

            # Send message and add navigation reactions
            message = await ctx.send(response)
            await message.add_reaction("⬅️")
            await message.add_reaction("➡️")

            self.logger.info("Connections sent successfully")

            # Wait for navigation reactions
            def check(reaction, user):
                return (
                    reaction.message.id == message.id
                    and str(reaction.emoji) in ["⬅️", "➡️"]
                    and not user.bot
                )

            while True:
                try:
                    reaction, user = await self.bot.wait_for(
                        "reaction_add", timeout=60.0, check=check
                    )
                    new_offset = month_offset
                    if str(reaction.emoji) == "⬅️":
                        new_offset -= 1
                    elif str(reaction.emoji) == "➡️":
                        new_offset += 1
                    await message.delete()
                    await self._send_connections(ctx, new_offset)
                    break
                except TimeoutError:
                    break
        except Exception:
            self.logger.exception("Error generating connections")
            await ctx.send("An error occurred while generating the connections.")

    @commands.command(name="cluster", help="Shows social clusters. Usage: !cluster [months]")
    async def show_clusters(self, ctx: commands.Context, months: int = 1):
        """Show social clusters using Louvain community detection."""
        try:
            self.logger.info(f"Generating clusters for {months} month(s)")

            # Validate months parameter
            if months < 1:
                await ctx.send("Number of months must be at least 1.")
                return
            if months > 12:
                await ctx.send("Number of months must be at most 12.")
                return

            # Calculate rolling window start
            start_year, start_month = get_rolling_start_month(months)

            # Fetch data
            reaction_data = await self.user_stats.get_reaction_network_rolling(
                start_year, start_month
            )

            if not reaction_data:
                await ctx.send("No reaction data available for this period.")
                return

            # Build graph
            directed_graph = build_reaction_graph(reaction_data)

            # Detect communities with resolution=1.5
            communities = detect_communities(directed_graph, resolution=1.5)

            if not communities:
                await ctx.send("No communities detected for this period.")
                return

            # Group members by community
            community_groups: dict[int, list[str]] = {}
            for member, comm_id in communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(member)

            # Sort groups by size (largest first) and members alphabetically
            sorted_groups = sorted(community_groups.items(), key=lambda x: (-len(x[1]), x[0]))

            # Format period string
            period_str = format_period_string(start_year, start_month, months)

            # Build response
            response = f"**Social Clusters** ({period_str})\n"
            response += f"*{len(sorted_groups)} communities detected*\n```\n"

            for comm_id, members in sorted_groups:
                members_sorted = sorted(members)
                response += f"Group {comm_id + 1} ({len(members)}): {', '.join(members_sorted)}\n"

            response += "```"

            await ctx.send(response)
            self.logger.info("Clusters sent successfully")
        except Exception:
            self.logger.exception("Error generating clusters")
            await ctx.send("An error occurred while generating clusters.")

    @commands.command(
        name="on-this-day",
        aliases=["otd"],
        help="Shows a memorable message from this day in a previous year.",
    )
    async def on_this_day(self, ctx: commands.Context):
        """Show the highest-reacted message from this day in a randomly selected past year."""
        try:
            today = datetime.datetime.now(datetime.UTC)
            current_month = today.month
            current_day = today.day
            current_year = today.year

            # Get years with messages on this day
            years = await self.db.get_on_this_day_years(current_month, current_day)

            # Filter out current year (we only want past years)
            past_years = [y for y in years if y < current_year]

            if not past_years:
                date_str = today.strftime("%B %d")
                await ctx.send(
                    f"No historical messages found for {date_str}. "
                    "Check back as the archive grows!"
                )
                return

            # Randomly select one year
            selected_year = random.choice(past_years)

            # Get the top message/attachment from that day
            message, attachment = await self.db.get_top_message_on_this_day(
                selected_year, current_month, current_day
            )

            if not message and not attachment:
                await ctx.send("No content found for this day. Please try again later.")
                self.logger.warning(
                    f"Year {selected_year} returned for on-this-day but no content found"
                )
                return

            # Get author info
            author_id = attachment.author_id if attachment else message.author_id
            username = await self.db.get_username_by_id(author_id)
            if not username:
                username = "Unknown"

            # Format the header
            years_ago = current_year - selected_year
            years_text = "year" if years_ago == 1 else "years"
            date_str = f"{current_month}/{current_day}/{selected_year}"

            if attachment:
                header = f"**On This Day** ({years_ago} {years_text} ago - {date_str})\n"
                header += f"*{attachment.reaction_count} reactions*"

                file_path = ATTACHMENTS_DIR / attachment.local_path
                if not file_path.exists():
                    self.logger.warning(f"Attachment file not found: {file_path}")
                    await ctx.send("Could not find the historical image file.")
                    return

                file = discord.File(file_path)
                content = header + f"\n\n**{username}**"
                if attachment.message_content:
                    content += f"\n{attachment.message_content}"
                await ctx.send(content=content, file=file)
                self.logger.info(
                    f"Sent on-this-day attachment from {date_str}: {attachment.original_filename}"
                )
            else:
                response = f"**On This Day** ({years_ago} {years_text} ago - {date_str})\n"
                response += f"*{message.reaction_count} reactions*\n\n"
                response += f"**{username}**\n{message.content}"
                await ctx.send(response)
                self.logger.info(
                    f"Sent on-this-day message from {date_str}: {message.content[:50]}..."
                )
        except Exception:
            self.logger.exception("Error in on-this-day command")
            await ctx.send("An error occurred while fetching historical content.")

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

    @commands.command(
        name="predict",
        help="Make a prediction for a future date. Formats: DD-MM-YYYY, 'tomorrow', 'next week', 'January 15'",
    )
    async def make_prediction(self, ctx: commands.Context, *, args: str = ""):
        """Store a prediction to be posted on the specified future date."""
        if not args.strip():
            await ctx.send(
                "**Usage:** `!predict <date> <prediction text>`\n"
                "**Date formats:** DD-MM-YYYY (e.g. 25-12-2025), 'tomorrow', 'next week', 'January 15'\n\n"
                "**Examples:**\n"
                "• `!predict tomorrow The weather will be sunny`\n"
                "• `!predict 25-12-2025 Christmas will be white`\n"
                "• `!predict next week I will finish this project`"
            )
            return

        # Parse date from args
        parsed_date, prediction_text = self._parse_prediction_args(args)

        if not parsed_date:
            await ctx.send(
                "I couldn't understand that date. Try formats like DD-MM-YYYY "
                "(e.g. 25-12-2025), 'tomorrow', 'next week', or 'January 15'."
            )
            return

        if not prediction_text.strip():
            await ctx.send("Please provide some prediction text after the date.")
            return

        # Validate date is in future
        today = datetime.datetime.now(datetime.UTC).date()
        if parsed_date.date() <= today:
            await ctx.send("That date is in the past! Please provide a future date.")
            return

        # Validate date isn't too far (max 5 years)
        max_date = today + datetime.timedelta(days=365 * 5)
        if parsed_date.date() > max_date:
            await ctx.send(
                "That's quite far in the future! Maximum prediction date is 5 years from now."
            )
            return

        # Store prediction
        try:
            prediction_id = await self.db.add_prediction(
                author_id=ctx.author.id,
                author_name=ctx.author.display_name,
                channel_id=ctx.channel.id,
                target_date=parsed_date.date(),
                prediction_text=prediction_text.strip(),
            )

            # Confirmation with parsed date
            formatted_date = parsed_date.strftime("%B %d, %Y")
            embed = discord.Embed(
                title="Prediction Recorded!",
                description=prediction_text.strip(),
                color=discord.Color.blue(),
            )
            embed.add_field(name="Will be posted on", value=formatted_date, inline=False)
            embed.set_footer(text=f"Prediction ID: {prediction_id}")

            await ctx.send(embed=embed)
            self.logger.info(
                f"Prediction #{prediction_id} created by {ctx.author} for {formatted_date}"
            )
        except Exception:
            self.logger.exception("Error storing prediction")
            await ctx.send("An error occurred while storing your prediction.")

    def _parse_prediction_args(self, args: str) -> tuple:
        """Parse date and text from prediction args.

        Returns tuple of (datetime or None, remaining_text).
        """
        import dateparser

        words = args.split()
        parsed_date = None
        date_word_count = 0

        # Try progressively longer prefixes as dates (up to 4 words)
        for i in range(1, min(len(words) + 1, 5)):
            candidate = " ".join(words[:i])
            result = dateparser.parse(
                candidate,
                settings={
                    "PREFER_DATES_FROM": "future",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "TIMEZONE": "UTC",
                    "DATE_ORDER": "DMY",
                },
            )
            if result:
                parsed_date = result
                date_word_count = i

        prediction_text = " ".join(words[date_word_count:]) if date_word_count > 0 else ""
        return parsed_date, prediction_text

    @commands.command(
        name="activity",
        help="Shows activity heatmap. Usage: !activity [@user] [--tz OFFSET]",
    )
    async def show_activity_heatmap(self, ctx: commands.Context, *, args: str = ""):
        """Display an hourly activity heatmap for a user.

        Optional arguments:
            @user: Mention a user to see their heatmap (default: self)
            --tz OFFSET: Timezone offset from UTC (e.g., --tz +1 for CET, --tz -5 for EST)
        """
        message_history_db = self.task_manager.message_history_db
        if not message_history_db:
            await ctx.send("Activity data is not available.")
            return

        try:
            # Parse target user (mentioned or self)
            target_user = ctx.message.mentions[0] if ctx.message.mentions else ctx.author

            # Parse timezone offset
            timezone_offset = 0
            timezone_label = "UTC"
            tz_match = re.search(r"--tz\s*([+-]?\d+)", args.lower())
            if tz_match:
                timezone_offset = int(tz_match.group(1))
                timezone_offset = max(-12, min(14, timezone_offset))  # Clamp to valid range
                sign = "+" if timezone_offset >= 0 else ""
                timezone_label = f"UTC{sign}{timezone_offset}"

            self.logger.info(
                f"Generating activity heatmap for user {target_user.id} "
                f"(tz_offset={timezone_offset})"
            )

            # Ensure database is initialized
            await message_history_db.ensure_connection()

            # Fetch data
            activity_data = await fetch_hourly_activity_data(
                message_history_db, target_user.id, timezone_offset
            )

            if activity_data is None or activity_data.sum() == 0:
                await ctx.send(f"No activity data found for {target_user.display_name}.")
                return

            # Generate plot
            plot = create_activity_heatmap(
                activity_data,
                target_user.display_name,
                timezone_label,
            )

            file = discord.File(fp=plot, filename="activity_heatmap.png")
            total_messages = int(activity_data.sum())
            await ctx.send(
                f"**Activity Heatmap for {target_user.display_name}**\n"
                f"*Based on {total_messages:,} messages (last 3 months)*",
                file=file,
            )
            self.logger.info(f"Activity heatmap sent for user {target_user.id}")

        except Exception:
            self.logger.exception("Error generating activity heatmap")
            await ctx.send("An error occurred while generating the heatmap.")


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
