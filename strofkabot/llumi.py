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
    GEMINI_MAX_CONTEXT_MESSAGES,
    GUILD_ID,
    UPDATE_INTERVAL_SECONDS,
)
from strofkabot.discord_db import Database
from strofkabot.gemini_client import GeminiClient
from strofkabot.message_filter import MessageFilter
from strofkabot.tasks import BackgroundTaskManager
from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    adjust_month,
    build_affinity_graph,
    build_reaction_graph,
    build_system_prompt,
    calculate_echo_chamber_metrics,
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
    fetch_context_messages,
    fetch_gdp_data,
    fetch_hdi_data,
    fetch_hourly_activity_data_for_range,
    fetch_inflation_data,
    format_clusters_report,
    format_connections_report,
    format_error_response,
    format_monthly_trade_report,
    format_period_string,
    format_yearly_trade_report,
    get_diversity_label,
    get_index_label,
    get_reaction_trade_data,
    get_reaction_trade_data_for_month,
    get_rolling_start_month,
    handle_month_navigation,
    parse_prediction_date,
    parse_rpm_args,
    prepare_context,
    send_leaderboard,
    send_most_liked_stats,
    send_personal_stats,
    split_response,
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

        self._gemini_client: GeminiClient | None = None

    @property
    def gemini_client(self) -> GeminiClient | None:
        """Lazy initialization of Gemini client."""
        if self._gemini_client is None:
            try:
                self._gemini_client = GeminiClient()
                self.logger.info("Gemini client initialized successfully")
            except ValueError as e:
                self.logger.warning(f"Gemini client not available: {e}")
                return None
        return self._gemini_client

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
        name="ask",
        help="Ask a question with AI assistance. Uses recent chat context and web search.",
    )
    async def ask_question(self, ctx: commands.Context, *, question: str = ""):
        """Answer a question using Gemini AI with conversation context.

        Usage: !ask <your question>

        The bot will consider the last 10 messages in the channel as context,
        including any images. It can also search the web for current information.
        """
        if not question.strip():
            await ctx.send(format_error_response("no_question"))
            return

        if len(question) > 20000:
            await ctx.send(format_error_response("too_long"))
            return

        if self.gemini_client is None:
            await ctx.send(format_error_response("config"))
            return

        self.logger.info(f"Ask command from {ctx.author}: {question[:50]}...")

        async with ctx.typing():
            try:
                messages = await fetch_context_messages(
                    ctx.channel,
                    exclude_message_id=ctx.message.id,
                    limit=GEMINI_MAX_CONTEXT_MESSAGES,
                )

                context_dicts, images = await prepare_context(messages)

                user_roles = [role.name for role in ctx.author.roles if role.name != "@everyone"]
                system_prompt = build_system_prompt(
                    guild_name=ctx.guild.name if ctx.guild else "Direct Message",
                    channel_name=ctx.channel.name if hasattr(ctx.channel, "name") else "DM",
                    user_name=ctx.author.display_name,
                    user_roles=user_roles,
                )

                response = await self.gemini_client.ask_with_context(
                    question=question,
                    system_prompt=system_prompt,
                    context_messages=context_dicts,
                    images=images if images else None,
                )

                if not response.success:
                    self.logger.error(f"Gemini error: {response.error_message}")
                    if "Daily limit" in (response.error_message or ""):
                        await ctx.send(format_error_response("exhausted"))
                    elif "rate" in (response.error_message or "").lower():
                        await ctx.send(format_error_response("rate_limit"))
                    else:
                        await ctx.send(format_error_response("api", response.error_message))
                    return

                chunks = split_response(response.text)
                for chunk in chunks:
                    await ctx.send(chunk)

                self.logger.info(
                    f"Ask command completed for {ctx.author} using model {response.model_used}"
                )

            except Exception:
                self.logger.exception("Unexpected error in ask command")
                await ctx.send(format_error_response("api"))

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
            report = format_yearly_trade_report(trade_data, target_user.display_name)
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
            now = datetime.datetime.now(datetime.UTC)
            target_year, target_month = adjust_month(now.year, now.month, month_offset)

            self.logger.info(
                f"Generating monthly trade report for user {target_user.id} "
                f"({target_year}-{target_month:02d})"
            )

            trade_data = await get_reaction_trade_data_for_month(
                self.user_stats, target_user.id, target_year, target_month, self.guild
            )

            period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")
            report = format_monthly_trade_report(trade_data, target_user.display_name, period_str)

            message = await ctx.send(report)
            self.logger.info(f"Monthly trade report sent for user {target_user.id}")

            await handle_month_navigation(
                self.bot,
                message,
                month_offset,
                lambda new_offset: self._send_monthly_trade(ctx, target_user, new_offset),
            )
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
    async def show_riekt_graph(self, ctx: commands.Context, months: int = 3):
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

            if not affinities:
                await ctx.send("Not enough mutual connections to build a graph.")
                return

            # Calculate top 25% threshold (75th percentile)
            from statistics import quantiles

            affinity_values = [a[2] for a in affinities]
            _, _, q3 = quantiles(affinity_values, n=4)

            # Build affinity graph with top 25% threshold
            affinity_graph = build_affinity_graph(
                directed_graph,
                affinities,
                min_affinity=q3,
                remove_isolated=True,
            )

            # Keep only the largest connected component
            if affinity_graph.number_of_nodes() > 0:
                import networkx as nx

                components = list(nx.connected_components(affinity_graph))
                if len(components) > 1:
                    largest_component = max(components, key=len)
                    affinity_graph = affinity_graph.subgraph(largest_component).copy()

            # Detect communities and compute layout
            communities = detect_communities(directed_graph)
            activity = compute_node_activity(directed_graph)
            positions = compute_community_layout(affinity_graph, communities)

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
            now = datetime.datetime.now(datetime.UTC)
            target_year, target_month = adjust_month(now.year, now.month, month_offset)
            period_str = datetime.date(target_year, target_month, 1).strftime("%B %Y")

            self.logger.info(f"Generating connections for {target_year}-{target_month:02d}")

            rows = await self.user_stats.get_reaction_network_for_month(target_year, target_month)

            if not rows:
                await ctx.send(f"No reaction data available for {period_str}.")
                return

            reaction_data = [
                (r["giver_username"], r["receiver_username"], r["reaction_count"]) for r in rows
            ]

            directed_graph = build_reaction_graph(reaction_data)
            affinities = compute_all_affinities(directed_graph)

            if not affinities:
                await ctx.send(f"No mutual connections found for {period_str}.")
                return

            response = format_connections_report(affinities, period_str)
            message = await ctx.send(response)
            self.logger.info("Connections sent successfully")

            await handle_month_navigation(
                self.bot,
                message,
                month_offset,
                lambda new_offset: self._send_connections(ctx, new_offset),
            )
        except Exception:
            self.logger.exception("Error generating connections")
            await ctx.send("An error occurred while generating the connections.")

    @commands.command(name="cluster", help="Shows social clusters. Usage: !cluster [months]")
    async def show_clusters(self, ctx: commands.Context, months: int = 3):
        """Show social clusters using Louvain community detection with navigation."""
        # Validate months parameter
        if months < 1:
            await ctx.send("Number of months must be at least 1.")
            return
        if months > 12:
            await ctx.send("Number of months must be at most 12.")
            return
        await self._send_clusters(ctx, months=months, month_offset=0)

    async def _send_clusters(self, ctx: commands.Context, months: int, month_offset: int):
        """Send clusters message with navigation reactions."""
        try:
            self.logger.info(
                f"Generating clusters for {months} month(s) with offset {month_offset}"
            )

            # Calculate rolling window: start from (current - 1 + offset) and go back months
            now = datetime.datetime.now(datetime.UTC)
            # End month is previous month + offset
            end_year, end_month = adjust_month(now.year, now.month, -1 + month_offset)
            # Start month is end month - (months - 1)
            start_year, start_month = adjust_month(end_year, end_month, -(months - 1))

            # Fetch data
            reaction_data = await self.user_stats.get_reaction_network_rolling(
                start_year, start_month
            )

            if not reaction_data:
                period_str = format_period_string(start_year, start_month, months)
                await ctx.send(f"No reaction data available for {period_str}.")
                return

            # Filter edges: min weight = 3 per month, bidirectional only
            min_edge_weight = 3 * months

            # Build edge dict with min weight filter
            edges: dict[tuple[str, str], int] = {}
            for giver, receiver, count in reaction_data:
                if count >= min_edge_weight:
                    edges[(giver, receiver)] = count

            # Keep only bidirectional edges (both A->B and B->A must exist)
            bidirectional_data: list[tuple[str, str, int]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for (giver, receiver), count in edges.items():
                if (receiver, giver) in edges:
                    pair = tuple(sorted([giver, receiver]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        bidirectional_data.append((giver, receiver, count))
                        bidirectional_data.append((receiver, giver, edges[(receiver, giver)]))

            if not bidirectional_data:
                period_str = format_period_string(start_year, start_month, months)
                await ctx.send(f"No strong mutual connections for {period_str}.")
                return

            # Build graph from bidirectional data
            directed_graph = build_reaction_graph(bidirectional_data)

            # Detect communities with resolution=1.5
            communities = detect_communities(directed_graph, resolution=1.5)

            if not communities:
                period_str = format_period_string(start_year, start_month, months)
                await ctx.send(f"No communities detected for {period_str}.")
                return

            # Group members by community
            community_groups: dict[int, list[str]] = {}
            for member, comm_id in communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(member)

            # Sort groups by size (largest first) and members alphabetically
            sorted_groups = sorted(community_groups.items(), key=lambda x: (-len(x[1]), x[0]))

            period_str = format_period_string(start_year, start_month, months)
            response = format_clusters_report(sorted_groups, period_str, len(communities))
            message = await ctx.send(response)
            self.logger.info("Clusters sent successfully")

            await handle_month_navigation(
                self.bot,
                message,
                month_offset,
                lambda new_offset: self._send_clusters(ctx, months, new_offset),
            )
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
        parsed_date, prediction_text = parse_prediction_date(args)

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

    @commands.command(
        name="activity",
        help="Shows activity heatmap. Usage: !activity [@user] [--months N]",
    )
    async def show_activity_heatmap(self, ctx: commands.Context, *, args: str = ""):
        """Display an hourly activity heatmap for a user with month navigation.

        Optional arguments:
            @user: Mention a user to see their heatmap (default: self)
            --months N: Number of months to include (1-12, default: 3)
        """
        message_history_db = self.task_manager.message_history_db
        if not message_history_db:
            await ctx.send("Activity data is not available.")
            return

        # Parse target user (mentioned or self)
        target_user = ctx.message.mentions[0] if ctx.message.mentions else ctx.author

        # Parse number of months (default 3, max 12)
        num_months = 3
        months_match = re.search(r"--months\s*(\d+)", args.lower())
        if months_match:
            num_months = int(months_match.group(1))
            num_months = max(1, min(12, num_months))  # Clamp to 1-12

        await self._send_activity_heatmap(ctx, target_user, num_months, window_offset=0)

    async def _send_activity_heatmap(
        self,
        ctx: commands.Context,
        target_user: discord.Member,
        num_months: int,
        window_offset: int,
    ):
        """Send activity heatmap with navigation reactions."""
        message_history_db = self.task_manager.message_history_db
        # Hardcoded to Tirana (Albania) timezone: UTC+1
        timezone_offset = 1
        timezone_label = "Tirana"

        try:
            # Calculate end month for the window
            # window_offset=0 means current month is end of window
            # window_offset=-1 means previous window (shifted back by num_months)
            now = datetime.datetime.now(datetime.UTC)
            end_year, end_month = adjust_month(now.year, now.month, window_offset * num_months)

            # Calculate start month for period string
            start_year, start_month = adjust_month(end_year, end_month, -(num_months - 1))

            self.logger.info(
                f"Generating activity heatmap for user {target_user.id} "
                f"({start_year}-{start_month:02d} to {end_year}-{end_month:02d})"
            )

            # Ensure database is initialized
            await message_history_db.ensure_connection()

            # Fetch data for the month range
            activity_data = await fetch_hourly_activity_data_for_range(
                message_history_db,
                target_user.id,
                end_year,
                end_month,
                num_months,
                timezone_offset,
            )

            # Format period string
            if num_months == 1:
                period_str = datetime.date(end_year, end_month, 1).strftime("%B %Y")
            else:
                start_str = datetime.date(start_year, start_month, 1).strftime("%b %Y")
                end_str = datetime.date(end_year, end_month, 1).strftime("%b %Y")
                period_str = f"{start_str} - {end_str}"

            if activity_data is None or activity_data.sum() == 0:
                # No data for this period - show message with navigation
                message = await ctx.send(
                    f"No activity data for {target_user.display_name} in {period_str}."
                )
            else:
                # Generate plot
                plot = create_activity_heatmap(
                    activity_data,
                    target_user.display_name,
                    timezone_label,
                )

                file = discord.File(fp=plot, filename="activity_heatmap.png")
                total_messages = int(activity_data.sum())
                message = await ctx.send(
                    f"**Activity Heatmap for {target_user.display_name}**\n"
                    f"*Based on {total_messages:,} messages ({period_str})*",
                    file=file,
                )
                self.logger.info(f"Activity heatmap sent for user {target_user.id}")

            # Add navigation reactions
            await handle_month_navigation(
                self.bot,
                message,
                window_offset,
                lambda new_offset: self._send_activity_heatmap(
                    ctx, target_user, num_months, new_offset
                ),
            )

        except Exception:
            self.logger.exception("Error generating activity heatmap")
            await ctx.send("An error occurred while generating the heatmap.")

    @commands.command(
        name="echo-chamber",
        aliases=["ec", "bubble"],
        help="Shows if you're in a reaction echo chamber. Usage: !echo-chamber [@user]",
    )
    async def show_echo_chamber(self, ctx: commands.Context, member: discord.Member = None):
        """Display echo chamber analysis for a user (rolling 3 months)."""
        try:
            target_user = member or ctx.author

            # Calculate rolling 3-month window
            now = datetime.datetime.now(datetime.UTC)
            # Go back 3 months from current month
            start_year, start_month = adjust_month(now.year, now.month, -3)

            self.logger.info(
                f"Generating echo chamber analysis for user {target_user.id} "
                f"from {start_year}-{start_month:02d}"
            )

            # Fetch data
            data = await self.user_stats.get_echo_chamber_data(
                target_user.id, start_year, start_month
            )

            # Check if there's any data
            if not data["outgoing"] and not data["incoming"]:
                await ctx.send(
                    f"No reaction data found for {target_user.display_name} in the last 3 months."
                )
                return

            # Calculate metrics
            metrics = calculate_echo_chamber_metrics(data["outgoing"], data["incoming"])

            # Resolve usernames for top partners
            async def resolve_name(user_id: int) -> str:
                member = self.guild.get_member(user_id)
                if member:
                    return member.display_name
                # Try database lookup
                username = await self.db.get_username_by_id(user_id)
                return username or f"User {user_id}"

            # Build response
            period_str = format_period_string(start_year, start_month, 3)

            response = f"**Echo Chamber Analysis for {target_user.display_name}**\n"
            response += f"*{period_str}*\n```\n"

            # Outgoing section
            response += "Outgoing Reactions (Given):\n"
            if data["outgoing"]:
                response += (
                    f"  Top 3 recipients: {metrics['outgoing_top3_pct']:.1f}% of all reactions\n"
                )
                response += f"  Diversity Score: {metrics['outgoing_diversity']:.2f}/1.00"
                diversity_label = get_diversity_label(metrics["outgoing_diversity"])
                response += f" ({diversity_label})\n"
                response += "  Your top targets: "
                top_names = []
                for user_id, _count, pct in metrics["outgoing_top"]:
                    name = await resolve_name(user_id)
                    top_names.append(f"{name} ({pct:.0f}%)")
                response += ", ".join(top_names) + "\n"
            else:
                response += "  No reactions given\n"

            response += "\n"

            # Incoming section
            response += "Incoming Reactions (Received):\n"
            if data["incoming"]:
                response += (
                    f"  Top 3 givers: {metrics['incoming_top3_pct']:.1f}% of all reactions\n"
                )
                response += f"  Diversity Score: {metrics['incoming_diversity']:.2f}/1.00"
                diversity_label = get_diversity_label(metrics["incoming_diversity"])
                response += f" ({diversity_label})\n"
                response += "  Your top fans: "
                top_names = []
                for user_id, _count, pct in metrics["incoming_top"]:
                    name = await resolve_name(user_id)
                    top_names.append(f"{name} ({pct:.0f}%)")
                response += ", ".join(top_names) + "\n"
            else:
                response += "  No reactions received\n"

            response += "\n"

            # Echo Chamber Index
            response += f"Echo Chamber Index: {metrics['echo_chamber_index']}/100"
            index_label = get_index_label(metrics["echo_chamber_index"])
            response += f" ({index_label})\n"
            response += metrics["interpretation"]

            response += "```"

            await ctx.send(response)
            self.logger.info(f"Echo chamber analysis sent for user {target_user.id}")

        except Exception:
            self.logger.exception("Error generating echo chamber analysis")
            await ctx.send("An error occurred while generating the echo chamber analysis.")


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
