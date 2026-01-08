"""Network analysis commands: !riekt-graph, !connections, !cluster, !echo-chamber."""

import datetime
import logging
from statistics import quantiles

import discord
import networkx as nx
from discord.ext import commands

from strofkabot.discord_db import Database
from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    adjust_month,
    build_affinity_graph,
    build_reaction_graph,
    calculate_echo_chamber_metrics,
    compute_all_affinities,
    compute_community_layout,
    compute_node_activity,
    create_reaction_graph_plot,
    detect_communities,
    format_clusters_report,
    format_connections_report,
    format_period_string,
    get_diversity_label,
    get_index_label,
    get_rolling_start_month,
    handle_month_navigation,
)


class NetworkCog(commands.Cog):
    """Network and relationship analysis commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        user_stats: UserStats,
        guild: discord.Guild | None,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.guild = guild
        self.logger = logger

    def set_guild(self, guild: discord.Guild) -> None:
        """Set the guild for user lookups."""
        self.guild = guild

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
