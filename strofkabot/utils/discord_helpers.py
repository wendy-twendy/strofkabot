"""Discord-specific utility functions."""

import datetime
from collections import defaultdict
from collections.abc import Awaitable, Callable

import discord
from discord.ext import commands

from strofkabot.utils.date_utils import adjust_month


async def handle_month_navigation(
    bot: commands.Bot,
    message: discord.Message,
    current_offset: int,
    callback: Callable[[int], Awaitable[None]],
    timeout: float = 60.0,
) -> None:
    """Add navigation reactions and handle month navigation.

    Adds left/right arrow reactions to the message and waits for user interaction.
    On navigation, deletes the message and calls the callback with the new offset.

    Args:
        bot: Discord bot instance (for wait_for)
        message: Message to add reactions to
        current_offset: Current month offset value
        callback: Async function(new_offset) to regenerate the view
        timeout: Seconds to wait for reaction (default 60)
    """
    await message.add_reaction("⬅️")
    await message.add_reaction("➡️")

    def check(reaction: discord.Reaction, user: discord.User) -> bool:
        return (
            reaction.message.id == message.id and str(reaction.emoji) in ["⬅️", "➡️"] and not user.bot
        )

    while True:
        try:
            reaction, user = await bot.wait_for("reaction_add", timeout=timeout, check=check)
            new_offset = current_offset
            if str(reaction.emoji) == "⬅️":
                new_offset -= 1
            elif str(reaction.emoji) == "➡️":
                new_offset += 1
            await message.delete()
            await callback(new_offset)
            break
        except TimeoutError:
            break


def parse_rpm_args(args: tuple) -> dict[str, bool]:
    """Parse command-line arguments for RPM commands."""
    options = {"option": None, "flags": []}
    for arg in args:
        if arg.startswith("--"):
            options["flags"].append(arg)
        elif not options["option"]:
            options["option"] = arg

    return {
        "least": "--least" in options["flags"],
        "all_users": "--all" in options["flags"],
        "leaderboard": "--leaderboard" in options["flags"],
    }


async def send_leaderboard(
    ctx, year: int, month: int, least: bool, all_users: bool, user_stats, bot
) -> None:
    """Send an interactive RPM leaderboard with navigation."""

    async def send_leaderboard_inner(year: int, month: int, month_offset: int = 0):
        adjusted_year, adjusted_month = adjust_month(year, month, month_offset)
        current_date = datetime.date.today()

        stats = await user_stats.get_monthly_stats(adjusted_year, adjusted_month)

        if not stats and datetime.date(adjusted_year, adjusted_month, 1) >= datetime.date(
            current_date.year, current_date.month, 1
        ):
            adjusted_year, adjusted_month = adjust_month(adjusted_year, adjusted_month, -1)
            stats = await user_stats.get_monthly_stats(adjusted_year, adjusted_month)

        if not stats:
            await ctx.send("No user statistics available for this month.")
            return

        filtered_stats = [stat for stat in stats if stat["total_msgs"] >= 30]

        if not filtered_stats:
            await ctx.send("No users with at least 30 messages found for this month.")
            return

        total_reactions = sum(stat["total_reacts"] for stat in stats)
        total_messages = sum(stat["total_msgs"] for stat in stats)
        server_avg_rpm = total_reactions / total_messages if total_messages > 0 else 0

        filtered_stats.sort(key=lambda x: x["avg_reacts"], reverse=not least)

        leaderboard_type = "Least" if least else "RPM"
        response = f"**{leaderboard_type} Leaderboard for {datetime.date(adjusted_year, adjusted_month, 1).strftime('%B %Y')}:**\n"
        response += f"Server Average RPM: {server_avg_rpm:.2f}\n"
        response += "(Users with at least 30 messages)\n```\n"
        response += f"{'User':<20} {'RPM':>5} {'Msgs':>5} {'Reacts':>7}\n"
        response += "-" * 40 + "\n"

        users_to_show = filtered_stats if all_users else filtered_stats[:10]

        for stat in users_to_show:
            display_name = stat["username"] or f"User {stat['author_id']}"
            response += f"{display_name[:20]:<20} {stat['avg_reacts']:5.2f} {stat['total_msgs']:5d} {stat['total_reacts']:7d}\n"
        response += "```"

        message = await ctx.send(response)
        await handle_month_navigation(
            bot,
            message,
            month_offset,
            lambda new_offset: send_leaderboard_inner(year, month, new_offset),
        )

    await send_leaderboard_inner(year, month)


async def send_personal_stats(ctx, year: int, month: int, user_stats) -> None:
    """Send a user's personal RPM stats for the past 12 months."""
    response = "**Your RPM Stats for the past 12 months:**\n```\n"
    response += f"{'Month':<10} {'Messages':>10} {'Reactions':>10} {'RPM':>5}\n"
    response += "-" * 40 + "\n"

    current_date = datetime.datetime.now(datetime.UTC)

    for i in range(12):
        month_date = current_date - datetime.timedelta(days=i * 30)
        y, m = month_date.year, month_date.month
        user_stat = await user_stats.get_user_monthly_stats(ctx.author.id, y, m)
        if user_stat:
            response += f"{month_date.strftime('%b %Y'):<10} {user_stat['total_msgs']:>10} {user_stat['total_reacts']:>10} {user_stat['avg_reacts']:>5.2f}\n"
        else:
            response += f"{month_date.strftime('%b %Y'):<10} {'0':>10} {'0':>10} {'0.00':>5}\n"

    response += "```"
    await ctx.send(response)


def get_reply_info(message: discord.Message) -> tuple[int | None, str | None, str | None]:
    """Extract reply information from a Discord message."""
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
            reply_to_author = (
                replied_msg.author.display_name if replied_msg.author else "Unknown User"
            )
            reply_to_content = (
                replied_msg.content if hasattr(replied_msg, "content") else "Content unavailable"
            )

    return reply_to_id, reply_to_author, reply_to_content


async def get_member_names(guild: discord.Guild, member_ids: list[int]) -> list[str]:
    """Convert member IDs to display names, filtering to ASCII characters."""
    member_names = []
    for user_id in member_ids:
        member = guild.get_member(user_id)
        if member:
            name = (
                "".join(char for char in member.display_name if ord(char) < 128)
                or f"User {user_id}"
            )
            member_names.append(name)
        else:
            member_names.append(f"User {user_id}")
    return member_names


async def get_non_bot_member_ids(guild: discord.Guild) -> list[int]:
    """Get IDs of all non-bot members in the guild."""
    return [member.id for member in guild.members if not member.bot]


def calculate_average_preference_share(
    rows: list[dict],
    min_unique_reactors: int = 5,
    min_giver_reactions: int = 10,
) -> list[tuple[str, float]]:
    """Calculate Average Preference Share scores for most-liked ranking.

    For each giver, calculates what fraction of their reactions go to each receiver.
    Then averages these shares across all givers to find who captures the most
    community attention.

    Args:
        rows: List of dicts with giver_username, receiver_username, reaction_count
        min_unique_reactors: Minimum unique reactors required to qualify for ranking
        min_giver_reactions: Minimum total reactions a giver must have to be included

    Returns:
        List of (username, score) tuples sorted by score descending
    """
    # Step 1: Calculate total reactions given by each giver
    giver_totals: dict[str, int] = defaultdict(int)
    for row in rows:
        giver_totals[row["giver_username"]] += row["reaction_count"]

    # Filter to only include givers with minimum reactions
    qualified_givers = {
        giver for giver, total in giver_totals.items() if total >= min_giver_reactions
    }

    # Step 2: Calculate preference shares and aggregate by receiver (only from qualified givers)
    receiver_shares: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        giver = row["giver_username"]
        receiver = row["receiver_username"]
        if giver != receiver and giver in qualified_givers:
            share = row["reaction_count"] / giver_totals[giver]
            receiver_shares[receiver].append(share)

    # Step 3: Calculate APS for each receiver (using only qualified givers count)
    num_givers = len(qualified_givers)
    scores: dict[str, float] = {}
    for receiver, shares in receiver_shares.items():
        unique_reactors = len(shares)
        if unique_reactors >= min_unique_reactors:
            scores[receiver] = sum(shares) / num_givers

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def send_most_liked_stats(
    ctx, year: int, month: int, month_offset: int, user_stats, bot, show_all: bool = False
) -> None:
    """Generate and send most-liked users for a specific month."""
    adjusted_year, adjusted_month = adjust_month(year, month, month_offset)
    current_date = datetime.date.today()

    # Get data from DAL
    rows = await user_stats.get_reaction_network_for_month(adjusted_year, adjusted_month)

    # Fall back to previous month if no data and we're at current month
    if not rows and datetime.date(adjusted_year, adjusted_month, 1) >= datetime.date(
        current_date.year, current_date.month, 1
    ):
        adjusted_year, adjusted_month = adjust_month(adjusted_year, adjusted_month, -1)
        rows = await user_stats.get_reaction_network_for_month(adjusted_year, adjusted_month)

    if not rows:
        await ctx.send(
            f"No reaction data available for {datetime.date(adjusted_year, adjusted_month, 1).strftime('%B %Y')}."
        )
        return

    # Calculate Average Preference Share scores
    ranked_users = calculate_average_preference_share(rows)

    if not ranked_users:
        await ctx.send("Not enough users with sufficient reactions to calculate rankings.")
        return

    # Select users to display
    users_to_show = ranked_users if show_all else ranked_users[:5]

    # Format response
    all_suffix = " (All)" if show_all else ""
    response = f"**Most Liked Users for {datetime.date(adjusted_year, adjusted_month, 1).strftime('%B %Y')}{all_suffix}:**\n```\n"
    response += f"{'User':<20} {'Score':>8}\n"
    response += "-" * 30 + "\n"

    for user, score in users_to_show:
        # Display score as percentage
        response += f"{user[:20]:<20} {score * 100:>7.2f}%\n"

    response += "```\n"
    response += "_Score = average share of each member's reactions you receive. "
    response += "Higher = more community attention, normalized so active reactors don't dominate._"

    message = await ctx.send(response)
    await handle_month_navigation(
        bot,
        message,
        month_offset,
        lambda new_offset: send_most_liked_stats(
            ctx, year, month, new_offset, user_stats, bot, show_all
        ),
    )


def get_diversity_label(score: float) -> str:
    """Get human-readable label for diversity score."""
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Moderate"
    elif score >= 0.25:
        return "Low"
    else:
        return "Very Low"


def get_index_label(index: int) -> str:
    """Get human-readable label for echo chamber index."""
    if index <= 25:
        return "Very Diverse"
    elif index <= 50:
        return "Moderate"
    elif index <= 75:
        return "Concentrated"
    else:
        return "High"


def _get_trade_status(balance: int) -> str:
    """Get trade status string based on balance."""
    if balance > 0:
        return "SURPLUS"
    elif balance < 0:
        return "DEFICIT"
    else:
        return "NEUTRAL"


def format_yearly_trade_report(trade_data: dict, display_name: str) -> str:
    """Format a yearly (rolling 12-month) trade report.

    Args:
        trade_data: Dict with exports, imports, total_given, total_received, trade_balance
        display_name: User's display name

    Returns:
        Formatted report string
    """
    report = f"**Reaction Trade Report for {display_name}**\n"
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

    status = _get_trade_status(trade_data["trade_balance"])
    report += f"\nTrade Status: {status}```"

    return report


def format_monthly_trade_report(trade_data: dict, display_name: str, period_str: str) -> str:
    """Format a monthly trade report.

    Args:
        trade_data: Dict with exports, imports, total_given, total_received, trade_balance
        display_name: User's display name
        period_str: Formatted period string (e.g., "January 2025")

    Returns:
        Formatted report string
    """
    report = f"**Reaction Trade Report for {display_name}**\n"
    report += f"*{period_str}*\n```\n"

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

        status = _get_trade_status(trade_data["trade_balance"])
        report += f"\nTrade Status: {status}"

    report += "```"

    return report


def format_connections_report(
    affinities: list[tuple[str, str, float]], period_str: str, limit: int = 10
) -> str:
    """Format a connections report showing top mutual relationships.

    Args:
        affinities: List of (user_a, user_b, affinity_score) tuples, sorted by score
        period_str: Formatted period string (e.g., "January 2025")
        limit: Maximum number of connections to show

    Returns:
        Formatted report string
    """
    response = f"**Top {limit} Mutual Relationships** ({period_str})\n"
    response += "*Mutual Affinity Score*\n```\n"

    for i, (user_a, user_b, affinity) in enumerate(affinities[:limit], 1):
        response += f"{i:2}. {user_a} <-> {user_b}: {affinity:.3f}\n"

    response += "```"

    return response


def format_clusters_report(
    sorted_groups: list[tuple[int, list[str]]], period_str: str, total_members: int
) -> str:
    """Format a social clusters report.

    Args:
        sorted_groups: List of (community_id, member_list) tuples, sorted by size
        period_str: Formatted period string (e.g., "Jan 2025 - Mar 2025")
        total_members: Total number of members across all clusters

    Returns:
        Formatted report string
    """
    response = f"**Social Clusters** ({period_str})\n"
    response += f"*{len(sorted_groups)} groups, {total_members} members*\n```\n"

    for comm_id, members in sorted_groups:
        members_sorted = sorted(members)
        response += f"Group {comm_id + 1} ({len(members)}): {', '.join(members_sorted)}\n"

    response += "```"

    return response
