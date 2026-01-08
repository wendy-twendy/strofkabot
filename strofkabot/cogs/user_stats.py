"""User statistics commands: !rpm, !most-liked, !activity."""

import datetime
import logging
import re

import discord
from discord.ext import commands

from strofkabot.message_history_db import MessageHistoryDatabase
from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    adjust_month,
    create_activity_heatmap,
    fetch_hourly_activity_data_for_range,
    handle_month_navigation,
    parse_rpm_args,
    send_leaderboard,
    send_most_liked_stats,
    send_personal_stats,
)


class UserStatsCog(commands.Cog):
    """User statistics commands."""

    def __init__(
        self,
        bot: commands.Bot,
        user_stats: UserStats,
        message_history_db: MessageHistoryDatabase | None,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.user_stats = user_stats
        self.message_history_db = message_history_db
        self.logger = logger

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
        name="activity",
        help="Shows activity heatmap. Usage: !activity [@user] [--months N]",
    )
    async def show_activity_heatmap(self, ctx: commands.Context, *, args: str = ""):
        """Display an hourly activity heatmap for a user with month navigation.

        Optional arguments:
            @user: Mention a user to see their heatmap (default: self)
            --months N: Number of months to include (1-12, default: 3)
        """
        if not self.message_history_db:
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
            await self.message_history_db.ensure_connection()

            # Fetch data for the month range
            activity_data = await fetch_hourly_activity_data_for_range(
                self.message_history_db,
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
