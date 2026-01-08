"""Economics commands: !inflation, !trade, !gdp, !hdi."""

import datetime
import logging

import discord
from discord.ext import commands

from strofkabot.user_stats import UserStats
from strofkabot.utils import (
    adjust_month,
    calculate_monthly_inflation,
    calculate_yearly_inflation,
    create_gdp_plot,
    create_hdi_plot,
    create_monthly_inflation_plot,
    create_yearly_inflation_plot,
    fetch_gdp_data,
    fetch_hdi_data,
    fetch_inflation_data,
    format_monthly_trade_report,
    format_yearly_trade_report,
    get_reaction_trade_data,
    get_reaction_trade_data_for_month,
    handle_month_navigation,
)


class EconomicsCog(commands.Cog):
    """Server economics and trade commands."""

    def __init__(
        self,
        bot: commands.Bot,
        user_stats: UserStats,
        guild: discord.Guild | None,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.user_stats = user_stats
        self.guild = guild
        self.logger = logger

    def set_guild(self, guild: discord.Guild) -> None:
        """Set the guild for trade reports."""
        self.guild = guild

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
