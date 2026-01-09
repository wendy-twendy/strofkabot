"""Entertainment commands: !llumi, !artan, !unsubscribe, !on-this-day."""

import datetime
import logging
import random

import discord
from discord.ext import commands

from strofkabot.artan_quotes import ArtanQuotes
from strofkabot.config import ATTACHMENTS_DIR
from strofkabot.discord_db import Database


class EntertainmentCog(commands.Cog):
    """Entertainment and fun commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        artan_quotes: ArtanQuotes | None,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.artan_quotes = artan_quotes
        self.logger = logger

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

    @commands.command(name="ditaezeze", help="Sends Ditaezeze's message")
    async def send_ditaezeze_response(self, ctx: commands.Context):
        response = (
            "Jo, nuk kam shume frike ate por the extent to which gjerat qe kam shkruar "
            "mund te me prishin pune ne te ardhmen. Ky server nuk eshte politically "
            "correct, to put it lightly, dhe une nuk e kam shume te sigurt te ardhmen "
            "(nese do vazhdoj rrugen akademike ku reputacioni ka rendesi) dhe frika ime "
            "eshte me teper ne lidhje me footprintin qe le ketu? Kjo ndodhi ate dite qe "
            "ju permendet Lussy-in dhe thate qe dukej se ishte nje alt account i Ditusit. "
            "Me frikesoi pak idea qe mund te kishte alt dhe troll accounte ketu dhe e "
            "kerkova kush ishte ky Ditusi qe mund te kishte alt lussy-in dhe qe e "
            "permendni ju dhe pashe qe ishin bere leaks nga ky server te Ditusi dhe tek "
            "servera te tjere. Pastaj mu kujtua qe nje nate kishim biseduar me ty, "
            "Qiron/altudon dhe Davin me duket, per pronarin e ketij serveri Jakun, i cili "
            "ishte nje nder te paktet qe nuk e kishit zbuluar kush ishte. E kerkova dhe "
            "ate dhe pashe qe ai kishte fshire gjithe channelin sepse nuk donte qe "
            "mesazhet e tij ti perdoreshin kunder ne te ardhmen, qe ishte e cuditshme "
            "pasi ju as nuk e njihnit. Une kur u futa tek r/albania e mora seriozisht ne "
            "fillim (pak si shume:D) dhe kur erdha ketu e kuptova qe Fazan ishte nje "
            "nivel me i larte nga ku krijoheshin psyop-et per subin qe ishte pak "
            "mindblowing ne fillim haha. U ndjeva pak keshtu dhe per kete serverin ketu "
            "pasi ne fund te fundit nuk ju njoh dhe ju nuk njihni tamam/keni besim as te "
            "njeri tjetri. Plus, une kisha shume mesazhe vetem per nje jave 101, me "
            "shume se ca veta qe kane ketu 1 vit dhe mu duk vetja si budallaqe qe "
            "fola/zbulova kaq shume per kaq pak kohe."
        )
        await ctx.send(response)
        self.logger.info("Sent ditaezeze response")

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
