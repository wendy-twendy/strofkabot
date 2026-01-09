"""AI commands: !ask, !predict."""

import asyncio
import datetime
import logging

import discord
from discord.ext import commands

from strofkabot.config import GEMINI_MAX_CONTEXT_MESSAGES, NICKNAMES_FILE
from strofkabot.discord_db import Database
from strofkabot.gemini_client import GeminiClient
from strofkabot.openrouter_client import OpenRouterClient
from strofkabot.url_extractor import extract_urls_from_messages, format_url_context
from strofkabot.utils import (
    build_system_prompt,
    extract_images_from_messages,
    fetch_context_messages,
    format_error_response,
    get_display_name,
    load_nicknames,
    parse_prediction_date,
    prepare_context,
    split_response,
)


class AICog(commands.Cog):
    """AI assistant and prediction commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.logger = logger
        self._gemini_client: GeminiClient | None = None
        self._openrouter_client: OpenRouterClient | None = None
        self._channel_locks: dict[int, asyncio.Lock] = {}
        self._nicknames = load_nicknames(NICKNAMES_FILE)

    def _get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific channel."""
        if channel_id not in self._channel_locks:
            self._channel_locks[channel_id] = asyncio.Lock()
        return self._channel_locks[channel_id]

    @property
    def openrouter_client(self) -> OpenRouterClient | None:
        """Lazy initialization of OpenRouter client."""
        if self._openrouter_client is None:
            try:
                self._openrouter_client = OpenRouterClient()
                self.logger.info("OpenRouter client initialized successfully")
            except ValueError as e:
                self.logger.warning(f"OpenRouter client not available: {e}")
                return None
        return self._openrouter_client

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

    async def _handle_ask(
        self,
        message: discord.Message,
        question: str,
    ) -> None:
        """Core logic for answering questions with AI.

        Called by both !ask command and @mention handler.

        Args:
            message: The Discord message containing the question.
            question: The question text to answer.
        """
        channel = message.channel
        author = message.author
        guild = message.guild

        async with channel.typing():
            try:
                # Fetch ALL context messages (for conversation history)
                messages = await fetch_context_messages(
                    channel,
                    exclude_message_id=message.id,
                    limit=GEMINI_MAX_CONTEXT_MESSAGES,
                )

                # Prepare text context from all messages
                context_dicts = await prepare_context(
                    messages, self._nicknames, bot_user_id=self.bot.user.id
                )

                # Extract images from: command message + last 5 context messages
                recent_messages = [message]
                if messages:
                    recent_messages.extend(messages[-5:])
                images = await extract_images_from_messages(recent_messages)

                # Extract URLs from: question + last 2 context messages only
                url_sources = [{"content": question}]
                if context_dicts:
                    url_sources.extend(context_dicts[-2:])
                url_contents = await extract_urls_from_messages(url_sources)
                url_context = format_url_context(url_contents)

                # Get query metadata FIRST for dynamic system prompt
                query_metadata = None
                if self.openrouter_client is not None and not images:
                    query_metadata = await self.openrouter_client.classify_query(
                        question, context_dicts
                    )

                system_prompt = build_system_prompt(
                    guild_name=guild.name if guild else "Direct Message",
                    channel_name=channel.name if hasattr(channel, "name") else "DM",
                    user_name=get_display_name(author.id, author.display_name, self._nicknames),
                    query_metadata=query_metadata,
                )

                response = None
                used_fallback = False

                # Try OpenRouter first (including for images with vision model)
                if self.openrouter_client is not None:
                    response = await self.openrouter_client.ask_with_context(
                        question=question,
                        system_prompt=system_prompt,
                        context_messages=context_dicts,
                        images=images if images else None,
                        query_metadata=query_metadata,
                        url_context=url_context if url_context else None,
                    )

                    if response.success:
                        self.logger.info(
                            f"OpenRouter success: model={response.model_used}, "
                            f"search={response.search_used}, thinking={response.thinking_used}"
                        )
                    else:
                        self.logger.warning(
                            f"OpenRouter failed: {response.error_message}, falling back to Gemini"
                        )
                        response = None  # Try Gemini fallback

                # Fall back to Gemini if OpenRouter failed
                if response is None and self.gemini_client is not None:
                    used_fallback = True
                    response = await self.gemini_client.ask_with_context(
                        question=question,
                        system_prompt=system_prompt,
                        context_messages=context_dicts,
                        images=images if images else None,
                    )

                # No response from either client
                if response is None:
                    await channel.send(format_error_response("config"))
                    return

                if not response.success:
                    provider = "Gemini" if used_fallback else "OpenRouter"
                    self.logger.error(f"{provider} error: {response.error_message}")
                    if "rate" in (response.error_message or "").lower():
                        await channel.send(format_error_response("rate_limit"))
                    else:
                        await channel.send(format_error_response("api", response.error_message))
                    return

                chunks = split_response(response.text)
                for chunk in chunks:
                    await channel.send(chunk)

                self.logger.info(
                    f"Ask completed for {author} using model {response.model_used}"
                    + (" (fallback)" if used_fallback else "")
                )

            except Exception:
                self.logger.exception("Unexpected error in ask handler")
                await channel.send(format_error_response("api"))

    @commands.command(
        name="ask",
        help="Ask a question with AI assistance. Uses recent chat context and web search.",
    )
    async def ask_question(self, ctx: commands.Context, *, question: str = ""):
        """Answer a question using AI with conversation context.

        Usage: !ask <your question>

        The bot will consider the last 10 messages in the channel as context,
        including any images. It can also search the web for current information.

        Uses OpenRouter as primary (free models), falls back to Gemini for images
        or if OpenRouter fails.
        """
        # Get per-channel lock
        channel_lock = self._get_channel_lock(ctx.channel.id)

        # Check if already processing a request in this channel
        if channel_lock.locked():
            await ctx.send(f"{ctx.author.mention} jam duke shkruar o kar, prit radhen")
            return

        async with channel_lock:
            if not question.strip():
                await ctx.send(format_error_response("no_question"))
                return

            if len(question) > 20000:
                await ctx.send(format_error_response("too_long"))
                return

            # Check if at least one client is available
            if self.openrouter_client is None and self.gemini_client is None:
                await ctx.send(format_error_response("config"))
                return

            self.logger.info(f"Ask command from {ctx.author}: {question[:50]}...")
            await self._handle_ask(ctx.message, question)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Handle bot mentions and replies to bot messages as !ask command."""
        # Ignore bot messages
        if message.author.bot:
            return

        # Skip command messages to avoid double-processing
        if message.content.startswith("!"):
            return

        question = None

        # Check if bot is mentioned
        if self.bot.user in message.mentions:
            # Extract question by removing the mention
            question = message.content
            question = question.replace(f"<@{self.bot.user.id}>", "")
            question = question.replace(f"<@!{self.bot.user.id}>", "")
            question = question.strip()

        # Check if replying to a bot message
        elif message.reference and message.reference.message_id:
            try:
                referenced_msg = message.reference.resolved
                if referenced_msg is None:
                    referenced_msg = await message.channel.fetch_message(
                        message.reference.message_id
                    )

                # Only trigger if replying to our bot's message
                if referenced_msg.author.id == self.bot.user.id:
                    question = message.content.strip()
                    # If no text but has image attachments, use default question
                    if not question and message.attachments:
                        question = "What's in this image?"
            except Exception:
                # Referenced message deleted or fetch failed
                return

        # If no trigger condition met, exit
        if question is None:
            return

        # Get per-channel lock
        channel_lock = self._get_channel_lock(message.channel.id)

        # Check if already processing a request in this channel
        if channel_lock.locked():
            await message.channel.send(
                f"{message.author.mention} jam duke shkruar o kar, prit radhen"
            )
            return

        async with channel_lock:
            if not question:
                await message.channel.send(format_error_response("no_question"))
                return

            if len(question) > 20000:
                await message.channel.send(format_error_response("too_long"))
                return

            # Check if at least one client is available
            if self.openrouter_client is None and self.gemini_client is None:
                await message.channel.send(format_error_response("config"))
                return

            self.logger.info(f"Implicit ask from {message.author}: {question[:50]}...")
            await self._handle_ask(message, question)

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
                "* `!predict tomorrow The weather will be sunny`\n"
                "* `!predict 25-12-2025 Christmas will be white`\n"
                "* `!predict next week I will finish this project`"
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
