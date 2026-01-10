"""AI commands: !ask, !predict."""

import asyncio
import datetime
import logging
from collections import OrderedDict

import discord
from discord.ext import commands

from strofkabot.config import (
    GEMINI_MAX_CONTEXT_MESSAGES,
    MEMORY_INJECTION_ENABLED,
    MEMORY_SERVER_LIMIT,
    MEMORY_USER_LIMIT,
    NICKNAMES_FILE,
)
from strofkabot.discord_db import Database
from strofkabot.gemini_client import GeminiClient
from strofkabot.memory_store import Memory, MemoryStore
from strofkabot.openrouter import OpenRouterClient
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

# Maximum number of channel locks to keep in memory
MAX_CHANNEL_LOCKS = 100


class AICog(commands.Cog):
    """AI assistant and prediction commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        logger: logging.Logger,
        memory_store: MemoryStore | None = None,
    ):
        self.bot = bot
        self.db = db
        self.logger = logger
        self.memory_store = memory_store
        self._gemini_client: GeminiClient | None = None
        self._openrouter_client: OpenRouterClient | None = None
        # Use OrderedDict for LRU-style eviction of channel locks
        self._channel_locks: OrderedDict[int, asyncio.Lock] = OrderedDict()
        self._nicknames = load_nicknames(NICKNAMES_FILE)

    def _get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific channel.

        Uses LRU eviction to prevent unbounded memory growth.
        """
        if channel_id in self._channel_locks:
            # Move to end (most recently used)
            self._channel_locks.move_to_end(channel_id)
            return self._channel_locks[channel_id]

        # Create new lock
        lock = asyncio.Lock()
        self._channel_locks[channel_id] = lock

        # Evict oldest locks if over limit (only if not currently held)
        while len(self._channel_locks) > MAX_CHANNEL_LOCKS:
            oldest_id, oldest_lock = next(iter(self._channel_locks.items()))
            if not oldest_lock.locked():
                del self._channel_locks[oldest_id]
            else:
                # Can't evict a locked lock, move it to end and try next
                self._channel_locks.move_to_end(oldest_id)
                # If all locks are held, allow temporary overflow
                if all(lock.locked() for lock in self._channel_locks.values()):
                    break

        return lock

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
        user_name = get_display_name(author.id, author.display_name, self._nicknames)

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

                # Load and filter relevant memories (only if injection is enabled)
                relevant_user_memories = []
                relevant_server_memories = []
                if MEMORY_INJECTION_ENABLED and self.memory_store and self.openrouter_client:
                    # Load all memories
                    all_user_memories = await self.memory_store.get_user_memories(author.id)
                    all_server_memories = await self.memory_store.get_server_memories()

                    # Combine for relevance filtering
                    all_memories = all_user_memories + all_server_memories

                    if all_memories:
                        # Filter to only relevant memories
                        relevant_indices = await self.openrouter_client.filter_relevant_memories(
                            question, all_memories
                        )

                        # Split back into user/server memories
                        user_count = len(all_user_memories)
                        for idx in relevant_indices:
                            if idx < user_count:
                                relevant_user_memories.append(all_user_memories[idx])
                            else:
                                relevant_server_memories.append(
                                    all_server_memories[idx - user_count]
                                )

                        # Mark relevant memories as accessed
                        for mem in relevant_user_memories:
                            await self.memory_store.mark_user_memory_accessed(author.id, mem.text)
                        for mem in relevant_server_memories:
                            await self.memory_store.mark_server_memory_accessed(mem.text)

                system_prompt = build_system_prompt(
                    guild_name=guild.name if guild else "Direct Message",
                    channel_name=channel.name if hasattr(channel, "name") else "DM",
                    user_name=user_name,
                    query_metadata=query_metadata,
                    user_memories=relevant_user_memories if relevant_user_memories else None,
                    server_memories=relevant_server_memories if relevant_server_memories else None,
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

                # Schedule async memory extraction (non-blocking)
                if self.memory_store and self.openrouter_client and not used_fallback:
                    asyncio.create_task(
                        self._extract_and_save_memories(
                            context_dicts,
                            question,
                            response.text,
                            author.id,
                            user_name,
                        )
                    )

            except Exception:
                self.logger.exception("Unexpected error in ask handler")
                await channel.send(format_error_response("api"))

    async def _extract_and_save_memories(
        self,
        context_dicts: list[dict],
        question: str,
        response_text: str,
        user_id: int,
        user_name: str,
    ) -> None:
        """Extract memories from conversation and save them (runs in background).

        Args:
            context_dicts: List of context message dicts.
            question: The user's question.
            response_text: The AI's response.
            user_id: Discord ID of the user who asked.
            user_name: Display name of the user.
        """
        try:
            # Build known_users mapping from context (name -> id)
            known_users: dict[str, int] = {}
            # Also build reverse mapping for looking up names by id
            id_to_name: dict[int, str] = {}

            for msg in context_dicts:
                author = msg.get("author")
                author_id = msg.get("author_id")
                if author and author_id and not msg.get("is_bot"):
                    known_users[author.lower()] = author_id
                    id_to_name[author_id] = author

            # Add the asking user
            known_users[user_name.lower()] = user_id
            id_to_name[user_id] = user_name

            # Load existing memories for context (helps detect duplicates and updates)
            existing_user_memories = await self.memory_store.get_user_memories(user_id)
            existing_server_memories = await self.memory_store.get_server_memories()

            # Extract memories using tool calling
            extracted = await self.openrouter_client.extract_memories(
                context_dicts,
                question,
                response_text,
                user_id,
                user_name,
                known_users,
                existing_user_memories=existing_user_memories,
                existing_server_memories=existing_server_memories,
            )

            # Process user memory invalidations first (before saves/updates)
            for inv_data in extracted.get("user_invalidations", []):
                target_user_id = inv_data.get("user_id")
                text_match = inv_data.get("text_match", "")
                if target_user_id and text_match:
                    if await self.memory_store.invalidate_user_memory(target_user_id, text_match):
                        self.logger.info(f"Invalidated user memory: '{text_match}'")

            # Process server memory invalidations
            for inv_data in extracted.get("server_invalidations", []):
                text_match = inv_data.get("text_match", "")
                if text_match:
                    if await self.memory_store.invalidate_server_memory(text_match):
                        self.logger.info(f"Invalidated server memory: '{text_match}'")

            # Process user memory updates
            for upd_data in extracted.get("user_updates", []):
                target_user_id = upd_data.get("user_id")
                old_match = upd_data.get("old_match", "")
                new_text = upd_data.get("new_memory_text", "")
                if target_user_id and old_match and new_text:
                    new_memory = Memory(
                        text=new_text,
                        category=upd_data.get("category", "facts"),
                        importance=upd_data.get("importance", 5),
                    )
                    if await self.memory_store.update_user_memory(
                        target_user_id, old_match, new_memory
                    ):
                        self.logger.info(f"Updated user memory: '{old_match}' -> '{new_text[:50]}'")

            # Process server memory updates
            for upd_data in extracted.get("server_updates", []):
                old_match = upd_data.get("old_match", "")
                new_text = upd_data.get("new_memory_text", "")
                if old_match and new_text:
                    new_memory = Memory(
                        text=new_text,
                        category=upd_data.get("category", "knowledge"),
                        importance=upd_data.get("importance", 7),
                    )
                    if await self.memory_store.update_server_memory(old_match, new_memory):
                        self.logger.info(
                            f"Updated server memory: '{old_match}' -> '{new_text[:50]}'"
                        )

            # Process new user memories
            for mem_data in extracted.get("user_memories", []):
                memory_text = mem_data.get("memory_text", "")
                if not memory_text:
                    continue

                target_user_id = mem_data.get("user_id")
                if target_user_id is None:
                    continue

                # Get username from our mapping
                target_user_name = id_to_name.get(target_user_id, user_name)

                # Check for duplicates before adding
                if await self.memory_store.is_duplicate_user_memory(target_user_id, memory_text):
                    self.logger.debug(f"Skipping duplicate user memory: {memory_text[:50]}")
                    continue

                memory = Memory(
                    text=memory_text,
                    category=mem_data.get("category", "facts"),
                    importance=mem_data.get("importance", 5),
                    confidence=mem_data.get("confidence", 1.0),
                    tags=mem_data.get("tags", []),
                )
                await self.memory_store.add_user_memory(target_user_id, memory, target_user_name)
                self.logger.info(f"Saved user memory for {target_user_name}: {memory_text[:50]}")

            # Process new server memories
            for mem_data in extracted.get("server_memories", []):
                memory_text = mem_data.get("memory_text", "")
                if not memory_text:
                    continue

                # Check for duplicates before adding
                if await self.memory_store.is_duplicate_server_memory(memory_text):
                    self.logger.debug(f"Skipping duplicate server memory: {memory_text[:50]}")
                    continue

                memory = Memory(
                    text=memory_text,
                    category=mem_data.get("category", "knowledge"),
                    importance=mem_data.get("importance", 7),
                    confidence=mem_data.get("confidence", 1.0),
                    tags=mem_data.get("tags", []),
                )
                await self.memory_store.add_server_memory(memory)
                self.logger.info(f"Saved server memory: {memory_text[:50]}")

            # Prune if needed
            await self.memory_store.prune_user_memories(user_id, limit=MEMORY_USER_LIMIT)
            await self.memory_store.prune_server_memories(limit=MEMORY_SERVER_LIMIT)

        except Exception:
            self.logger.exception("Error in memory extraction")

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
