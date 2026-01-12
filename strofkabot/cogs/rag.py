"""Discord RAG commands cog.

Commands:
- !qa <question> - Q&A with LLM-generated answers based on server history
"""

from __future__ import annotations

import datetime
import logging

import discord
from discord.ext import commands

from strofkabot.config import (
    GUILD_ID,
    NICKNAMES_FILE,
    RAG_LLM_MODEL,
    RAG_MAX_QUERY_VARIANTS,
    RAG_VECTOR_STORE_DIR,
)
from strofkabot.rag.embeddings import OpenRouterEmbeddingClient
from strofkabot.rag.llm_client import RAGLLMClient
from strofkabot.rag.pipeline import RAGPipeline, RAGResponse
from strofkabot.rag.query_rewriter import (
    ConversationMessage,
    MemberInfo,
    RewrittenQuery,
    rewrite_query,
)
from strofkabot.utils.nickname_loader import load_nicknames

logger = logging.getLogger(__name__)

# Cache TTL for member list (1 hour)
_MEMBER_CACHE_TTL = datetime.timedelta(hours=1)


class RAGCog(commands.Cog):
    """RAG-based semantic search and Q&A commands."""

    def __init__(
        self,
        bot: commands.Bot,
        db,
        logger: logging.Logger,
    ):
        self.bot = bot
        self.db = db
        self.logger = logger

        # Lazy initialization
        self._pipeline: RAGPipeline | None = None
        self._nicknames: dict[int, list[str]] | None = None
        self._openrouter_client = None

        # Member cache for query rewriting
        self._member_cache: list[MemberInfo] | None = None
        self._member_cache_time: datetime.datetime | None = None

    @property
    def nicknames(self) -> dict[int, list[str]]:
        """Lazy load nicknames."""
        if self._nicknames is None:
            self._nicknames = load_nicknames(NICKNAMES_FILE)
            self.logger.info(f"RAG: Loaded {len(self._nicknames)} nickname mappings")
        return self._nicknames

    @property
    def pipeline(self) -> RAGPipeline | None:
        """Lazy initialization of RAG pipeline."""
        if self._pipeline is None:
            # Check if vector store exists
            if not RAG_VECTOR_STORE_DIR.exists():
                self.logger.warning(
                    f"RAG vector store directory not found: {RAG_VECTOR_STORE_DIR}. "
                    "Run ingestion first to populate the vector store."
                )
                return None

            try:
                embedding_client = OpenRouterEmbeddingClient()
                llm_client = RAGLLMClient(model=RAG_LLM_MODEL)
                self._pipeline = RAGPipeline(
                    vector_store_dir=RAG_VECTOR_STORE_DIR,
                    embedding_client=embedding_client,
                    nicknames=self.nicknames,
                    llm_client=llm_client,
                    # use_bm25 defaults to False (BM25 index uses ~4GB RAM)
                )

                chunk_count = self._pipeline.vector_store.count()
                if chunk_count == 0:
                    self.logger.warning("RAG vector store is empty. Run ingestion to populate it.")
                else:
                    self.logger.info(f"RAG pipeline initialized with {chunk_count} chunks")
            except Exception as e:
                self.logger.exception(f"Failed to initialize RAG pipeline: {e}")
                return None

        return self._pipeline

    @property
    def openrouter_client(self):
        """Lazy initialization of OpenRouter client for query rewriting."""
        if self._openrouter_client is None:
            try:
                from strofkabot.openrouter import OpenRouterClient

                self._openrouter_client = OpenRouterClient()
                self.logger.info("OpenRouter client initialized for RAG query rewriting")
            except ValueError as e:
                self.logger.warning(f"OpenRouter client not available: {e}")
                return None
        return self._openrouter_client

    async def _get_members(self) -> list[MemberInfo]:
        """Get cached member list for entity resolution.

        Returns:
            List of MemberInfo with display names, usernames, and nicknames.
            Cached for 1 hour to avoid repeated guild queries.
        """
        now = datetime.datetime.now(datetime.UTC)

        # Check if cache is valid
        if (
            self._member_cache is not None
            and self._member_cache_time is not None
            and now - self._member_cache_time < _MEMBER_CACHE_TTL
        ):
            return self._member_cache

        # Get guild
        guild = self.bot.get_guild(GUILD_ID)
        if not guild:
            self.logger.warning(f"Guild {GUILD_ID} not found for member cache")
            return []

        # Build member list
        members = []
        for member in guild.members:
            if member.bot:
                continue

            custom_nicks = self.nicknames.get(member.id, [])
            members.append(
                MemberInfo(
                    author_id=member.id,
                    display_name=member.display_name,
                    username=member.name,
                    nicknames=custom_nicks,
                )
            )

        self._member_cache = members
        self._member_cache_time = now
        self.logger.debug(f"Refreshed member cache: {len(members)} members")

        return members

    async def _get_conversation_history(
        self,
        channel: discord.TextChannel,
        exclude_message_id: int,
        limit: int = 5,
    ) -> list[ConversationMessage]:
        """Fetch recent messages for conversation context.

        Args:
            channel: The channel to fetch from.
            exclude_message_id: Message ID to exclude (the command itself).
            limit: Maximum messages to fetch.

        Returns:
            List of ConversationMessage, oldest first.
        """
        messages = []
        async for msg in channel.history(limit=limit + 1):
            if msg.id == exclude_message_id:
                continue
            if msg.author.bot:
                continue
            messages.append(
                ConversationMessage(
                    author=msg.author.display_name,
                    author_id=msg.author.id,
                    content=msg.content or "",
                )
            )
            if len(messages) >= limit:
                break

        # Return oldest first
        return list(reversed(messages))

    @commands.command(name="qa")
    async def qa_command(self, ctx: commands.Context, *, question: str):
        """Ask a question about server history with AI-generated answer.

        Usage: !qa <question>
        Example: !qa What does taka think about politics?
        """
        if not self.pipeline:
            await ctx.reply(
                "RAG system is not available. The vector store may need to be populated."
            )
            return

        async with ctx.typing():
            try:
                # Rewrite query for better retrieval
                rewritten = None
                if self.openrouter_client:
                    try:
                        # Get context for query rewriting
                        members = await self._get_members()
                        conversation_history = await self._get_conversation_history(
                            ctx.channel,
                            ctx.message.id,
                            limit=5,
                        )

                        rewritten = await rewrite_query(
                            self.openrouter_client._client,
                            question,
                            members=members,
                            conversation_history=conversation_history,
                        )
                        self.logger.info(
                            f"QA rewritten: queries={rewritten.rag_queries}, "
                            f"entities={rewritten.detected_entities}, "
                            f"ids={rewritten.resolved_entity_ids}, "
                            f"strategy={rewritten.retrieval_strategy}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Query rewriting failed: {e}")

                # Use enhanced search if rewriting succeeded
                if rewritten and len(rewritten.rag_queries) > 0:
                    response = await self._search_with_rewritten_query(question, rewritten)
                else:
                    # Fallback to original behavior
                    response = await self.pipeline.answer_question(question, context_k=5)

                # Create embed
                embed = discord.Embed(
                    title=f"Q: {question[:200]}",
                    color=discord.Color.green() if response.success else discord.Color.orange(),
                )

                # Truncate answer if too long
                answer = response.answer
                if len(answer) > 2000:
                    answer = answer[:1997] + "..."

                embed.description = answer

                # Add source info
                if response.sources:
                    source_info = []
                    for s in response.sources[:3]:
                        meta = s.get("metadata", {})
                        channel = meta.get("channel_name", "?")
                        sim = s.get("similarity", 0)
                        source_info.append(f"#{channel} ({sim:.2f})")
                    embed.set_footer(text=f"Sources: {', '.join(source_info)}")

                await ctx.reply(embed=embed)

            except Exception as e:
                self.logger.exception("QA command error")
                await ctx.reply(f"Question answering failed: {str(e)[:100]}")

    async def _search_with_rewritten_query(
        self,
        original_question: str,
        rewritten: RewrittenQuery,
    ) -> RAGResponse:
        """Search using rewritten query variants.

        Args:
            original_question: The user's original question.
            rewritten: RewrittenQuery with enhanced search parameters.

        Returns:
            RAGResponse with answer and sources.
        """
        queries = rewritten.rag_queries[:RAG_MAX_QUERY_VARIANTS]

        # Use resolved query for answer generation if pronouns were resolved
        question_for_answer = rewritten.resolved_query or original_question

        # Collect results from all query variants
        all_results = []
        for query in queries:
            results = await self.pipeline.semantic_search(
                query=query,
                k=5,
                auto_filter_participants=True,
            )
            all_results.extend(results)

        if not all_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer this question.",
                sources=[],
                query=original_question,
                success=True,
            )

        # Deduplicate by chunk_id, keeping highest similarity
        seen = {}
        for r in all_results:
            if r.chunk_id not in seen or r.similarity > seen[r.chunk_id].similarity:
                seen[r.chunk_id] = r
        unique_results = list(seen.values())

        # Filter by resolved entity IDs if participant_focused strategy
        if rewritten.retrieval_strategy == "participant_focused" and rewritten.resolved_entity_ids:
            filtered = []
            for r in unique_results:
                # Check if any resolved entity is in the chunk's participant_ids
                participant_ids = r.metadata.get("participant_ids", [])
                if any(str(eid) in participant_ids for eid in rewritten.resolved_entity_ids):
                    filtered.append(r)
            # Use filtered results if we got any, otherwise fall back to all
            if filtered:
                unique_results = filtered

        # Sort by similarity and take top 5
        unique_results = sorted(unique_results, key=lambda x: x.similarity, reverse=True)[:5]

        # Generate answer using pipeline's LLM
        context = self.pipeline._build_context(unique_results)
        answer, success, error = await self.pipeline._generate_answer(question_for_answer, context)

        sources = [
            {
                "chunk_id": r.chunk_id,
                "text": r.document,
                "metadata": r.metadata,
                "similarity": r.similarity,
            }
            for r in unique_results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=original_question,
            success=success,
            error_message=error,
        )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog via bot.load_extension()."""
    # This function is used when loading via load_extension()
    # For direct instantiation, use RAGCog constructor
    pass
