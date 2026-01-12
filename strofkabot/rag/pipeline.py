# RAG Pipeline implementation

"""RAG (Retrieval-Augmented Generation) pipeline for semantic search and Q&A.

Orchestrates:
- Query embedding via OpenRouter
- Vector store search via LanceDB
- Nickname resolution for user queries
- Answer generation with LLM
- User insights generation
"""

from __future__ import annotations

import datetime
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .bm25_index import BM25Index, BM25Result
from .llm_client import RAGLLMClient
from .reranker import LLMReranker, SimpleReranker
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """Request parameters for semantic search.

    Attributes:
        query: The search query text.
        k: Maximum number of results to return (default 10).
        filters: Optional metadata filters (e.g., {"channel_name": "kanapeja"}).
    """

    query: str
    k: int = 10
    filters: dict | None = None


@dataclass
class RAGResponse:
    """Response from answer_question containing answer and sources.

    Attributes:
        answer: The generated answer text.
        sources: List of source chunks used to generate the answer.
        query: The original query.
        success: Whether the answer was generated successfully.
        error_message: Error message if generation failed.
    """

    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    success: bool = True
    error_message: str | None = None


@dataclass
class UserInsight:
    """Insight about a user based on their messages.

    Attributes:
        user_name: The user's display name.
        user_id: The user's Discord ID.
        insight: The generated insight text.
        insight_type: Type of insight (summary, topics, personality).
        sources: List of source chunks used.
        success: Whether generation was successful.
        error_message: Error message if generation failed.
    """

    user_name: str
    user_id: int
    insight: str
    insight_type: str = "summary"
    sources: list[dict] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


class RAGPipeline:
    """RAG pipeline for semantic search and question answering.

    Combines vector store search with embedding generation and
    LLM-based answer generation. Supports hybrid search using
    BM25 keyword matching combined with vector similarity via
    Reciprocal Rank Fusion (RRF).

    Args:
        vector_store_dir: Directory for vector store persistence.
        embedding_client: Client for generating query embeddings.
        nicknames: Optional dict mapping author_id -> list of nicknames.
        collection_name: Name of the vector store collection.
        llm_client: Optional LLM client for answer generation.
        reranker: Optional reranker for result reordering.
        bm25_persist_path: Optional path for BM25 index persistence.
    """

    def __init__(
        self,
        vector_store_dir: Path,
        embedding_client,
        nicknames: dict[int, list[str]] | None = None,
        collection_name: str = "strofka_messages",
        llm_client: RAGLLMClient | None = None,
        reranker: LLMReranker | SimpleReranker | None = None,
        bm25_persist_path: Path | None = None,
        use_bm25: bool = False,  # Disabled by default: BM25 index uses ~4GB RAM
    ):
        self.vector_store = VectorStore(
            persist_dir=vector_store_dir,
            collection_name=collection_name,
        )
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.reranker = reranker
        self.nicknames = nicknames or {}
        self.use_bm25 = use_bm25  # Global flag to enable/disable BM25 hybrid search

        # BM25 index for hybrid search (lazy-loaded)
        self._bm25_index: BM25Index | None = None
        self._bm25_persist_path = bm25_persist_path or (Path(vector_store_dir) / "bm25_index.pkl")

        # Build reverse lookup: nickname -> author_id
        self._reverse_nicknames: dict[str, int] = {}
        for author_id, nicks in self.nicknames.items():
            for nick in nicks:
                self._reverse_nicknames[nick.lower()] = author_id

        # Build author_id -> display name lookup
        self._author_names: dict[int, str] = {}

    def _ensure_bm25_index(self) -> BM25Index | None:
        """Lazy-load or auto-build BM25 index.

        If the BM25 index exists on disk, load it. Otherwise, build it
        from the vector store documents and persist it.

        Returns:
            BM25Index instance, or None if vector store is empty.
        """
        if self._bm25_index is not None:
            return self._bm25_index

        if self._bm25_persist_path.exists():
            logger.info(f"Loading BM25 index from {self._bm25_persist_path}")
            self._bm25_index = BM25Index.load(self._bm25_persist_path)
            return self._bm25_index

        # Auto-build from vector store
        docs = self.vector_store.get_all_documents()
        if not docs:
            logger.warning("Vector store is empty, cannot build BM25 index")
            return None

        logger.info(f"Building BM25 index from {len(docs)} documents...")
        self._bm25_index = BM25Index()
        chunk_ids = [d[0] for d in docs]
        documents = [d[1] for d in docs]
        self._bm25_index.build_index(chunk_ids, documents)
        self._bm25_index.save(self._bm25_persist_path)
        logger.info(f"BM25 index saved to {self._bm25_persist_path}")

        return self._bm25_index

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[BM25Result],
        k: int = 60,
    ) -> list[SearchResult]:
        """Merge vector and BM25 results using Reciprocal Rank Fusion.

        RRF combines rankings from multiple retrievers without needing
        score normalization. For each document, the RRF score is:
            RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranker i

        Args:
            vector_results: Results from vector similarity search.
            bm25_results: Results from BM25 keyword search.
            k: RRF constant (default 60, as per original paper).

        Returns:
            Merged list of SearchResult objects sorted by RRF score.
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        # Score from vector search rankings
        for rank, result in enumerate(vector_results, start=1):
            rrf_scores[result.chunk_id] += 1 / (k + rank)

        # Score from BM25 rankings
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[result.chunk_id] += 1 / (k + rank)

        # Build lookup of SearchResult objects by chunk_id
        result_lookup: dict[str, SearchResult] = {}
        for result in vector_results:
            result_lookup[result.chunk_id] = result

        # For BM25-only results, we need to create SearchResult objects
        for bm25_result in bm25_results:
            if bm25_result.chunk_id not in result_lookup:
                # Get full chunk data from vector store
                chunk_data = self.vector_store.get_chunk(bm25_result.chunk_id)
                if chunk_data:
                    result_lookup[bm25_result.chunk_id] = SearchResult(
                        chunk_id=bm25_result.chunk_id,
                        document=chunk_data["document"],
                        metadata=chunk_data.get("metadata", {}),
                        distance=0.5,  # Neutral distance for BM25-only results
                    )

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Return SearchResult objects in RRF order
        merged_results = []
        for chunk_id in sorted_ids:
            if chunk_id in result_lookup:
                merged_results.append(result_lookup[chunk_id])

        return merged_results

    async def semantic_search(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
        auto_filter_participants: bool = True,
        use_reranker: bool = True,
        use_bm25: bool | None = None,
    ) -> list[SearchResult]:
        """Perform semantic search over conversation chunks.

        By default, uses hybrid search combining vector similarity with
        BM25 keyword matching via Reciprocal Rank Fusion (RRF).

        Args:
            query: The search query text.
            k: Maximum number of results to return.
            filters: Optional metadata filters.
            auto_filter_participants: If True, automatically detect nicknames
                in the query and filter results to chunks with those participants.
            use_reranker: If True and reranker is configured, rerank results.
            use_bm25: If True, combine vector search with BM25 keyword search
                using Reciprocal Rank Fusion. If None, uses instance default.

        Returns:
            List of SearchResult objects, sorted by relevance.
        """
        # Use instance default if not specified
        if use_bm25 is None:
            use_bm25 = self.use_bm25

        # Build combined filters
        combined_filters = dict(filters) if filters else {}

        # Detect nicknames and add participant filter
        if auto_filter_participants:
            found_nicknames = self.find_nicknames_in_text(query)
            if found_nicknames:
                author_ids = list(set(found_nicknames.values()))  # Deduplicate

                if len(author_ids) == 1:
                    # Single participant filter using $contains on comma-separated IDs
                    author_id = author_ids[0]
                    combined_filters["participant_ids"] = {"$contains": str(author_id)}
                    logger.debug(f"Auto-filtering by participant_id: {author_id}")
                else:
                    # Multiple participants - require ALL to be present using $and
                    participant_filters = [
                        {"participant_ids": {"$contains": str(aid)}} for aid in author_ids
                    ]
                    if combined_filters:
                        # Combine with existing filters
                        combined_filters = {"$and": [combined_filters] + participant_filters}
                    else:
                        combined_filters = {"$and": participant_filters}
                    logger.debug(f"Auto-filtering by multiple participant_ids: {author_ids}")

        # Generate query embedding
        query_embedding = await self.embedding_client.embed_query(query)

        # Determine how many results to retrieve for each method
        # If using hybrid + reranking, get more for better fusion
        if use_bm25:
            vector_k = k * 2  # Get more for RRF fusion
            bm25_k = k * 3  # BM25 gets more since we post-filter
        elif use_reranker and self.reranker:
            vector_k = k * 2
        else:
            vector_k = k

        # Vector similarity search
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=vector_k,
            where=combined_filters if combined_filters else None,
        )

        # BM25 keyword search (hybrid mode)
        if use_bm25:
            bm25_index = self._ensure_bm25_index()
            if bm25_index is not None:
                bm25_results = bm25_index.search(query, k=bm25_k)

                # Post-filter BM25 results by metadata if filters are set
                if combined_filters and bm25_results:
                    bm25_results = self._filter_bm25_results(bm25_results, combined_filters)

                # Merge using Reciprocal Rank Fusion
                if bm25_results:
                    results = self._reciprocal_rank_fusion(vector_results, bm25_results)
                    logger.debug(
                        f"Hybrid search: {len(vector_results)} vector + "
                        f"{len(bm25_results)} BM25 -> {len(results)} merged"
                    )
                else:
                    results = vector_results
            else:
                # No BM25 index available, fall back to vector-only
                results = vector_results
        else:
            results = vector_results

        # Apply reranking if enabled and available
        if use_reranker and self.reranker and results:
            try:
                results = await self.reranker.rerank(query, results, top_k=k)
                logger.debug(f"Reranked results to top {k}")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original order")
                results = results[:k]
        elif len(results) > k:
            results = results[:k]

        return results

    def _filter_bm25_results(
        self,
        bm25_results: list[BM25Result],
        filters: dict,
    ) -> list[BM25Result]:
        """Post-filter BM25 results by metadata.

        Since BM25 doesn't support metadata filtering natively, we filter
        results after retrieval by checking chunk metadata in the vector store.

        Args:
            bm25_results: BM25 search results to filter.
            filters: Metadata filter dict.

        Returns:
            Filtered list of BM25Result objects.
        """
        filtered = []
        for result in bm25_results:
            chunk_data = self.vector_store.get_chunk(result.chunk_id)
            if chunk_data and self._matches_filters(chunk_data.get("metadata", {}), filters):
                filtered.append(result)
        return filtered

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches the given filters.

        Args:
            metadata: Chunk metadata dict.
            filters: Filter conditions to check.

        Returns:
            True if metadata matches all filter conditions.
        """
        if "$and" in filters:
            return all(self._matches_filters(metadata, cond) for cond in filters["$and"])

        for key, value in filters.items():
            if key.startswith("$"):
                continue  # Skip operators

            meta_value = metadata.get(key)
            if meta_value is None:
                return False

            if isinstance(value, dict):
                if "$contains" in value:
                    if value["$contains"] not in str(meta_value):
                        return False
            elif meta_value != value:
                return False

        return True

    async def answer_question(
        self,
        question: str,
        context_k: int = 5,
        filters: dict | None = None,
    ) -> RAGResponse:
        """Answer a question using retrieved context.

        Args:
            question: The question to answer.
            context_k: Number of context chunks to retrieve.
            filters: Optional metadata filters.

        Returns:
            RAGResponse with answer and source chunks.
        """
        # Retrieve relevant chunks
        results = await self.semantic_search(question, k=context_k, filters=filters)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer this question.",
                sources=[],
                query=question,
                success=True,  # Not an error, just no results
            )

        # Build context from search results
        context = self._build_context(results)

        # Generate answer using LLM
        answer, success, error = await self._generate_answer(question, context)

        # Build sources list
        sources = [
            {
                "chunk_id": r.chunk_id,
                "text": r.document,
                "metadata": r.metadata,
                "similarity": r.similarity,
            }
            for r in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            success=success,
            error_message=error,
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results.

        Args:
            results: List of search results to include as context.

        Returns:
            Formatted context string for LLM prompt.
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            # Include metadata in context
            metadata = result.metadata
            channel = metadata.get("channel_name", "unknown")
            year = metadata.get("year", "")
            month = metadata.get("month", "")

            header = f"[Source {i}: #{channel}"
            if year:
                header += f", {year}"
                if month:
                    header += f"-{month:02d}" if isinstance(month, int) else f"-{month}"
            header += "]"

            context_parts.append(f"{header}\n{result.document}")

        return "\n\n---\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> tuple[str, bool, str | None]:
        """Generate answer using LLM.

        Args:
            question: The question to answer.
            context: Retrieved context from vector store.

        Returns:
            Tuple of (answer_text, success, error_message).
        """
        if self.llm_client is None:
            # Fallback: return context directly if no LLM client
            return (
                f"Based on the context, here's what I found:\n\n{context}",
                True,
                None,
            )

        response = await self.llm_client.generate_answer(question, context)

        if response.success:
            return (response.text, True, None)
        else:
            # On LLM error, return context with error note
            return (
                f"(LLM generation failed: {response.error_message})\n\nContext:\n{context}",
                False,
                response.error_message,
            )

    def resolve_nickname(self, nickname: str) -> int | None:
        """Resolve a nickname to an author ID.

        Args:
            nickname: The nickname to look up (case-insensitive).

        Returns:
            The author_id if found, None otherwise.
        """
        return self._reverse_nicknames.get(nickname.lower())

    def find_nicknames_in_text(self, text: str) -> dict[str, int]:
        """Find all known nicknames mentioned in text.

        Args:
            text: Text to search for nicknames.

        Returns:
            Dict mapping found nicknames to their author IDs.
        """
        found = {}
        text_lower = text.lower()

        # Check each known nickname
        for nick, author_id in self._reverse_nicknames.items():
            # Use word boundary matching
            pattern = rf"\b{re.escape(nick)}\b"
            if re.search(pattern, text_lower):
                found[nick] = author_id

        return found

    def get_chunks_by_participant(
        self,
        author_id: int | None = None,
        nickname: str | None = None,
    ) -> list[str]:
        """Get chunk IDs containing a specific participant.

        Args:
            author_id: The author's Discord ID.
            nickname: Alternatively, the author's nickname.

        Returns:
            List of chunk IDs where the participant appears.
        """
        if nickname and not author_id:
            author_id = self.resolve_nickname(nickname)

        if not author_id:
            return []

        # Note: This is a basic implementation.
        # For production, consider adding a dedicated index.
        matching_ids = []

        # This would be more efficient with proper indexing
        # For now, we rely on metadata filters in search()
        return matching_ids

    async def get_user_insights(
        self,
        author_id: int | None = None,
        nickname: str | None = None,
        insight_type: str = "summary",
        context_k: int = 10,
    ) -> UserInsight:
        """Generate insights about a user based on their messages.

        Args:
            author_id: The user's Discord ID.
            nickname: Alternatively, the user's nickname.
            insight_type: Type of insight ("summary", "topics", "personality").
            context_k: Number of chunks to retrieve for context.

        Returns:
            UserInsight with generated analysis.
        """
        # Resolve nickname to author_id if needed
        if nickname and not author_id:
            author_id = self.resolve_nickname(nickname)

        if not author_id:
            return UserInsight(
                user_name="Unknown",
                user_id=0,
                insight="",
                insight_type=insight_type,
                success=False,
                error_message="Could not resolve user. Please provide a valid nickname or user ID.",
            )

        # Get user's nicknames
        user_nicknames = self.nicknames.get(author_id, [])
        user_name = user_nicknames[0] if user_nicknames else f"User {author_id}"

        # Search for chunks containing this user
        # Use a generic query that will be filtered by participant
        query = f"messages from {user_name}"
        filters = {"participant_ids": {"$contains": str(author_id)}}

        results = await self.semantic_search(
            query=query,
            k=context_k,
            filters=filters,
            auto_filter_participants=False,  # We're filtering manually
        )

        if not results:
            return UserInsight(
                user_name=user_name,
                user_id=author_id,
                insight=f"I couldn't find any messages from {user_name} in the conversation history.",
                insight_type=insight_type,
                success=True,
            )

        # Build context from results
        context = self._build_context(results)

        # Check for LLM client
        if self.llm_client is None:
            return UserInsight(
                user_name=user_name,
                user_id=author_id,
                insight=f"Found {len(results)} conversations involving {user_name}, but no LLM client configured for insight generation.",
                insight_type=insight_type,
                sources=[
                    {"chunk_id": r.chunk_id, "text": r.document, "metadata": r.metadata}
                    for r in results
                ],
                success=False,
                error_message="No LLM client configured",
            )

        # Generate insight
        response = await self.llm_client.generate_user_insight(
            user_name=user_name,
            user_nicknames=user_nicknames,
            context=context,
            insight_type=insight_type,
        )

        sources = [
            {"chunk_id": r.chunk_id, "text": r.document, "metadata": r.metadata} for r in results
        ]

        return UserInsight(
            user_name=user_name,
            user_id=author_id,
            insight=response.text if response.success else "",
            insight_type=insight_type,
            sources=sources,
            success=response.success,
            error_message=response.error_message,
        )

    def _parse_timeframe(self, timeframe: str) -> datetime.datetime | None:
        """Parse a timeframe string to a cutoff datetime.

        Args:
            timeframe: Timeframe string like "24h", "7d", "30d", "week", "month".

        Returns:
            Datetime cutoff, or None if invalid timeframe.
        """
        now = datetime.datetime.now(datetime.UTC)

        timeframe_map = {
            "1h": datetime.timedelta(hours=1),
            "6h": datetime.timedelta(hours=6),
            "12h": datetime.timedelta(hours=12),
            "24h": datetime.timedelta(hours=24),
            "1d": datetime.timedelta(days=1),
            "3d": datetime.timedelta(days=3),
            "7d": datetime.timedelta(days=7),
            "14d": datetime.timedelta(days=14),
            "30d": datetime.timedelta(days=30),
            "week": datetime.timedelta(weeks=1),
            "month": datetime.timedelta(days=30),
        }

        delta = timeframe_map.get(timeframe.lower())
        if delta:
            return now - delta
        return None

    async def generate_channel_recap(
        self,
        timeframe: str = "24h",
        channel_name: str | None = None,
        context_k: int = 15,
    ) -> RAGResponse:
        """Generate an activity summary/recap for a channel.

        Args:
            timeframe: Time period ("24h", "7d", "30d", "week", "month").
            channel_name: Channel to recap (None for all channels).
            context_k: Number of chunks to retrieve for context.

        Returns:
            RAGResponse with recap summary and source chunks.
        """
        # Parse timeframe
        cutoff = self._parse_timeframe(timeframe)
        if cutoff is None:
            return RAGResponse(
                answer=f"Unknown timeframe: {timeframe}. Valid options: 24h, 7d, 30d, week, month",
                sources=[],
                query=f"recap {timeframe} {channel_name or 'all'}",
                success=False,
                error_message="Invalid timeframe",
            )

        # Build filters
        filters = {"start_time": {"$gte": cutoff.isoformat()}}
        if channel_name:
            filters["channel_name"] = channel_name

        # Generic query to retrieve recent activity
        channel_ref = f"#{channel_name}" if channel_name else "server"
        query = f"recent activity discussions conversations in {channel_ref}"

        # Search without auto-participant filtering
        results = await self.semantic_search(
            query=query,
            k=context_k,
            filters=filters,
            auto_filter_participants=False,
        )

        if not results:
            return RAGResponse(
                answer=f"No activity found in {channel_ref} during the last {timeframe}.",
                sources=[],
                query=query,
                success=True,
            )

        # Build context
        context = self._build_context(results)

        # Generate recap using LLM
        if self.llm_client is None:
            # Fallback without LLM
            return RAGResponse(
                answer=f"Found {len(results)} conversation chunks from the last {timeframe}, but no LLM configured for recap generation.",
                sources=[
                    {
                        "chunk_id": r.chunk_id,
                        "text": r.document,
                        "metadata": r.metadata,
                        "similarity": r.similarity,
                    }
                    for r in results
                ],
                query=query,
                success=False,
                error_message="No LLM client configured",
            )

        response = await self.llm_client.generate_recap(
            timeframe=timeframe,
            channel_name=channel_name,
            context=context,
        )

        sources = [
            {
                "chunk_id": r.chunk_id,
                "text": r.document,
                "metadata": r.metadata,
                "similarity": r.similarity,
            }
            for r in results
        ]

        return RAGResponse(
            answer=response.text
            if response.success
            else f"Recap generation failed: {response.error_message}",
            sources=sources,
            query=query,
            success=response.success,
            error_message=response.error_message,
        )
