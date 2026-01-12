"""Batch ingestion pipeline for RAG with progress tracking.

Handles:
- Loading messages from database in memory-efficient batches
- Chunking with conversation grouping
- Embedding generation with parallel batching
- Incremental saving to vector store (no OOM)
- Resumable progress tracking
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .bm25_index import BM25Index
from .chunking.chunk_formatter import ChunkFormatter
from .chunking.conversation_grouper import ConversationGrouper
from .embeddings import OpenRouterEmbeddingClient
from .metadata.extractor import MetadataExtractor
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionProgress:
    """Progress tracking for ingestion pipeline."""

    total_messages: int = 0
    processed_messages: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    embedded_chunks: int = 0
    metadata_extracted: int = 0
    bm25_indexed: int = 0
    started_at: str | None = None
    last_updated: str | None = None
    status: str = "not_started"
    error: str | None = None
    # Track completed time periods for resumability
    completed_periods: list[str] = field(default_factory=list)
    current_period: str | None = None

    def to_dict(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "processed_messages": self.processed_messages,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "embedded_chunks": self.embedded_chunks,
            "metadata_extracted": self.metadata_extracted,
            "bm25_indexed": self.bm25_indexed,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "status": self.status,
            "error": self.error,
            "completed_periods": self.completed_periods,
            "current_period": self.current_period,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IngestionProgress:
        # Handle old progress files without new fields
        data = data.copy()
        data.setdefault("completed_periods", [])
        data.setdefault("current_period", None)
        return cls(**data)


@dataclass
class HistoryMessage:
    """Message from message_history table."""

    id: int
    channel_id: int
    channel_name: str
    author_id: int
    author_name: str
    content: str
    timestamp: datetime.datetime
    reply_to_id: int | None = None
    reply_to_author: str | None = None
    reply_to_content: str | None = None
    reactions: str = "[]"


class IngestionPipeline:
    """Pipeline for batch ingestion of messages into vector store.

    Args:
        db_path: Path to SQLite database with message_history table.
        vector_store_dir: Directory for vector store persistence.
        nicknames: Dict mapping author_id -> list of nicknames.
        progress_file: Path to save progress state.
        time_window_minutes: Time window for conversation grouping.
        merge_small_chunks: Whether to merge small chunks.
        embedding_batch_size: Batch size for embedding API.
    """

    def __init__(
        self,
        db_path: Path,
        vector_store_dir: Path,
        nicknames: dict[int, list[str]],
        progress_file: Path | None = None,
        time_window_minutes: int = 30,
        merge_small_chunks: bool = True,
        embedding_batch_size: int = 50,
        bm25_persist_path: Path | None = None,
        max_chunk_chars: int = 8000,
    ):
        self.db_path = db_path
        self.vector_store_dir = vector_store_dir
        self.nicknames = nicknames
        self.progress_file = progress_file or vector_store_dir / "ingestion_progress.json"
        self.time_window_minutes = time_window_minutes
        self.merge_small_chunks = merge_small_chunks
        self.embedding_batch_size = embedding_batch_size
        self.bm25_persist_path = bm25_persist_path or vector_store_dir / "bm25_index.pkl"
        self.max_chunk_chars = max_chunk_chars  # ~2000 tokens limit per chunk

        self.vector_store = VectorStore(
            persist_dir=vector_store_dir,
            collection_name="strofka_messages",
        )
        self.embedding_client = OpenRouterEmbeddingClient()
        self.grouper = ConversationGrouper(time_window_minutes=time_window_minutes)
        self.formatter = ChunkFormatter(nicknames=nicknames)

        self.progress = self._load_progress()

    def _load_progress(self) -> IngestionProgress:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                return IngestionProgress.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
        return IngestionProgress()

    def _save_progress(self) -> None:
        """Save progress to file."""
        self.progress.last_updated = datetime.datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    def _get_date_range(
        self,
        after_date: datetime.datetime | None = None,
        before_date: datetime.datetime | None = None,
    ) -> tuple[datetime.datetime, datetime.datetime]:
        """Get the date range of messages in the database.

        Args:
            after_date: Optional minimum date override.
            before_date: Optional maximum date override.

        Returns:
            Tuple of (min_date, max_date).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM message_history")
        row = cursor.fetchone()
        conn.close()

        db_min = datetime.datetime.fromisoformat(row[0]) if row[0] else datetime.datetime.now()
        db_max = datetime.datetime.fromisoformat(row[1]) if row[1] else datetime.datetime.now()

        # Apply user overrides
        min_date = after_date if after_date else db_min
        max_date = before_date if before_date else db_max

        return min_date, max_date

    def _get_total_message_count(
        self,
        after_date: datetime.datetime | None = None,
        before_date: datetime.datetime | None = None,
    ) -> int:
        """Get total message count in date range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM message_history WHERE 1=1"
        params = []

        if after_date:
            query += " AND timestamp > ?"
            params.append(after_date.isoformat())

        if before_date:
            query += " AND timestamp < ?"
            params.append(before_date.isoformat())

        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def load_messages(
        self,
        limit: int | None = None,
        channel_name: str | None = None,
        after_date: datetime.datetime | None = None,
        before_date: datetime.datetime | None = None,
    ) -> list[HistoryMessage]:
        """Load messages from database.

        Args:
            limit: Maximum number of messages to load.
            channel_name: Filter by channel name.
            after_date: Only load messages after this date.
            before_date: Only load messages before this date.

        Returns:
            List of HistoryMessage objects.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT id, channel_id, channel_name, author_id, author_name,
                   content, timestamp, reply_to_id, reply_to_author, reply_to_content, reactions
            FROM message_history
            WHERE 1=1
        """
        params = []

        if channel_name:
            query += " AND channel_name = ?"
            params.append(channel_name)

        if after_date:
            query += " AND timestamp > ?"
            params.append(after_date.isoformat())

        if before_date:
            query += " AND timestamp < ?"
            params.append(before_date.isoformat())

        query += " ORDER BY timestamp"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            timestamp = datetime.datetime.fromisoformat(row[6])
            msg = HistoryMessage(
                id=row[0],
                channel_id=row[1],
                channel_name=row[2],
                author_id=row[3],
                author_name=row[4],
                content=row[5] or "",
                timestamp=timestamp,
                reply_to_id=row[7],
                reply_to_author=row[8],
                reply_to_content=row[9],
                reactions=row[10] or "[]",
            )
            messages.append(msg)

        return messages

    def create_chunks(self, messages: list[HistoryMessage]) -> list:
        """Create chunks from messages.

        Args:
            messages: List of messages to chunk.

        Returns:
            List of FormattedChunk objects.
        """
        # Group messages into conversations
        groups = self.grouper.group_messages(messages)

        # Format chunks
        chunks = [self.formatter.format_for_embedding(g) for g in groups]

        # Split large chunks to keep token counts manageable
        chunks = self._split_large_chunks(chunks)

        # Optionally merge small chunks
        if self.merge_small_chunks:
            chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_large_chunks(self, chunks: list) -> list:
        """Split chunks that exceed max_chunk_chars into smaller pieces.

        Large conversation groups (e.g., 30+ minutes of active chat) can produce
        chunks with 10k-70k characters, which wastes embedding tokens and reduces
        retrieval precision. This splits them by message count.

        Args:
            chunks: List of FormattedChunk objects.

        Returns:
            List of chunks with large ones split into smaller pieces.
        """
        if not chunks or self.max_chunk_chars <= 0:
            return chunks

        from .chunking.conversation_grouper import ConversationGroup

        result = []
        for chunk in chunks:
            if len(chunk.raw_text) <= self.max_chunk_chars:
                result.append(chunk)
                continue

            # Need to split this chunk
            messages = chunk.messages
            if len(messages) <= 1:
                # Can't split a single message, keep as-is
                result.append(chunk)
                continue

            # Estimate messages per sub-chunk (use 70% of max to account for variance)
            avg_msg_chars = len(chunk.raw_text) / len(messages)
            target_chars = self.max_chunk_chars * 0.7
            msgs_per_chunk = max(1, int(target_chars / avg_msg_chars))

            # Split messages into sub-groups and re-format
            for i in range(0, len(messages), msgs_per_chunk):
                sub_messages = messages[i : i + msgs_per_chunk]
                if not sub_messages:
                    continue

                # Create a new ConversationGroup for these messages
                sub_group = ConversationGroup(
                    channel_id=sub_messages[0].channel_id,
                    channel_name=sub_messages[0].channel_name,
                    start_time=sub_messages[0].timestamp,
                    end_time=sub_messages[-1].timestamp,
                    messages=sub_messages,
                    participant_ids={msg.author_id for msg in sub_messages},
                )

                # Format as new chunk
                sub_chunk = self.formatter.format_for_embedding(sub_group)
                result.append(sub_chunk)

        return result

    def _merge_small_chunks(self, chunks: list, min_chars: int = 100) -> list:
        """Merge small chunks with neighbors."""
        if not chunks:
            return chunks

        # Sort by channel and time
        sorted_chunks = sorted(
            chunks, key=lambda c: (c.metadata["channel_name"], c.metadata["start_time"])
        )

        merged = []
        i = 0
        while i < len(sorted_chunks):
            current = sorted_chunks[i]
            current_chars = current.metadata["total_chars"]

            # If small chunk, try to merge with next in same channel
            if current_chars < min_chars and i + 1 < len(sorted_chunks):
                next_chunk = sorted_chunks[i + 1]
                if next_chunk.metadata["channel_name"] == current.metadata["channel_name"]:
                    # Merge current into next (prepend)
                    merged_text = current.formatted_text + "\n\n" + next_chunk.formatted_text
                    merged_raw = current.raw_text + "\n" + next_chunk.raw_text

                    # Update next chunk's metadata
                    next_chunk.formatted_text = merged_text
                    next_chunk.raw_text = merged_raw
                    next_chunk.metadata["total_chars"] = (
                        current_chars + next_chunk.metadata["total_chars"]
                    )
                    next_chunk.metadata["message_count"] = (
                        current.metadata["message_count"] + next_chunk.metadata["message_count"]
                    )
                    next_chunk.metadata["start_time"] = current.metadata["start_time"]
                    next_chunk.messages = current.messages + next_chunk.messages

                    # Skip current, will process merged next
                    i += 1
                    continue

            merged.append(current)
            i += 1

        return merged

    async def generate_embeddings(
        self,
        chunks: list,
        max_concurrent: int = 50,
        on_progress: callable | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for chunks with parallel processing.

        Args:
            chunks: List of chunks to embed.
            max_concurrent: Maximum concurrent API requests (default 50).
            on_progress: Optional callback(processed, total) for progress.

        Returns:
            List of embedding vectors.
        """
        texts = [c.raw_text for c in chunks]
        response = await self.embedding_client.embed_texts(
            texts,
            batch_size=self.embedding_batch_size,
            max_concurrent=max_concurrent,
            max_retries=3,
        )

        if not response.success:
            # Partial failure - can't continue with mismatched chunk/embedding counts
            raise RuntimeError(
                f"Embedding generation failed for {len(response.failed_indices)} chunks: "
                f"{response.error_message}"
            )

        return response.embeddings

    async def extract_metadata(
        self,
        chunks: list,
        extractor: MetadataExtractor,
        on_progress: callable | None = None,
    ) -> list[dict]:
        """Extract LLM metadata for chunks.

        Args:
            chunks: List of chunks.
            extractor: MetadataExtractor instance.
            on_progress: Optional callback(processed, total) for progress.

        Returns:
            List of updated metadata dicts.
        """
        metadatas = [c.metadata.copy() for c in chunks]

        for i, chunk in enumerate(chunks):
            participants = chunk.metadata.get("participant_names", "").split(",")
            result = await extractor.extract(chunk.formatted_text, participants)

            if result.success and result.metadata:
                flat = result.metadata.to_flat_strings()
                metadatas[i] = {**metadatas[i], **flat}

            if on_progress:
                on_progress(i + 1, len(chunks))

            self.progress.metadata_extracted = i + 1
            if (i + 1) % 10 == 0:
                self._save_progress()

        return metadatas

    async def run(
        self,
        message_limit: int | None = None,
        channel_name: str | None = None,
        after_date: datetime.datetime | None = None,
        before_date: datetime.datetime | None = None,
        extract_metadata: bool = False,
        clear_existing: bool = False,
        on_progress: callable | None = None,
        batch_days: int = 30,
        resume: bool = True,
    ) -> IngestionProgress:
        """Run the full ingestion pipeline with memory-efficient batching.

        Processes messages in time-based batches to avoid OOM. Each batch is:
        1. Loaded from database
        2. Chunked
        3. Embedded (parallel within batch)
        4. Saved to vector store immediately
        5. Memory freed before next batch

        Args:
            message_limit: Maximum messages to process (applies per batch if set).
            channel_name: Filter by channel.
            after_date: Only process messages after this date.
            before_date: Only process messages before this date.
            extract_metadata: Whether to extract LLM metadata.
            clear_existing: Clear existing vector store first.
            on_progress: Callback(stage, current, total) for progress.
            batch_days: Number of days to process per batch (default 30).
            resume: Whether to resume from previous progress (default True).

        Returns:
            IngestionProgress with final stats.
        """
        # Get date range
        min_date, max_date = self._get_date_range(after_date, before_date)
        total_messages = self._get_total_message_count(after_date, before_date)

        # Initialize or resume progress
        if resume and self.progress.status == "running" and self.progress.completed_periods:
            logger.info(f"Resuming from {len(self.progress.completed_periods)} completed periods")
        else:
            self.progress = IngestionProgress(
                started_at=datetime.datetime.now().isoformat(),
                status="running",
                total_messages=total_messages,
            )

        self._save_progress()

        try:
            # Clear existing if requested (only on fresh start)
            if clear_existing and not self.progress.completed_periods:
                logger.info("Clearing existing vector store...")
                self.vector_store.clear()
                # Also clear BM25 index
                if self.bm25_persist_path.exists():
                    self.bm25_persist_path.unlink()

            # Generate time periods
            periods = self._generate_time_periods(min_date, max_date, batch_days)
            logger.info(
                f"Processing {total_messages:,} messages in {len(periods)} batches "
                f"({batch_days} days each)"
            )

            # Process each time period
            for period_start, period_end in periods:
                period_key = f"{period_start.date()}_{period_end.date()}"

                # Skip already completed periods
                if period_key in self.progress.completed_periods:
                    logger.debug(f"Skipping completed period: {period_key}")
                    continue

                self.progress.current_period = period_key
                self._save_progress()

                # Load messages for this period
                if on_progress:
                    on_progress("loading", self.progress.processed_messages, total_messages)

                messages = self.load_messages(
                    limit=message_limit,
                    channel_name=channel_name,
                    after_date=period_start,
                    before_date=period_end,
                )

                if not messages:
                    logger.debug(f"No messages in period {period_key}")
                    self.progress.completed_periods.append(period_key)
                    self._save_progress()
                    continue

                logger.info(f"Processing period {period_key}: {len(messages):,} messages")

                # Create chunks
                chunks = self.create_chunks(messages)
                if not chunks:
                    logger.debug(f"No chunks created for period {period_key}")
                    self.progress.processed_messages += len(messages)
                    self.progress.completed_periods.append(period_key)
                    self._save_progress()
                    del messages
                    gc.collect()
                    continue

                # Generate embeddings (parallel within batch)
                if on_progress:
                    on_progress("embedding", self.progress.embedded_chunks, 0)

                embeddings = await self.generate_embeddings(chunks)

                # Optional metadata extraction
                metadatas = [c.metadata for c in chunks]
                if extract_metadata:
                    extractor = MetadataExtractor()
                    metadatas = await self.extract_metadata(chunks, extractor, on_progress)

                # Save to vector store immediately
                self.vector_store.add_chunks(
                    chunk_ids=[c.chunk_id for c in chunks],
                    embeddings=embeddings,
                    documents=[c.formatted_text for c in chunks],
                    metadatas=metadatas,
                )

                # Update progress
                self.progress.processed_messages += len(messages)
                self.progress.total_chunks += len(chunks)
                self.progress.processed_chunks += len(chunks)
                self.progress.embedded_chunks += len(chunks)
                self.progress.completed_periods.append(period_key)
                self._save_progress()

                logger.info(
                    f"Completed period {period_key}: {len(chunks)} chunks saved. "
                    f"Total: {self.progress.embedded_chunks:,} chunks"
                )

                # Free memory before next batch
                del messages, chunks, embeddings, metadatas
                gc.collect()

            # Build BM25 index from all stored chunks
            if on_progress:
                on_progress("bm25_indexing", 0, self.progress.embedded_chunks)
            logger.info("Building BM25 index from stored chunks...")

            all_docs = self.vector_store.get_all_documents()
            if all_docs:
                bm25_index = BM25Index()
                # get_all_documents returns list of (chunk_id, document) tuples
                chunk_ids = [doc[0] for doc in all_docs]
                documents = [doc[1] for doc in all_docs]
                bm25_index.build_index(chunk_ids=chunk_ids, documents=documents)
                bm25_index.save(self.bm25_persist_path)
                self.progress.bm25_indexed = len(all_docs)
                logger.info(f"Built BM25 index with {len(all_docs):,} documents")

            self.progress.status = "completed"
            self.progress.current_period = None
            self._save_progress()

            logger.info(
                f"Ingestion completed: {self.progress.embedded_chunks:,} chunks "
                f"from {self.progress.processed_messages:,} messages"
            )

        except Exception as e:
            logger.exception("Ingestion failed")
            self.progress.status = "failed"
            self.progress.error = str(e)
            self._save_progress()
            raise

        return self.progress

    def _generate_time_periods(
        self,
        min_date: datetime.datetime,
        max_date: datetime.datetime,
        batch_days: int,
    ) -> list[tuple[datetime.datetime, datetime.datetime]]:
        """Generate time periods for batched processing.

        Args:
            min_date: Start date.
            max_date: End date.
            batch_days: Days per batch.

        Returns:
            List of (start, end) tuples.
        """
        periods = []
        current = min_date
        delta = datetime.timedelta(days=batch_days)

        while current < max_date:
            period_end = min(current + delta, max_date)
            periods.append((current, period_end))
            current = period_end

        return periods

    def get_progress(self) -> IngestionProgress:
        """Get current progress."""
        return self.progress
