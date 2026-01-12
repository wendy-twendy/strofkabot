# Semantic splitter for Layer 2 chunking

"""Semantic splitting based on embedding similarity.

This module implements Layer 2 of the 3-layer chunking strategy:
- Uses embeddings to detect topic shifts
- Applies sliding window smoothing to reduce noise
- Preserves small groups by merging with neighbors
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strofkabot.db.message_history import HistoryMessage

from .conversation_grouper import ConversationGroup


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity (-1 to 1, higher = more similar)
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions must match: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Calculate element-wise mean of vectors.

    Args:
        vectors: List of embedding vectors

    Returns:
        Mean vector
    """
    if not vectors:
        return []

    dim = len(vectors[0])
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    return [val / len(vectors) for val in result]


class SemanticSplitter:
    """Splits conversation groups by semantic boundaries.

    Uses cosine similarity between embeddings to detect topic shifts.
    Applies sliding window smoothing to reduce noise from single
    off-topic messages.

    Args:
        similarity_threshold: Split when similarity drops below this (default: 0.65)
        window_size: Number of messages to average for smoothing (default: 3)
        min_chunk_messages: Minimum messages per chunk (default: 3)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        window_size: int = 3,
        min_chunk_messages: int = 3,
    ):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.min_chunk_messages = min_chunk_messages

    async def split_by_topic(
        self,
        group: ConversationGroup,
        embeddings: list[list[float]],
    ) -> list[ConversationGroup]:
        """Split a conversation group by semantic boundaries.

        Args:
            group: ConversationGroup to potentially split
            embeddings: Embedding vectors for each message (same order as group.messages)

        Returns:
            List of ConversationGroup objects (may be 1 if no split needed)
        """
        if not group.messages or not embeddings:
            return []

        if len(group.messages) == 1:
            return [group]

        if len(group.messages) != len(embeddings):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of messages ({len(group.messages)})"
            )

        # Find semantic boundaries
        boundaries = self._find_boundaries(embeddings)

        # If no boundaries, return original group
        if not boundaries:
            return [group]

        # Split messages at boundaries
        splits = self._split_at_boundaries(group.messages, boundaries)

        # Merge small chunks
        merged = self._merge_small_chunks(splits)

        # Convert to ConversationGroup objects
        return self._create_groups(merged, group)

    def _find_boundaries(self, embeddings: list[list[float]]) -> list[int]:
        """Find indices where topic shifts occur.

        Uses sliding window comparison to smooth out noise.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of boundary indices (indices after which to split)
        """
        if len(embeddings) < 2 * self.window_size:
            # Too few messages for meaningful window comparison
            return []

        boundaries = []

        # Compare sliding windows
        for i in range(self.window_size, len(embeddings) - self.window_size + 1):
            # Get window before current position
            prev_start = max(0, i - self.window_size)
            prev_window = embeddings[prev_start:i]
            prev_mean = mean_vector(prev_window)

            # Get window after current position
            next_end = min(len(embeddings), i + self.window_size)
            next_window = embeddings[i:next_end]
            next_mean = mean_vector(next_window)

            # Calculate similarity between windows
            similarity = cosine_similarity(prev_mean, next_mean)

            if similarity < self.similarity_threshold:
                # Only add if not too close to previous boundary
                if not boundaries or i - boundaries[-1] >= self.min_chunk_messages:
                    boundaries.append(i)

        return boundaries

    def _split_at_boundaries(
        self,
        messages: list[HistoryMessage],
        boundaries: list[int],
    ) -> list[list[HistoryMessage]]:
        """Split messages at boundary indices.

        Args:
            messages: List of messages
            boundaries: Indices after which to split

        Returns:
            List of message lists (splits)
        """
        splits = []
        start = 0

        for boundary in boundaries:
            splits.append(messages[start:boundary])
            start = boundary

        # Add remaining messages
        if start < len(messages):
            splits.append(messages[start:])

        return splits

    def _merge_small_chunks(
        self,
        splits: list[list[HistoryMessage]],
    ) -> list[list[HistoryMessage]]:
        """Merge chunks smaller than min_chunk_messages with neighbors.

        Args:
            splits: List of message lists

        Returns:
            List of message lists with small chunks merged
        """
        if not splits:
            return []

        if len(splits) == 1:
            return splits

        result = []
        current = list(splits[0])

        for next_split in splits[1:]:
            if len(current) < self.min_chunk_messages:
                # Current chunk too small, merge with next
                current.extend(next_split)
            elif len(next_split) < self.min_chunk_messages:
                # Next chunk too small, merge with current
                current.extend(next_split)
            else:
                # Both chunks are large enough
                result.append(current)
                current = list(next_split)

        # Don't forget the last chunk
        if current:
            if result and len(current) < self.min_chunk_messages:
                # Merge last small chunk with previous
                result[-1].extend(current)
            else:
                result.append(current)

        return result

    def _create_groups(
        self,
        splits: list[list[HistoryMessage]],
        original: ConversationGroup,
    ) -> list[ConversationGroup]:
        """Create ConversationGroup objects from message splits.

        Args:
            splits: List of message lists
            original: Original ConversationGroup (for channel info)

        Returns:
            List of ConversationGroup objects
        """
        groups = []

        for messages in splits:
            if not messages:
                continue

            group = ConversationGroup(
                channel_id=original.channel_id,
                channel_name=original.channel_name,
                start_time=messages[0].timestamp,
                end_time=messages[-1].timestamp,
                messages=messages,
                participant_ids={m.author_id for m in messages},
            )
            groups.append(group)

        return groups
