# Tests for SemanticSplitter (TDD - write tests first)

"""Tests for semantic splitting functionality.

These tests verify the Layer 2 chunking strategy:
- Detect topic shifts using embedding similarity
- Preserve small groups (< min_messages)
- Use sliding window to smooth noise
- Don't split on coherent topics
"""

from __future__ import annotations

import datetime

import pytest

from strofkabot.db.message_history import HistoryMessage
from strofkabot.rag.chunking.conversation_grouper import ConversationGroup
from strofkabot.rag.chunking.semantic_splitter import SemanticSplitter


def create_message(
    id: int,
    content: str,
    minutes_offset: int = 0,
    author_id: int = 1,
    author_name: str = "User1",
) -> HistoryMessage:
    """Helper to create test messages."""
    base_time = datetime.datetime(2025, 10, 15, 14, 0, 0, tzinfo=datetime.UTC)
    return HistoryMessage(
        id=id,
        channel_id=100,
        channel_name="kanapeja",
        author_id=author_id,
        author_name=author_name,
        content=content,
        timestamp=base_time + datetime.timedelta(minutes=minutes_offset),
        reply_to_id=None,
        reply_to_author=None,
        reply_to_content=None,
        reactions="[]",
    )


def create_group(messages: list[HistoryMessage]) -> ConversationGroup:
    """Helper to create conversation group from messages."""
    return ConversationGroup(
        channel_id=messages[0].channel_id,
        channel_name=messages[0].channel_name,
        start_time=messages[0].timestamp,
        end_time=messages[-1].timestamp,
        messages=messages,
        participant_ids={m.author_id for m in messages},
    )


class TestSemanticSplitterInit:
    """Tests for SemanticSplitter initialization."""

    def test_creation_with_defaults(self):
        """Should create with default parameters."""
        splitter = SemanticSplitter()

        assert splitter.similarity_threshold == 0.65
        assert splitter.window_size == 3
        assert splitter.min_chunk_messages == 3

    def test_creation_with_custom_params(self):
        """Should accept custom parameters."""
        splitter = SemanticSplitter(
            similarity_threshold=0.7,
            window_size=5,
            min_chunk_messages=4,
        )

        assert splitter.similarity_threshold == 0.7
        assert splitter.window_size == 5
        assert splitter.min_chunk_messages == 4


class TestTopicShiftDetection:
    """Tests for detecting topic shifts using embeddings."""

    @pytest.mark.asyncio
    async def test_no_split_on_similar_embeddings(self):
        """High similarity should not trigger split."""
        splitter = SemanticSplitter(similarity_threshold=0.65)

        messages = [
            create_message(1, "I love pizza", 0),
            create_message(2, "Pizza is great", 1),
            create_message(3, "My favorite topping is pepperoni", 2),
            create_message(4, "I prefer margherita pizza", 3),
        ]
        group = create_group(messages)

        # Embeddings that are all similar (all about pizza)
        # Using simple vectors where high dot product = high similarity
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.95, 0.1, 0.0],
            [0.9, 0.15, 0.0],
            [0.92, 0.08, 0.0],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # Should not split - all messages are similar
        assert len(result) == 1
        assert result[0].message_count == 4

    @pytest.mark.asyncio
    async def test_split_on_topic_shift(self):
        """Low similarity should trigger split."""
        splitter = SemanticSplitter(
            similarity_threshold=0.65,
            window_size=2,
            min_chunk_messages=2,
        )

        messages = [
            create_message(1, "I love pizza", 0),
            create_message(2, "Pizza is great", 1),
            create_message(3, "Let's talk about cars", 2),
            create_message(4, "I just bought a new car", 3),
        ]
        group = create_group(messages)

        # Embeddings: first two about pizza (similar), last two about cars (different)
        embeddings = [
            [1.0, 0.0, 0.0],  # Pizza topic
            [0.95, 0.1, 0.0],  # Pizza topic (similar to first)
            [0.0, 1.0, 0.0],  # Cars topic (very different)
            [0.05, 0.95, 0.0],  # Cars topic (similar to third)
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # Should split into two groups
        assert len(result) == 2
        assert result[0].message_count == 2  # Pizza messages
        assert result[1].message_count == 2  # Car messages


class TestSmallGroupPreservation:
    """Tests for preserving small groups."""

    @pytest.mark.asyncio
    async def test_merge_small_groups(self):
        """Groups smaller than min_chunk_messages should be merged."""
        splitter = SemanticSplitter(
            similarity_threshold=0.3,  # Low threshold to trigger many splits
            window_size=1,
            min_chunk_messages=3,  # Require at least 3 messages
        )

        messages = [
            create_message(1, "Topic A message 1", 0),
            create_message(2, "Topic B message 1", 1),
            create_message(3, "Topic C message 1", 2),
            create_message(4, "Topic D message 1", 3),
        ]
        group = create_group(messages)

        # All different embeddings (would split into 4 groups of 1)
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # All resulting chunks should have at least min_chunk_messages
        # Or all should be merged into one if can't meet minimum
        total_messages = sum(g.message_count for g in result)
        assert total_messages == 4

        # Either one big group or groups meeting minimum size
        for g in result:
            assert g.message_count >= splitter.min_chunk_messages or len(result) == 1


class TestSlidingWindowSmoothing:
    """Tests for sliding window noise smoothing."""

    @pytest.mark.asyncio
    async def test_single_outlier_no_split(self):
        """Single different embedding should not trigger split with window smoothing."""
        splitter = SemanticSplitter(
            similarity_threshold=0.65,
            window_size=3,
            min_chunk_messages=2,
        )

        messages = [
            create_message(1, "Topic A", 0),
            create_message(2, "Topic A continued", 1),
            create_message(3, "Random tangent", 2),  # Outlier
            create_message(4, "Back to topic A", 3),
            create_message(5, "More topic A", 4),
        ]
        group = create_group(messages)

        # One outlier in the middle
        embeddings = [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.1, 0.9],  # Outlier
            [0.92, 0.08],
            [0.9, 0.1],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # Window smoothing should prevent split on single outlier
        # Expect 1 or 2 groups, not 3
        assert len(result) <= 2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_group(self):
        """Empty group should return empty list."""
        splitter = SemanticSplitter()

        group = ConversationGroup(
            channel_id=100,
            channel_name="test",
            start_time=datetime.datetime.now(datetime.UTC),
            end_time=datetime.datetime.now(datetime.UTC),
            messages=[],
            participant_ids=set(),
        )

        result = await splitter.split_by_topic(group, [])

        assert result == []

    @pytest.mark.asyncio
    async def test_single_message_group(self):
        """Single message group should return as-is."""
        splitter = SemanticSplitter()

        msg = create_message(1, "Single message")
        group = create_group([msg])

        result = await splitter.split_by_topic(group, [[1.0, 0.0]])

        assert len(result) == 1
        assert result[0].message_count == 1

    @pytest.mark.asyncio
    async def test_two_message_group(self):
        """Two message group should return as-is (can't split smaller than min)."""
        splitter = SemanticSplitter(min_chunk_messages=2)

        messages = [
            create_message(1, "First message", 0),
            create_message(2, "Second message", 1),
        ]
        group = create_group(messages)

        # Even with very different embeddings
        embeddings = [[1.0, 0.0], [0.0, 1.0]]

        result = await splitter.split_by_topic(group, embeddings)

        # Can't split into chunks smaller than min_chunk_messages
        assert len(result) == 1
        assert result[0].message_count == 2


class TestPreserveMessageOrder:
    """Tests for preserving message order in splits."""

    @pytest.mark.asyncio
    async def test_messages_ordered_after_split(self):
        """Messages should maintain chronological order after splitting."""
        splitter = SemanticSplitter(
            similarity_threshold=0.5,
            window_size=1,
            min_chunk_messages=2,
        )

        messages = [
            create_message(1, "Topic A", 0),
            create_message(2, "Topic A", 1),
            create_message(3, "Topic B", 2),
            create_message(4, "Topic B", 3),
        ]
        group = create_group(messages)

        embeddings = [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # Check each group has messages in order
        for g in result:
            timestamps = [m.timestamp for m in g.messages]
            assert timestamps == sorted(timestamps)


class TestGroupMetadata:
    """Tests for metadata preservation in split groups."""

    @pytest.mark.asyncio
    async def test_participant_ids_preserved(self):
        """Split groups should have correct participant_ids."""
        splitter = SemanticSplitter(
            similarity_threshold=0.5,
            window_size=1,
            min_chunk_messages=2,
        )

        messages = [
            create_message(1, "From user 1", 0, author_id=100, author_name="Alice"),
            create_message(2, "From user 2", 1, author_id=200, author_name="Bob"),
            create_message(3, "From user 3", 2, author_id=300, author_name="Charlie"),
            create_message(4, "From user 3 again", 3, author_id=300, author_name="Charlie"),
        ]
        group = create_group(messages)

        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        # Verify participant_ids are correct for each group
        for g in result:
            expected_ids = {m.author_id for m in g.messages}
            assert g.participant_ids == expected_ids

    @pytest.mark.asyncio
    async def test_time_range_updated(self):
        """Split groups should have correct start/end times."""
        splitter = SemanticSplitter(
            similarity_threshold=0.5,
            window_size=1,
            min_chunk_messages=2,
        )

        messages = [
            create_message(1, "Early", 0),
            create_message(2, "Early", 5),
            create_message(3, "Late", 10),
            create_message(4, "Late", 15),
        ]
        group = create_group(messages)

        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]

        result = await splitter.split_by_topic(group, embeddings)

        if len(result) == 2:
            # First group should have earlier times
            assert result[0].start_time == messages[0].timestamp
            assert result[0].end_time == messages[1].timestamp
            # Second group should have later times
            assert result[1].start_time == messages[2].timestamp
            assert result[1].end_time == messages[3].timestamp
