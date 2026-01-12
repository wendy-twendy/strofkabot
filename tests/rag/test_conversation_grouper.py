# Tests for ConversationGrouper (TDD - write tests first)

"""Tests for conversation grouping functionality.

These tests verify the Layer 1 chunking strategy:
- Reply chain grouping: Messages linked by reply_to_id stay together
- Time window grouping: Messages within N minutes group together
- Channel boundaries: Different channels never merge
- Overlapping groups: Reply chains within time windows merge
"""

from __future__ import annotations

import datetime

from strofkabot.db.message_history import HistoryMessage
from strofkabot.rag.chunking.conversation_grouper import ConversationGroup, ConversationGrouper


class TestConversationGroup:
    """Tests for ConversationGroup dataclass."""

    def test_creation(self, sample_messages):
        """ConversationGroup should be creatable with required fields."""
        msg = sample_messages[0]
        group = ConversationGroup(
            channel_id=msg.channel_id,
            channel_name=msg.channel_name,
            start_time=msg.timestamp,
            end_time=msg.timestamp,
            messages=[msg],
            participant_ids={msg.author_id},
        )

        assert group.channel_id == 100
        assert group.channel_name == "kanapeja"
        assert len(group.messages) == 1
        assert msg.author_id in group.participant_ids

    def test_message_count_property(self, sample_messages):
        """ConversationGroup should report correct message count."""
        group = ConversationGroup(
            channel_id=100,
            channel_name="kanapeja",
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[2].timestamp,
            messages=sample_messages[:3],
            participant_ids={sample_messages[0].author_id, sample_messages[1].author_id},
        )

        assert group.message_count == 3

    def test_duration_property(self, sample_messages):
        """ConversationGroup should calculate duration correctly."""
        group = ConversationGroup(
            channel_id=100,
            channel_name="kanapeja",
            start_time=sample_messages[0].timestamp,
            end_time=sample_messages[2].timestamp,  # 4 minutes later
            messages=sample_messages[:3],
            participant_ids={sample_messages[0].author_id},
        )

        assert group.duration == datetime.timedelta(minutes=4)


class TestGroupByTimeWindow:
    """Tests for time-based grouping."""

    def test_messages_within_window_group_together(self, sample_messages):
        """Messages within 10 min window should be in same group."""
        grouper = ConversationGrouper(time_window_minutes=10)

        # First 3 messages are within 5 minutes
        messages = sample_messages[:3]
        groups = grouper.group_messages(messages)

        assert len(groups) == 1
        assert groups[0].message_count == 3

    def test_messages_outside_window_separate(self, sample_messages):
        """Messages outside time window should be in separate groups."""
        grouper = ConversationGrouper(time_window_minutes=10)

        # Message at 0 min and message at 30 min (no reply link)
        messages = [sample_messages[0], sample_messages[3]]
        groups = grouper.group_messages(messages)

        assert len(groups) == 2
        assert groups[0].message_count == 1
        assert groups[1].message_count == 1

    def test_custom_time_window(self, sample_messages):
        """Custom time window should be respected for consecutive message gaps."""
        # With 3 min window, first 3 messages (0, 2, 4 min) are all within window
        # because consecutive gaps are 2 min each
        grouper = ConversationGrouper(time_window_minutes=3)

        messages = sample_messages[:3]
        groups = grouper.group_messages(messages)

        # All messages chain together (0->2 is 2min, 2->4 is 2min, both ≤3min)
        assert len(groups) == 1

    def test_gap_exceeds_time_window(self):
        """Messages with gap exceeding window should split into separate groups."""
        base_time = datetime.datetime(2025, 10, 15, 14, 0, 0, tzinfo=datetime.UTC)

        msg1 = HistoryMessage(
            id=1,
            channel_id=100,
            channel_name="kanapeja",
            author_id=1,
            author_name="User1",
            content="First message",
            timestamp=base_time,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )
        msg2 = HistoryMessage(
            id=2,
            channel_id=100,
            channel_name="kanapeja",
            author_id=1,
            author_name="User1",
            content="Second message",
            timestamp=base_time + datetime.timedelta(minutes=5),  # 5 min gap
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )

        grouper = ConversationGrouper(time_window_minutes=3)
        groups = grouper.group_messages([msg1, msg2])

        # 5 min gap > 3 min window, should be 2 groups
        assert len(groups) == 2

    def test_empty_messages_returns_empty(self):
        """Empty input should return empty list."""
        grouper = ConversationGrouper()
        groups = grouper.group_messages([])

        assert groups == []

    def test_single_message_returns_single_group(self, sample_messages):
        """Single message should return single group."""
        grouper = ConversationGrouper()
        groups = grouper.group_messages([sample_messages[0]])

        assert len(groups) == 1
        assert groups[0].message_count == 1


class TestGroupByReplyChains:
    """Tests for reply chain grouping."""

    def test_reply_chain_groups_together(self, sample_messages):
        """Messages linked by reply_to_id should group together."""
        grouper = ConversationGrouper(time_window_minutes=10)

        # Messages 4, 5, 6 form a reply chain (4 is parent, 5 replies to 4, 6 replies to 5)
        # Even though they span 30-46 minutes
        reply_chain = sample_messages[3:6]
        groups = grouper.group_messages(reply_chain)

        assert len(groups) == 1
        assert groups[0].message_count == 3
        assert all(msg in groups[0].messages for msg in reply_chain)

    def test_reply_to_older_message_starts_new_group(self):
        """Reply to message outside window should NOT start new group if target not present."""
        base_time = datetime.datetime(2025, 10, 15, 14, 0, 0, tzinfo=datetime.UTC)

        # Message that replies to a non-existent ID (orphan reply)
        orphan = HistoryMessage(
            id=100,
            channel_id=100,
            channel_name="kanapeja",
            author_id=1,
            author_name="User1",
            content="Replying to something old",
            timestamp=base_time,
            reply_to_id=999,  # Non-existent
            reply_to_author="OldUser",
            reply_to_content="Old message",
            reactions="[]",
        )

        grouper = ConversationGrouper()
        groups = grouper.group_messages([orphan])

        # Should still create a group (orphan replies are standalone)
        assert len(groups) == 1
        assert groups[0].message_count == 1

    def test_branching_reply_chain(self):
        """Multiple replies to same message should be in same group."""
        base_time = datetime.datetime(2025, 10, 15, 14, 0, 0, tzinfo=datetime.UTC)

        parent = HistoryMessage(
            id=1,
            channel_id=100,
            channel_name="kanapeja",
            author_id=1,
            author_name="User1",
            content="Parent message",
            timestamp=base_time,
            reply_to_id=None,
            reply_to_author=None,
            reply_to_content=None,
            reactions="[]",
        )
        reply1 = HistoryMessage(
            id=2,
            channel_id=100,
            channel_name="kanapeja",
            author_id=2,
            author_name="User2",
            content="Reply 1",
            timestamp=base_time + datetime.timedelta(minutes=20),
            reply_to_id=1,
            reply_to_author="User1",
            reply_to_content="Parent message",
            reactions="[]",
        )
        reply2 = HistoryMessage(
            id=3,
            channel_id=100,
            channel_name="kanapeja",
            author_id=3,
            author_name="User3",
            content="Reply 2",
            timestamp=base_time + datetime.timedelta(minutes=25),
            reply_to_id=1,
            reply_to_author="User1",
            reply_to_content="Parent message",
            reactions="[]",
        )

        grouper = ConversationGrouper(time_window_minutes=10)
        groups = grouper.group_messages([parent, reply1, reply2])

        # All should be in same group due to reply chain
        assert len(groups) == 1
        assert groups[0].message_count == 3


class TestChannelBoundaries:
    """Tests for channel boundary preservation."""

    def test_different_channels_never_merge(self, sample_messages):
        """Messages in different channels should never be in same group."""
        grouper = ConversationGrouper(time_window_minutes=60)  # Large window

        # All messages including the one from different channel
        groups = grouper.group_messages(sample_messages)

        # Check that channel 200 message is in its own group
        channel_200_groups = [g for g in groups if g.channel_id == 200]
        assert len(channel_200_groups) == 1
        assert channel_200_groups[0].channel_name == "muzika"

    def test_messages_sorted_by_channel_then_time(self, sample_messages):
        """Groups should be organized by channel."""
        grouper = ConversationGrouper()

        # Shuffle order to test sorting
        shuffled = [sample_messages[6], sample_messages[0], sample_messages[3]]
        groups = grouper.group_messages(shuffled)

        # Should have separate groups for each channel
        channel_ids = [g.channel_id for g in groups]
        assert 100 in channel_ids
        assert 200 in channel_ids


class TestMergeOverlappingGroups:
    """Tests for merging time-based and reply-based groups."""

    def test_reply_chain_within_time_window_merges(self, sample_messages):
        """Reply chain that falls within time window of other messages should merge."""
        grouper = ConversationGrouper(time_window_minutes=10)

        # Messages 0-2 are within 5 min, no replies
        # If a reply to message 2 comes at minute 8, it should be in same group
        base_time = sample_messages[0].timestamp

        late_reply = HistoryMessage(
            id=100,
            channel_id=100,
            channel_name="kanapeja",
            author_id=999,
            author_name="LateUser",
            content="Late reply",
            timestamp=base_time + datetime.timedelta(minutes=8),
            reply_to_id=sample_messages[2].id,
            reply_to_author=sample_messages[2].author_name,
            reply_to_content=sample_messages[2].content,
            reactions="[]",
        )

        messages = sample_messages[:3] + [late_reply]
        groups = grouper.group_messages(messages)

        # All should be in one group
        assert len(groups) == 1
        assert groups[0].message_count == 4


class TestParticipantTracking:
    """Tests for participant ID tracking."""

    def test_participants_collected(self, sample_messages):
        """All unique author IDs should be in participant_ids."""
        grouper = ConversationGrouper(time_window_minutes=10)

        messages = sample_messages[:3]  # Two different authors
        groups = grouper.group_messages(messages)

        expected_authors = {msg.author_id for msg in messages}
        assert groups[0].participant_ids == expected_authors

    def test_duplicate_authors_not_duplicated(self, sample_messages):
        """Same author appearing multiple times should only appear once in participants."""
        grouper = ConversationGrouper()

        # First 3 messages have author appearing twice
        messages = sample_messages[:3]
        groups = grouper.group_messages(messages)

        # Count unique
        assert len(groups[0].participant_ids) == 2  # Only 2 unique authors


class TestTimeRangeCalculation:
    """Tests for start_time and end_time calculation."""

    def test_time_range_spans_all_messages(self, sample_messages):
        """start_time and end_time should span all messages in group."""
        grouper = ConversationGrouper(time_window_minutes=10)

        messages = sample_messages[:3]
        groups = grouper.group_messages(messages)

        assert groups[0].start_time == messages[0].timestamp
        assert groups[0].end_time == messages[2].timestamp

    def test_single_message_same_start_end(self, sample_messages):
        """Single message should have same start and end time."""
        grouper = ConversationGrouper()

        groups = grouper.group_messages([sample_messages[0]])

        assert groups[0].start_time == groups[0].end_time


class TestMessageOrdering:
    """Tests for message ordering within groups."""

    def test_messages_ordered_by_timestamp(self, sample_messages):
        """Messages within a group should be ordered by timestamp."""
        grouper = ConversationGrouper(time_window_minutes=10)

        # Shuffle input order
        shuffled = [sample_messages[2], sample_messages[0], sample_messages[1]]
        groups = grouper.group_messages(shuffled)

        # Should be sorted by timestamp in output
        timestamps = [msg.timestamp for msg in groups[0].messages]
        assert timestamps == sorted(timestamps)
