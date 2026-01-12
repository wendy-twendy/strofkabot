# Conversation grouper for Layer 1 chunking

"""Conversation grouping based on reply chains and time windows.

This module implements Layer 1 of the 3-layer chunking strategy:
- Groups messages by reply chains (following reply_to_id links)
- Groups messages by time proximity (configurable window)
- Preserves channel boundaries (different channels never merge)
- Merges overlapping time-based and reply-based groups
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strofkabot.db.message_history import HistoryMessage


@dataclass
class ConversationGroup:
    """A group of related messages forming a conversation chunk.

    Attributes:
        channel_id: Discord channel ID
        channel_name: Channel name for display
        start_time: Timestamp of earliest message
        end_time: Timestamp of latest message
        messages: List of messages in chronological order
        participant_ids: Set of unique author IDs
    """

    channel_id: int
    channel_name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    messages: list[HistoryMessage] = field(default_factory=list)
    participant_ids: set[int] = field(default_factory=set)

    @property
    def message_count(self) -> int:
        """Number of messages in this group."""
        return len(self.messages)

    @property
    def duration(self) -> datetime.timedelta:
        """Time span of this conversation."""
        return self.end_time - self.start_time


class ConversationGrouper:
    """Groups messages into conversation chunks.

    Uses a two-pass algorithm:
    1. Build reply chains using Union-Find
    2. Merge groups within time windows

    Args:
        time_window_minutes: Maximum gap between messages to consider
            them part of the same conversation (default: 10)
    """

    def __init__(self, time_window_minutes: int = 10):
        self.time_window = datetime.timedelta(minutes=time_window_minutes)

    def group_messages(self, messages: list[HistoryMessage]) -> list[ConversationGroup]:
        """Group messages by reply chains and time proximity.

        Args:
            messages: List of messages to group (any order)

        Returns:
            List of ConversationGroup objects, sorted by start_time
        """
        if not messages:
            return []

        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        # Group by channel first
        by_channel: dict[int, list[HistoryMessage]] = defaultdict(list)
        for msg in sorted_messages:
            by_channel[msg.channel_id].append(msg)

        # Process each channel separately
        all_groups: list[ConversationGroup] = []
        for _channel_id, channel_messages in by_channel.items():
            groups = self._group_channel_messages(channel_messages)
            all_groups.extend(groups)

        # Sort all groups by start_time
        return sorted(all_groups, key=lambda g: g.start_time)

    def _group_channel_messages(self, messages: list[HistoryMessage]) -> list[ConversationGroup]:
        """Group messages within a single channel.

        Args:
            messages: Messages from one channel, sorted by timestamp

        Returns:
            List of ConversationGroup objects
        """
        if not messages:
            return []

        # Build message ID lookup
        msg_by_id: dict[int, HistoryMessage] = {msg.id: msg for msg in messages}

        # Union-Find data structure for grouping
        parent: dict[int, int] = {}  # message_id -> root message_id

        def find(msg_id: int) -> int:
            """Find root of a message's group with path compression."""
            if msg_id not in parent:
                parent[msg_id] = msg_id
            if parent[msg_id] != msg_id:
                parent[msg_id] = find(parent[msg_id])
            return parent[msg_id]

        def union(id1: int, id2: int) -> None:
            """Merge two message groups."""
            root1, root2 = find(id1), find(id2)
            if root1 != root2:
                # Always use smaller ID as root for consistency
                if root1 < root2:
                    parent[root2] = root1
                else:
                    parent[root1] = root2

        # Initialize all messages
        for msg in messages:
            find(msg.id)

        # Pass 1: Union by reply chains
        for msg in messages:
            if msg.reply_to_id is not None and msg.reply_to_id in msg_by_id:
                union(msg.id, msg.reply_to_id)

        # Pass 2: Union by time proximity within same eventual group
        # First, get current groups after reply chain pass
        for i in range(len(messages) - 1):
            msg1, msg2 = messages[i], messages[i + 1]
            time_diff = msg2.timestamp - msg1.timestamp

            if time_diff <= self.time_window:
                union(msg1.id, msg2.id)

        # Build final groups
        groups_dict: dict[int, list[HistoryMessage]] = defaultdict(list)
        for msg in messages:
            root = find(msg.id)
            groups_dict[root].append(msg)

        # Convert to ConversationGroup objects
        result: list[ConversationGroup] = []
        for _root_id, group_messages in groups_dict.items():
            # Messages should already be sorted, but ensure it
            group_messages.sort(key=lambda m: m.timestamp)

            group = ConversationGroup(
                channel_id=group_messages[0].channel_id,
                channel_name=group_messages[0].channel_name,
                start_time=group_messages[0].timestamp,
                end_time=group_messages[-1].timestamp,
                messages=group_messages,
                participant_ids={msg.author_id for msg in group_messages},
            )
            result.append(group)

        return sorted(result, key=lambda g: g.start_time)
