# Chunk formatter for embedding-ready text

"""Format conversation groups into chunks ready for embedding.

This module formats ConversationGroup objects into:
- Raw text for embedding (message content only)
- Formatted text for storage (with header and nicknames)
- Metadata for ChromaDB filtering
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from strofkabot.db.message_history import HistoryMessage

from .conversation_grouper import ConversationGroup

# URL pattern for extracting links from messages
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')


@dataclass
class FormattedChunk:
    """A conversation chunk formatted for embedding and storage.

    Attributes:
        chunk_id: Unique identifier (UUID)
        raw_text: Plain text for embedding (message content only)
        formatted_text: Text with header and nicknames for storage
        messages: Original HistoryMessage objects
        metadata: Dict for ChromaDB metadata filtering
    """

    chunk_id: str
    raw_text: str
    formatted_text: str
    messages: list[HistoryMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ChunkFormatter:
    """Formats conversation groups for embedding and storage.

    Args:
        nicknames: Dict mapping author_id to list of nicknames.
            If None, no nicknames will be included.
    """

    def __init__(self, nicknames: dict[int, list[str]] | None = None):
        self.nicknames = nicknames or {}

    def format_for_embedding(self, group: ConversationGroup) -> FormattedChunk:
        """Format a conversation group for embedding.

        Args:
            group: ConversationGroup to format

        Returns:
            FormattedChunk with raw_text, formatted_text, and metadata
        """
        chunk_id = str(uuid.uuid4())

        # Build raw text (just message content)
        raw_text = self._build_raw_text(group.messages)

        # Build formatted text (with header and structure)
        formatted_text = self._build_formatted_text(group)

        # Build metadata for ChromaDB
        metadata = self._build_metadata(group)

        return FormattedChunk(
            chunk_id=chunk_id,
            raw_text=raw_text,
            formatted_text=formatted_text,
            messages=group.messages,
            metadata=metadata,
        )

    def _build_raw_text(self, messages: list[HistoryMessage]) -> str:
        """Build raw text from messages (for embedding).

        This contains just the message content, preserving author names
        for context but without extra formatting.
        """
        parts = []
        for msg in messages:
            content = msg.content.strip() if msg.content else "[no text]"
            parts.append(f"{msg.author_name}: {content}")

        return "\n".join(parts)

    def _build_formatted_text(self, group: ConversationGroup) -> str:
        """Build formatted text with header and nicknames."""
        lines = []

        # Header with channel and time range
        header = self._format_header(group)
        lines.append(header)
        lines.append("")  # Empty line after header

        # Format each message
        for msg in group.messages:
            formatted_msg = self._format_message(msg)
            lines.append(formatted_msg)

        return "\n".join(lines)

    def _format_header(self, group: ConversationGroup) -> str:
        """Format the chunk header with channel, time, and participants."""
        # Time range formatting
        start = group.start_time
        end = group.end_time

        if start.date() == end.date():
            time_str = (
                f"{start.strftime('%Y-%m-%d')} {start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
            )
        else:
            time_str = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}"

        # Participant list with nicknames
        participants = self._format_participants(group)

        return f"[Channel: {group.channel_name} | {time_str}]\n[Participants: {participants}]"

    def _format_participants(self, group: ConversationGroup) -> str:
        """Format participant list with nicknames."""
        # Get unique author names from messages
        author_map: dict[int, str] = {}
        for msg in group.messages:
            if msg.author_id not in author_map:
                author_map[msg.author_id] = msg.author_name

        # Build participant strings with nicknames
        parts = []
        for author_id, author_name in author_map.items():
            nicks = self.nicknames.get(author_id, [])
            if nicks:
                nick_str = ", ".join(nicks)
                parts.append(f"{author_name} ({nick_str})")
            else:
                parts.append(author_name)

        return ", ".join(parts)

    def _format_message(self, msg: HistoryMessage) -> str:
        """Format a single message with author and optional reply indicator."""
        author_name = msg.author_name
        content = msg.content.strip() if msg.content else "[no text]"

        # Check for reply
        if msg.reply_to_id is not None and msg.reply_to_author:
            return f"{author_name} (reply to {msg.reply_to_author}): {content}"
        else:
            return f"{author_name}: {content}"

    def _extract_link_metadata(self, messages: list) -> dict:
        """Extract URL/link metadata from messages.

        Returns:
            Dict with has_links, link_count, link_domains
        """
        urls = []
        for msg in messages:
            if msg.content:
                urls.extend(URL_PATTERN.findall(msg.content))

        domains = set()
        for url in urls:
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                if domain:
                    domains.add(domain)
            except Exception:
                pass

        return {
            "has_links": len(urls) > 0,
            "link_count": len(urls),
            "link_domains": ",".join(sorted(domains)) if domains else "",
        }

    def _build_metadata(self, group: ConversationGroup) -> dict:
        """Build metadata dict for ChromaDB filtering."""
        start = group.start_time
        end = group.end_time

        # Get participant info
        participant_ids = sorted(group.participant_ids)
        participant_names = []
        participant_nicknames = []

        for msg in group.messages:
            if msg.author_id in group.participant_ids:
                if msg.author_name not in participant_names:
                    participant_names.append(msg.author_name)
                nicks = self.nicknames.get(msg.author_id, [])
                for nick in nicks:
                    if nick not in participant_nicknames:
                        participant_nicknames.append(nick)

        # Check if it's a reply chain
        is_reply_chain = any(msg.reply_to_id is not None for msg in group.messages)

        # Calculate total chars
        total_chars = sum(len(msg.content) for msg in group.messages)

        # Check for reactions
        has_reactions = False
        reaction_count = 0
        for msg in group.messages:
            if msg.reactions and msg.reactions != "[]":
                try:
                    reactions = json.loads(msg.reactions)
                    if reactions:
                        has_reactions = True
                        reaction_count += sum(r.get("count", 0) for r in reactions)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Extract link metadata
        link_meta = self._extract_link_metadata(group.messages)

        return {
            # Location
            "channel_id": group.channel_id,
            "channel_name": group.channel_name,
            # Temporal
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "year": start.year,
            "month": start.month,
            "day_of_week": start.weekday(),
            "hour_of_day": start.hour,
            # Participants
            "participant_ids": ",".join(str(p) for p in participant_ids),
            "participant_names": ",".join(participant_names),
            "participant_nicknames": ",".join(participant_nicknames),
            "participant_count": len(participant_ids),
            # Content
            "message_count": len(group.messages),
            "total_chars": total_chars,
            "has_reactions": has_reactions,
            "reaction_count": reaction_count,
            "is_reply_chain": is_reply_chain,
            # Links
            **link_meta,
        }
