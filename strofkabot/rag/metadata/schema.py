"""Metadata schema for conversation chunks.

Defines dataclasses for storing chunk metadata in ChromaDB.
Follows ChromaDB best practices:
- Atomic fields (no nested JSON)
- Consistent field names
- Boolean fields for category filtering
- Integers for range queries
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedMetadata:
    """LLM-extracted metadata from conversation chunks.

    This holds the raw LLM output before flattening for ChromaDB.
    Lists are converted to comma-separated strings for storage.
    """

    mentioned_users: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    conversation_type: str = "discussion"
    summary: str = ""
    key_phrases: list[str] = field(default_factory=list)

    def to_flat_strings(self) -> dict[str, str]:
        """Convert lists to comma-separated strings for ChromaDB storage."""
        return {
            "mentioned_users": ",".join(self.mentioned_users),
            "topics": ",".join(self.topics),
            "sentiment": self.sentiment,
            "conversation_type": self.conversation_type,
            "summary": self.summary,
            "key_phrases": ",".join(self.key_phrases),
        }

    @classmethod
    def from_llm_response(cls, response: dict) -> ExtractedMetadata:
        """Create ExtractedMetadata from LLM JSON response.

        Handles missing fields gracefully with defaults.
        """
        return cls(
            mentioned_users=response.get("mentioned_users", []),
            topics=response.get("topics", []),
            sentiment=response.get("sentiment", "neutral"),
            conversation_type=response.get("conversation_type", "discussion"),
            summary=response.get("summary", ""),
            key_phrases=response.get("key_phrases", []),
        )


@dataclass
class ChunkMetadata:
    """Complete metadata for a conversation chunk.

    Designed for ChromaDB storage with atomic fields.
    All fields are either strings, integers, or booleans.
    """

    # === Identity ===
    chunk_id: str

    # === Temporal ===
    start_time: str  # ISO timestamp
    end_time: str  # ISO timestamp
    year: int
    month: int
    day_of_week: int  # 0-6 (Monday=0)
    hour_of_day: int  # 0-23

    # === Location ===
    channel_id: int
    channel_name: str

    # === Participants ===
    participant_ids: str  # Comma-separated author_ids
    participant_names: str  # Comma-separated display names
    participant_nicknames: str  # Comma-separated known nicknames
    participant_count: int

    # === Content Characteristics ===
    message_count: int
    total_chars: int
    has_reactions: bool
    reaction_count: int
    is_reply_chain: bool

    # === Extracted Entities (LLM-enriched) ===
    mentioned_users: str = ""  # Comma-separated nicknames/names
    topics: str = ""  # Comma-separated topics
    sentiment: str = "neutral"  # positive, negative, neutral, mixed
    conversation_type: str = (
        "discussion"  # discussion, banter, question_answer, announcement, debate
    )

    # === Retrieval Helpers ===
    summary: str = ""  # 1-2 sentence summary
    key_phrases: str = ""  # Comma-separated key phrases

    def to_dict(self) -> dict:
        """Convert to dictionary for ChromaDB storage."""
        return {
            "chunk_id": self.chunk_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "year": self.year,
            "month": self.month,
            "day_of_week": self.day_of_week,
            "hour_of_day": self.hour_of_day,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "participant_ids": self.participant_ids,
            "participant_names": self.participant_names,
            "participant_nicknames": self.participant_nicknames,
            "participant_count": self.participant_count,
            "message_count": self.message_count,
            "total_chars": self.total_chars,
            "has_reactions": self.has_reactions,
            "reaction_count": self.reaction_count,
            "is_reply_chain": self.is_reply_chain,
            "mentioned_users": self.mentioned_users,
            "topics": self.topics,
            "sentiment": self.sentiment,
            "conversation_type": self.conversation_type,
            "summary": self.summary,
            "key_phrases": self.key_phrases,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChunkMetadata:
        """Create ChunkMetadata from dictionary."""
        return cls(
            chunk_id=d["chunk_id"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            year=d["year"],
            month=d["month"],
            day_of_week=d["day_of_week"],
            hour_of_day=d["hour_of_day"],
            channel_id=d["channel_id"],
            channel_name=d["channel_name"],
            participant_ids=d["participant_ids"],
            participant_names=d["participant_names"],
            participant_nicknames=d["participant_nicknames"],
            participant_count=d["participant_count"],
            message_count=d["message_count"],
            total_chars=d["total_chars"],
            has_reactions=d["has_reactions"],
            reaction_count=d["reaction_count"],
            is_reply_chain=d["is_reply_chain"],
            mentioned_users=d.get("mentioned_users", ""),
            topics=d.get("topics", ""),
            sentiment=d.get("sentiment", "neutral"),
            conversation_type=d.get("conversation_type", "discussion"),
            summary=d.get("summary", ""),
            key_phrases=d.get("key_phrases", ""),
        )

    def merge_extracted(self, extracted: ExtractedMetadata) -> ChunkMetadata:
        """Merge LLM-extracted metadata into this chunk metadata.

        Returns a new ChunkMetadata with the extracted fields filled in.
        """
        flat = extracted.to_flat_strings()
        return ChunkMetadata(
            chunk_id=self.chunk_id,
            start_time=self.start_time,
            end_time=self.end_time,
            year=self.year,
            month=self.month,
            day_of_week=self.day_of_week,
            hour_of_day=self.hour_of_day,
            channel_id=self.channel_id,
            channel_name=self.channel_name,
            participant_ids=self.participant_ids,
            participant_names=self.participant_names,
            participant_nicknames=self.participant_nicknames,
            participant_count=self.participant_count,
            message_count=self.message_count,
            total_chars=self.total_chars,
            has_reactions=self.has_reactions,
            reaction_count=self.reaction_count,
            is_reply_chain=self.is_reply_chain,
            mentioned_users=flat["mentioned_users"],
            topics=flat["topics"],
            sentiment=flat["sentiment"],
            conversation_type=flat["conversation_type"],
            summary=flat["summary"],
            key_phrases=flat["key_phrases"],
        )
