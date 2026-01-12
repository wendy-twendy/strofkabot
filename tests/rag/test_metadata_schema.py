# Tests for ChunkMetadata schema (TDD - write tests first)

"""Tests for metadata schema.

These tests verify the ChunkMetadata dataclass:
- All required fields are present
- Default values work correctly
- Conversion to/from dict for ChromaDB
- Validation of field types
"""

from __future__ import annotations

from strofkabot.rag.metadata.schema import ChunkMetadata, ExtractedMetadata


class TestChunkMetadataCreation:
    """Tests for ChunkMetadata dataclass creation."""

    def test_creation_with_all_fields(self):
        """ChunkMetadata should be creatable with all fields."""
        metadata = ChunkMetadata(
            chunk_id="test-chunk-123",
            start_time="2025-10-15T14:00:00+00:00",
            end_time="2025-10-15T14:05:00+00:00",
            year=2025,
            month=10,
            day_of_week=2,
            hour_of_day=14,
            channel_id=100,
            channel_name="kanapeja",
            participant_ids="301411562487545857,686998161163812968",
            participant_names="Takarak,basstein",
            participant_nicknames="taka,bas",
            participant_count=2,
            message_count=5,
            total_chars=250,
            has_reactions=True,
            reaction_count=3,
            is_reply_chain=False,
            mentioned_users="",
            topics="gaming,music",
            sentiment="positive",
            conversation_type="banter",
            summary="Users discussed gaming and music preferences.",
            key_phrases="new album,game release",
        )

        assert metadata.chunk_id == "test-chunk-123"
        assert metadata.year == 2025
        assert metadata.participant_count == 2
        assert metadata.has_reactions is True

    def test_creation_with_defaults(self):
        """ChunkMetadata should have sensible defaults for LLM fields."""
        metadata = ChunkMetadata(
            chunk_id="test-chunk-456",
            start_time="2025-10-15T14:00:00+00:00",
            end_time="2025-10-15T14:05:00+00:00",
            year=2025,
            month=10,
            day_of_week=2,
            hour_of_day=14,
            channel_id=100,
            channel_name="kanapeja",
            participant_ids="301411562487545857",
            participant_names="Takarak",
            participant_nicknames="taka",
            participant_count=1,
            message_count=1,
            total_chars=50,
            has_reactions=False,
            reaction_count=0,
            is_reply_chain=False,
        )

        # LLM fields should have defaults
        assert metadata.mentioned_users == ""
        assert metadata.topics == ""
        assert metadata.sentiment == "neutral"
        assert metadata.conversation_type == "discussion"
        assert metadata.summary == ""
        assert metadata.key_phrases == ""


class TestChunkMetadataConversion:
    """Tests for converting ChunkMetadata to/from dict."""

    def test_to_dict(self):
        """ChunkMetadata should convert to dict for ChromaDB."""
        metadata = ChunkMetadata(
            chunk_id="test-chunk-789",
            start_time="2025-10-15T14:00:00+00:00",
            end_time="2025-10-15T14:05:00+00:00",
            year=2025,
            month=10,
            day_of_week=2,
            hour_of_day=14,
            channel_id=100,
            channel_name="kanapeja",
            participant_ids="301411562487545857",
            participant_names="Takarak",
            participant_nicknames="taka",
            participant_count=1,
            message_count=3,
            total_chars=150,
            has_reactions=True,
            reaction_count=2,
            is_reply_chain=True,
            topics="politics",
            sentiment="mixed",
        )

        d = metadata.to_dict()

        assert isinstance(d, dict)
        assert d["chunk_id"] == "test-chunk-789"
        assert d["year"] == 2025
        assert d["has_reactions"] is True
        assert d["topics"] == "politics"

    def test_from_dict(self):
        """ChunkMetadata should be creatable from dict."""
        d = {
            "chunk_id": "from-dict-chunk",
            "start_time": "2025-10-15T14:00:00+00:00",
            "end_time": "2025-10-15T14:05:00+00:00",
            "year": 2025,
            "month": 10,
            "day_of_week": 2,
            "hour_of_day": 14,
            "channel_id": 100,
            "channel_name": "kanapeja",
            "participant_ids": "123,456",
            "participant_names": "User1,User2",
            "participant_nicknames": "u1,u2",
            "participant_count": 2,
            "message_count": 4,
            "total_chars": 200,
            "has_reactions": False,
            "reaction_count": 0,
            "is_reply_chain": False,
            "mentioned_users": "taka",
            "topics": "sports",
            "sentiment": "positive",
            "conversation_type": "discussion",
            "summary": "A discussion about sports.",
            "key_phrases": "football,match",
        }

        metadata = ChunkMetadata.from_dict(d)

        assert metadata.chunk_id == "from-dict-chunk"
        assert metadata.participant_count == 2
        assert metadata.topics == "sports"

    def test_roundtrip_conversion(self):
        """Converting to dict and back should preserve all fields."""
        original = ChunkMetadata(
            chunk_id="roundtrip-test",
            start_time="2025-10-15T14:00:00+00:00",
            end_time="2025-10-15T14:05:00+00:00",
            year=2025,
            month=10,
            day_of_week=2,
            hour_of_day=14,
            channel_id=100,
            channel_name="kanapeja",
            participant_ids="301411562487545857",
            participant_names="Takarak",
            participant_nicknames="taka",
            participant_count=1,
            message_count=2,
            total_chars=100,
            has_reactions=True,
            reaction_count=5,
            is_reply_chain=True,
            mentioned_users="bas,jezi",
            topics="music,movies",
            sentiment="positive",
            conversation_type="banter",
            summary="Friends chatting about entertainment.",
            key_phrases="new movie,concert,album",
        )

        d = original.to_dict()
        restored = ChunkMetadata.from_dict(d)

        assert restored.chunk_id == original.chunk_id
        assert restored.year == original.year
        assert restored.has_reactions == original.has_reactions
        assert restored.topics == original.topics
        assert restored.summary == original.summary


class TestExtractedMetadata:
    """Tests for ExtractedMetadata dataclass (LLM output)."""

    def test_creation(self):
        """ExtractedMetadata should hold LLM extraction results."""
        extracted = ExtractedMetadata(
            mentioned_users=["taka", "bas"],
            topics=["gaming", "music"],
            sentiment="positive",
            conversation_type="banter",
            summary="Users discussed their favorite games.",
            key_phrases=["new game", "soundtrack", "release date"],
        )

        assert extracted.sentiment == "positive"
        assert len(extracted.topics) == 2
        assert "gaming" in extracted.topics

    def test_to_flat_strings(self):
        """ExtractedMetadata should convert lists to comma-separated strings."""
        extracted = ExtractedMetadata(
            mentioned_users=["taka", "bas"],
            topics=["gaming", "music", "movies"],
            sentiment="mixed",
            conversation_type="discussion",
            summary="A varied conversation.",
            key_phrases=["new release", "weekend plans"],
        )

        flat = extracted.to_flat_strings()

        assert flat["mentioned_users"] == "taka,bas"
        assert flat["topics"] == "gaming,music,movies"
        assert flat["sentiment"] == "mixed"
        assert flat["key_phrases"] == "new release,weekend plans"

    def test_from_llm_response(self):
        """ExtractedMetadata should parse LLM JSON response."""
        llm_json = {
            "mentioned_users": ["shark", "geri"],
            "topics": ["politics", "economics"],
            "sentiment": "negative",
            "conversation_type": "debate",
            "summary": "A heated political discussion.",
            "key_phrases": ["election", "policy", "government"],
        }

        extracted = ExtractedMetadata.from_llm_response(llm_json)

        assert extracted.sentiment == "negative"
        assert extracted.conversation_type == "debate"
        assert "shark" in extracted.mentioned_users

    def test_from_llm_response_with_missing_fields(self):
        """ExtractedMetadata should handle missing fields gracefully."""
        llm_json = {
            "topics": ["random"],
            "sentiment": "neutral",
        }

        extracted = ExtractedMetadata.from_llm_response(llm_json)

        assert extracted.topics == ["random"]
        assert extracted.sentiment == "neutral"
        assert extracted.mentioned_users == []
        assert extracted.summary == ""
        assert extracted.key_phrases == []

    def test_empty_response(self):
        """ExtractedMetadata should handle empty response."""
        extracted = ExtractedMetadata.from_llm_response({})

        assert extracted.mentioned_users == []
        assert extracted.topics == []
        assert extracted.sentiment == "neutral"
        assert extracted.conversation_type == "discussion"
        assert extracted.summary == ""
        assert extracted.key_phrases == []


class TestChunkMetadataValidation:
    """Tests for field validation."""

    def test_sentiment_values(self):
        """Sentiment should be one of the allowed values."""
        valid_sentiments = ["positive", "negative", "neutral", "mixed"]

        for sentiment in valid_sentiments:
            metadata = ChunkMetadata(
                chunk_id="test",
                start_time="2025-10-15T14:00:00+00:00",
                end_time="2025-10-15T14:05:00+00:00",
                year=2025,
                month=10,
                day_of_week=2,
                hour_of_day=14,
                channel_id=100,
                channel_name="test",
                participant_ids="123",
                participant_names="User",
                participant_nicknames="",
                participant_count=1,
                message_count=1,
                total_chars=10,
                has_reactions=False,
                reaction_count=0,
                is_reply_chain=False,
                sentiment=sentiment,
            )
            assert metadata.sentiment == sentiment

    def test_conversation_type_values(self):
        """Conversation type should be one of the allowed values."""
        valid_types = ["discussion", "banter", "question_answer", "announcement", "debate"]

        for conv_type in valid_types:
            metadata = ChunkMetadata(
                chunk_id="test",
                start_time="2025-10-15T14:00:00+00:00",
                end_time="2025-10-15T14:05:00+00:00",
                year=2025,
                month=10,
                day_of_week=2,
                hour_of_day=14,
                channel_id=100,
                channel_name="test",
                participant_ids="123",
                participant_names="User",
                participant_nicknames="",
                participant_count=1,
                message_count=1,
                total_chars=10,
                has_reactions=False,
                reaction_count=0,
                is_reply_chain=False,
                conversation_type=conv_type,
            )
            assert metadata.conversation_type == conv_type
