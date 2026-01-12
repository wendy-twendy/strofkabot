# Tests for MetadataExtractor (TDD - write tests first)

"""Tests for LLM-based metadata extraction.

These tests verify the MetadataExtractor:
- Extracts topics from conversation text
- Identifies sentiment correctly
- Classifies conversation type
- Generates summaries
- Handles API errors gracefully
- Parses LLM JSON responses
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.rag.metadata.extractor import ExtractionResponse, MetadataExtractor
from strofkabot.rag.metadata.schema import ExtractedMetadata


class TestExtractionResponse:
    """Tests for ExtractionResponse dataclass."""

    def test_creation_success(self):
        """ExtractionResponse should be creatable with success=True."""
        extracted = ExtractedMetadata(
            topics=["gaming"],
            sentiment="positive",
            summary="A chat about games.",
        )
        response = ExtractionResponse(
            metadata=extracted,
            success=True,
            error_message=None,
        )

        assert response.success is True
        assert response.metadata.sentiment == "positive"

    def test_creation_failure(self):
        """ExtractionResponse should be creatable with success=False."""
        response = ExtractionResponse(
            metadata=None,
            success=False,
            error_message="API error",
        )

        assert response.success is False
        assert response.error_message == "API error"
        assert response.metadata is None


class TestMetadataExtractorInit:
    """Tests for extractor initialization."""

    def test_missing_api_key_raises(self):
        """Should raise ValueError if no API key provided or in env."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                MetadataExtractor()

    def test_init_with_explicit_key(self):
        """Should accept explicit API key."""
        with patch("strofkabot.rag.metadata.extractor.AsyncOpenAI"):
            extractor = MetadataExtractor(api_key="test-key")
            assert extractor is not None

    def test_model_configuration(self):
        """Extractor should use gemini-2.0-flash-lite-001 model."""
        with patch("strofkabot.rag.metadata.extractor.AsyncOpenAI"):
            extractor = MetadataExtractor(api_key="test-key")
            assert "gemini" in extractor.model.lower()
            assert "flash" in extractor.model.lower() or "lite" in extractor.model.lower()


class TestExtractMetadata:
    """Tests for extract method."""

    @pytest.fixture
    def mock_extractor(self):
        """Create an extractor with mocked AsyncOpenAI."""
        with patch("strofkabot.rag.metadata.extractor.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            extractor = MetadataExtractor(api_key="test-key")
            yield extractor, mock_instance

    @pytest.mark.asyncio
    async def test_extract_topics(self, mock_extractor):
        """Should extract topics from conversation."""
        extractor, mock_openai = mock_extractor

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "mentioned_users": [],
                "topics": ["gaming", "music"],
                "sentiment": "positive",
                "conversation_type": "banter",
                "summary": "Friends discussing games and music.",
                "key_phrases": ["new game", "album release"],
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        chunk_text = """
        Takarak: Hey, did you play the new game?
        basstein: Yeah it's amazing! The soundtrack is great too.
        Takarak: I know right, reminds me of the new album.
        """

        response = await extractor.extract(chunk_text, ["Takarak", "basstein"])

        assert response.success is True
        assert "gaming" in response.metadata.topics
        assert "music" in response.metadata.topics

    @pytest.mark.asyncio
    async def test_extract_sentiment(self, mock_extractor):
        """Should correctly identify sentiment."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "mentioned_users": [],
                "topics": ["politics"],
                "sentiment": "negative",
                "conversation_type": "debate",
                "summary": "A heated political discussion.",
                "key_phrases": ["government", "policy"],
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        chunk_text = """
        User1: This policy is terrible.
        User2: I completely disagree with everything about it.
        """

        response = await extractor.extract(chunk_text, ["User1", "User2"])

        assert response.success is True
        assert response.metadata.sentiment == "negative"

    @pytest.mark.asyncio
    async def test_extract_mentioned_users(self, mock_extractor):
        """Should identify users mentioned in conversation (not speakers)."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "mentioned_users": ["shark", "geri"],
                "topics": ["friends"],
                "sentiment": "neutral",
                "conversation_type": "discussion",
                "summary": "Talking about shark and geri.",
                "key_phrases": ["shark said", "geri thinks"],
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        chunk_text = """
        Takarak: Did you hear what shark said yesterday?
        basstein: No, what happened? Was geri there too?
        """

        response = await extractor.extract(chunk_text, ["Takarak", "basstein"])

        assert response.success is True
        assert "shark" in response.metadata.mentioned_users
        assert "geri" in response.metadata.mentioned_users

    @pytest.mark.asyncio
    async def test_extract_conversation_type(self, mock_extractor):
        """Should classify conversation type correctly."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "mentioned_users": [],
                "topics": ["help"],
                "sentiment": "neutral",
                "conversation_type": "question_answer",
                "summary": "User asking for help with code.",
                "key_phrases": ["how do I", "you can use"],
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        chunk_text = """
        User1: How do I fix this error?
        User2: You can use the debug mode to find the issue.
        """

        response = await extractor.extract(chunk_text, ["User1", "User2"])

        assert response.success is True
        assert response.metadata.conversation_type == "question_answer"

    @pytest.mark.asyncio
    async def test_error_response_on_api_failure(self, mock_extractor):
        """Should return success=False on API error."""
        extractor, mock_openai = mock_extractor

        mock_openai.chat.completions.create = AsyncMock(side_effect=Exception("API rate limit"))

        response = await extractor.extract("Some text", ["User"])

        assert response.success is False
        assert "rate limit" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, mock_extractor):
        """Should handle invalid JSON in LLM response."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON {{"

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await extractor.extract("Some text", ["User"])

        assert response.success is False
        assert "json" in response.error_message.lower() or "parse" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_handles_partial_json(self, mock_extractor):
        """Should handle partial JSON with missing fields."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Only some fields present
        mock_response.choices[0].message.content = json.dumps(
            {
                "topics": ["random"],
                "sentiment": "positive",
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await extractor.extract("Some text", ["User"])

        assert response.success is True
        assert response.metadata.topics == ["random"]
        assert response.metadata.sentiment == "positive"
        # Defaults for missing fields
        assert response.metadata.mentioned_users == []
        assert response.metadata.summary == ""

    @pytest.mark.asyncio
    async def test_albanian_text(self, mock_extractor):
        """Should handle Albanian text correctly."""
        extractor, mock_openai = mock_extractor

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "mentioned_users": [],
                "topics": ["greeting", "daily life"],
                "sentiment": "positive",
                "conversation_type": "banter",
                "summary": "Friends greeting each other in Albanian.",
                "key_phrases": ["si jeni", "mirë"],
            }
        )

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        chunk_text = """
        Takarak: Ç'kemi, si jeni sot?
        basstein: Mirë faleminderit, ti si je?
        """

        response = await extractor.extract(chunk_text, ["Takarak", "basstein"])

        assert response.success is True
        assert response.metadata.sentiment == "positive"


class TestExtractBatch:
    """Tests for batch extraction."""

    @pytest.fixture
    def mock_extractor(self):
        """Create an extractor with mocked AsyncOpenAI."""
        with patch("strofkabot.rag.metadata.extractor.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            extractor = MetadataExtractor(api_key="test-key")
            yield extractor, mock_instance

    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_extractor):
        """Should extract metadata for multiple chunks."""
        extractor, mock_openai = mock_extractor

        # Create mock responses for batch
        def create_mock_response(topic):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = json.dumps(
                {
                    "topics": [topic],
                    "sentiment": "neutral",
                    "conversation_type": "discussion",
                    "summary": f"Discussion about {topic}.",
                    "key_phrases": [topic],
                }
            )
            return mock_resp

        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[
                create_mock_response("gaming"),
                create_mock_response("music"),
                create_mock_response("sports"),
            ]
        )

        chunks = [
            ("Chunk about gaming", ["User1"]),
            ("Chunk about music", ["User2"]),
            ("Chunk about sports", ["User3"]),
        ]

        responses = await extractor.extract_batch(chunks)

        assert len(responses) == 3
        assert all(r.success for r in responses)
        assert responses[0].metadata.topics == ["gaming"]
        assert responses[1].metadata.topics == ["music"]
        assert responses[2].metadata.topics == ["sports"]


# E2E test - requires real API key
@pytest.mark.e2e
class TestMetadataExtractorE2E:
    """End-to-end tests with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_extract_real(self):
        """Should extract real metadata from OpenRouter API."""
        import os

        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        extractor = MetadataExtractor()

        chunk_text = """
        Takarak: Hey did you watch the football match yesterday?
        basstein: Yeah it was incredible! That last minute goal was insane.
        Takarak: I know right, I was screaming at the TV.
        basstein: Same here, my neighbors probably hate me now lol
        """

        response = await extractor.extract(chunk_text, ["Takarak", "basstein"])

        assert response.success is True
        assert response.metadata is not None
        # Should detect sports/football topic
        assert len(response.metadata.topics) > 0
        # Should have positive or mixed sentiment (excitement)
        assert response.metadata.sentiment in ["positive", "mixed", "neutral"]
        # Should have a summary
        assert len(response.metadata.summary) > 0
