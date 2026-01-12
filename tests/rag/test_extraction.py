"""Tests for query-focused RAG extraction."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strofkabot.rag.extraction import ExtractionResult, extract_relevant_context
from strofkabot.rag.vector_store import SearchResult


@pytest.fixture
def mock_client():
    """Create a mock AsyncOpenAI client."""
    return AsyncMock()


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    return [
        SearchResult(
            chunk_id="chunk_1",
            document="[12:30] User1: I think Python is great\n[12:31] User2: Agreed!",
            metadata={"channel_name": "general", "year": 2024, "month": 6},
            distance=0.1,
        ),
        SearchResult(
            chunk_id="chunk_2",
            document="[14:00] User1: Python vs JavaScript debate continues",
            metadata={"channel_name": "tech", "year": 2024, "month": 6},
            distance=0.2,
        ),
    ]


@pytest.mark.asyncio
async def test_extract_relevant_context_success(mock_client, sample_search_results):
    """Test successful extraction."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(message=MagicMock(content="User1 expressed positive views about Python."))
            ]
        )
    )

    result = await extract_relevant_context(
        client=mock_client,
        question="What does User1 think about Python?",
        search_results=sample_search_results,
    )

    assert result.success
    assert "User1" in result.text
    assert result.sources_used == 2


@pytest.mark.asyncio
async def test_extract_relevant_context_empty_results(mock_client):
    """Test extraction with no search results."""
    result = await extract_relevant_context(
        client=mock_client,
        question="Test question",
        search_results=[],
    )

    assert result.success
    assert result.text == ""
    assert result.sources_used == 0


@pytest.mark.asyncio
async def test_extract_relevant_context_api_error(mock_client, sample_search_results):
    """Test graceful handling of API errors."""
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    result = await extract_relevant_context(
        client=mock_client,
        question="Test question",
        search_results=sample_search_results,
    )

    assert not result.success
    assert "API Error" in result.error_message


@pytest.mark.asyncio
async def test_extract_relevant_context_empty_response(mock_client, sample_search_results):
    """Test handling of empty response from model."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
    )

    result = await extract_relevant_context(
        client=mock_client,
        question="Test question",
        search_results=sample_search_results,
    )

    assert not result.success
    assert "Empty response" in result.error_message


@pytest.mark.asyncio
async def test_extract_relevant_context_respects_max_tokens(mock_client, sample_search_results):
    """Test that max_tokens is passed to the API."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Test response"))])
    )

    await extract_relevant_context(
        client=mock_client,
        question="Test question",
        search_results=sample_search_results,
        max_tokens=300,
    )

    # Verify max_tokens was passed
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 300


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_extraction_result_defaults(self):
        """Test default values."""
        result = ExtractionResult(text="test", success=True)
        assert result.error_message is None
        assert result.sources_used == 0

    def test_extraction_result_with_all_fields(self):
        """Test with all fields specified."""
        result = ExtractionResult(
            text="extracted content",
            success=True,
            error_message=None,
            sources_used=5,
        )
        assert result.text == "extracted content"
        assert result.success is True
        assert result.sources_used == 5
