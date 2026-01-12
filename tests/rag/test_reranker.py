# Tests for RAG Reranker

"""Tests for reranking functionality.

These tests verify:
- SimpleReranker keyword-based reranking
- LLMReranker score parsing
- Integration with pipeline
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.rag.reranker import LLMReranker, SimpleReranker
from strofkabot.rag.vector_store import SearchResult


def make_search_result(
    chunk_id: str,
    document: str,
    similarity: float,
    metadata: dict | None = None,
) -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        document=document,
        metadata=metadata or {},
        distance=1.0 - similarity,  # Convert similarity to distance
    )


class TestSimpleReranker:
    """Tests for SimpleReranker."""

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Should handle empty results."""
        reranker = SimpleReranker()
        result = await reranker.rerank("test query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_result(self):
        """Should return single result unchanged."""
        reranker = SimpleReranker()
        results = [make_search_result("1", "test document", 0.5)]

        reranked = await reranker.rerank("test query", results)

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "1"

    @pytest.mark.asyncio
    async def test_boosts_keyword_matches(self):
        """Should boost results with query keywords."""
        reranker = SimpleReranker()
        results = [
            make_search_result("1", "unrelated document about cats", 0.6),
            make_search_result("2", "music discussions about rock", 0.4),
            make_search_result("3", "more content about dogs", 0.5),
        ]

        reranked = await reranker.rerank("music rock", results)

        # Result with music keywords should be first
        assert reranked[0].chunk_id == "2"

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        """Should limit results to top_k."""
        reranker = SimpleReranker()
        results = [
            make_search_result("1", "doc 1", 0.5),
            make_search_result("2", "doc 2", 0.4),
            make_search_result("3", "doc 3", 0.3),
        ]

        reranked = await reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_removes_stop_words(self):
        """Should ignore stop words in query."""
        reranker = SimpleReranker()
        results = [
            make_search_result("1", "the document", 0.5),
            make_search_result("2", "music content", 0.4),
        ]

        # Query with only stop words shouldn't boost anything
        reranked = await reranker.rerank("the a an", results)

        # Should maintain similarity order
        assert reranked[0].chunk_id == "1"


class TestLLMRerankerScoreParsing:
    """Tests for LLMReranker score parsing."""

    def test_parse_simple_json(self):
        """Should parse simple JSON array."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        scores = reranker._parse_scores("[8, 5, 9]", 3)
        assert scores == [8.0, 5.0, 9.0]

    def test_parse_json_with_code_block(self):
        """Should extract JSON from markdown code block."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        text = "```json\n[7, 4, 9]\n```"
        scores = reranker._parse_scores(text, 3)
        assert scores == [7.0, 4.0, 9.0]

    def test_parse_fallback_regex(self):
        """Should extract numbers when JSON fails."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        text = "The scores are 8, 5, 9 respectively."
        scores = reranker._parse_scores(text, 3)
        assert scores == [8.0, 5.0, 9.0]

    def test_parse_fallback_uniform(self):
        """Should return uniform scores when parsing fails."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        scores = reranker._parse_scores("gibberish", 3)
        assert scores == [5.0, 5.0, 5.0]


class TestLLMRerankerAsync:
    """Tests for LLMReranker async methods."""

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Should handle empty results."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_single_result(self):
        """Should return single result unchanged."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        results = [make_search_result("1", "test doc", 0.5)]
        reranked = await reranker.rerank("query", results)

        assert len(reranked) == 1
        assert reranked[0].chunk_id == "1"

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_score(self):
        """Should reorder results based on LLM scores."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[3, 9, 5]"

        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = [
            make_search_result("1", "doc 1", 0.9),  # Score: 3
            make_search_result("2", "doc 2", 0.4),  # Score: 9
            make_search_result("3", "doc 3", 0.5),  # Score: 5
        ]

        reranked = await reranker.rerank("query", results)

        # Should be reordered by LLM score: 2 (9), 3 (5), 1 (3)
        assert reranked[0].chunk_id == "2"
        assert reranked[1].chunk_id == "3"
        assert reranked[2].chunk_id == "1"

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self):
        """Should limit results to top_k after reranking."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[9, 5, 7]"

        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_response)

        results = [
            make_search_result("1", "doc 1", 0.5),
            make_search_result("2", "doc 2", 0.5),
            make_search_result("3", "doc 3", 0.5),
        ]

        reranked = await reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_error(self):
        """Should use original similarity on LLM error."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            reranker = LLMReranker()

        # Mock LLM to raise error
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        results = [
            make_search_result("1", "doc 1", 0.9),
            make_search_result("2", "doc 2", 0.5),
        ]

        reranked = await reranker.rerank("query", results)

        # Should maintain original order (by similarity)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "1"


class TestPipelineWithReranker:
    """Tests for pipeline integration with reranker."""

    @pytest.fixture
    def pipeline_with_reranker(self, tmp_path):
        """Create pipeline with SimpleReranker."""
        from strofkabot.rag.pipeline import RAGPipeline

        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        reranker = SimpleReranker()

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
            reranker=reranker,
        )

        # Add test data
        pipeline.vector_store.add_chunks(
            chunk_ids=["music-1", "cats-1", "dogs-1"],
            embeddings=[[0.5] * 3072] * 3,
            documents=[
                "Discussion about rock music and bands",
                "Cats are cute animals",
                "Dogs playing in the park",
            ],
            metadatas=[
                {"channel_name": "muzika"},
                {"channel_name": "kanapeja"},
                {"channel_name": "kanapeja"},
            ],
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_search_with_reranker(self, pipeline_with_reranker):
        """Should apply reranking when enabled."""
        results = await pipeline_with_reranker.semantic_search(
            "rock music bands",
            k=3,
            use_reranker=True,
        )

        assert len(results) > 0
        # Music result should be boosted to top
        assert "music" in results[0].document.lower()

    @pytest.mark.asyncio
    async def test_search_without_reranker(self, pipeline_with_reranker):
        """Should skip reranking when disabled."""
        results = await pipeline_with_reranker.semantic_search(
            "rock music bands",
            k=3,
            use_reranker=False,
        )

        assert len(results) > 0
        # Results should be in original similarity order

    @pytest.mark.asyncio
    async def test_search_respects_k_with_reranker(self, pipeline_with_reranker):
        """Should return exactly k results after reranking."""
        results = await pipeline_with_reranker.semantic_search(
            "anything",
            k=2,
            use_reranker=True,
        )

        assert len(results) == 2
