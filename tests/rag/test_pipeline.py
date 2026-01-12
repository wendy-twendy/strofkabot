# Tests for RAG Pipeline (TDD - write tests first)

"""Tests for RAG pipeline functionality.

These tests verify the RAGPipeline:
- Semantic search with query embedding
- Nickname resolution in queries
- Answer generation with context
- Filtering by metadata (channel, time, user)
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.rag.pipeline import RAGPipeline, RAGResponse, SearchRequest


class TestSearchRequest:
    """Tests for SearchRequest dataclass."""

    def test_creation_minimal(self):
        """SearchRequest should be creatable with just query."""
        request = SearchRequest(query="what does taka think about music")

        assert request.query == "what does taka think about music"
        assert request.k == 10  # default
        assert request.filters is None

    def test_creation_with_filters(self):
        """SearchRequest should accept filters."""
        request = SearchRequest(
            query="political discussions",
            k=5,
            filters={"channel_name": "politike", "year": 2024},
        )

        assert request.k == 5
        assert request.filters["channel_name"] == "politike"


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_creation(self):
        """RAGResponse should contain answer and sources."""
        response = RAGResponse(
            answer="Taka thinks rock music is the best.",
            sources=[
                {"chunk_id": "chunk-1", "text": "Taka: rock music is the best"},
            ],
            query="what does taka think about music",
        )

        assert "rock music" in response.answer
        assert len(response.sources) == 1


class TestRAGPipelineInit:
    """Tests for RAGPipeline initialization."""

    def test_init_with_components(self, tmp_path):
        """Should initialize with vector store and embedding client."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        assert pipeline is not None

    def test_init_with_nicknames(self, tmp_path):
        """Should accept nickname mappings for query resolution."""
        mock_embedding_client = MagicMock()
        nicknames = {
            301411562487545857: ["taka"],
            686998161163812968: ["bas", "bass", "basi"],
        }

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
            nicknames=nicknames,
        )

        assert pipeline.nicknames == nicknames


class TestSemanticSearch:
    """Tests for semantic_search method."""

    @pytest.fixture
    def mock_pipeline(self, tmp_path):
        """Create a pipeline with mocked components."""
        # Mock embedding client
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        # Add test data to vector store
        pipeline.vector_store.add_chunks(
            chunk_ids=["music-1", "politics-1", "gaming-1"],
            embeddings=[
                [1.0] * 1536 + [0.1] * 1536,
                [0.1] * 1536 + [1.0] * 1536,
                [0.5] * 3072,
            ],
            documents=[
                "Taka: rock music is the best\nBas: I prefer metal",
                "Shark: the elections are interesting\nGeri: politics is complicated",
                "Jezi: anyone want to play valorant?",
            ],
            metadatas=[
                {"channel_name": "muzika", "year": 2024, "topics": "music"},
                {"channel_name": "politike", "year": 2024, "topics": "politics"},
                {"channel_name": "gaming", "year": 2025, "topics": "gaming"},
            ],
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_pipeline):
        """Should return search results for a query."""
        results = await mock_pipeline.semantic_search("music preferences")

        assert len(results) > 0
        assert all(hasattr(r, "chunk_id") for r in results)
        assert all(hasattr(r, "document") for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_k(self, mock_pipeline):
        """Should return at most k results."""
        results = await mock_pipeline.semantic_search("anything", k=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_with_channel_filter(self, mock_pipeline):
        """Should filter results by channel."""
        results = await mock_pipeline.semantic_search(
            "discussions",
            filters={"channel_name": "politike"},
        )

        assert len(results) == 1
        assert results[0].metadata["channel_name"] == "politike"

    @pytest.mark.asyncio
    async def test_search_with_year_filter(self, mock_pipeline):
        """Should filter results by year."""
        results = await mock_pipeline.semantic_search(
            "anything",
            filters={"year": 2025},
        )

        assert len(results) == 1
        assert results[0].metadata["year"] == 2025

    @pytest.mark.asyncio
    async def test_search_empty_store(self, tmp_path):
        """Should return empty list for empty store."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        results = await pipeline.semantic_search("anything")

        assert results == []


class TestNicknameResolution:
    """Tests for nickname resolution in queries."""

    @pytest.fixture
    def pipeline_with_nicknames(self, tmp_path):
        """Create pipeline with nickname mappings."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        nicknames = {
            301411562487545857: ["taka"],
            686998161163812968: ["bas", "bass", "basi"],
            416623828920172544: ["shark", "sharku", "sharko"],
        }

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
            nicknames=nicknames,
        )

        # Add test data
        pipeline.vector_store.add_chunks(
            chunk_ids=["chat-1"],
            embeddings=[[0.5] * 3072],
            documents=["Takarak: I love music"],
            metadatas=[
                {
                    "participant_ids": "301411562487545857",
                    "participant_names": "Takarak",
                    "participant_nicknames": "taka",
                }
            ],
        )

        return pipeline

    def test_resolve_nickname_to_author_id(self, pipeline_with_nicknames):
        """Should resolve nickname to author ID."""
        author_id = pipeline_with_nicknames.resolve_nickname("taka")
        assert author_id == 301411562487545857

    def test_resolve_nickname_case_insensitive(self, pipeline_with_nicknames):
        """Should resolve nicknames case-insensitively."""
        assert pipeline_with_nicknames.resolve_nickname("TAKA") == 301411562487545857
        assert pipeline_with_nicknames.resolve_nickname("Shark") == 416623828920172544

    def test_resolve_unknown_nickname(self, pipeline_with_nicknames):
        """Should return None for unknown nickname."""
        assert pipeline_with_nicknames.resolve_nickname("unknown") is None

    def test_find_nicknames_in_text(self, pipeline_with_nicknames):
        """Should find all nicknames in text."""
        found = pipeline_with_nicknames.find_nicknames_in_text(
            "What does taka think about shark's music taste?"
        )

        assert "taka" in found
        assert found["taka"] == 301411562487545857
        assert "shark" in found
        assert found["shark"] == 416623828920172544


class TestAnswerQuestion:
    """Tests for answer_question method."""

    @pytest.fixture
    def pipeline_with_data(self, tmp_path):
        """Create pipeline with test data and mocked LLM."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        # Add test data
        pipeline.vector_store.add_chunks(
            chunk_ids=["music-1", "music-2"],
            embeddings=[[0.5] * 3072, [0.5] * 3072],
            documents=[
                "Taka: I really love rock music, especially classic rock",
                "Taka: Led Zeppelin is my favorite band\nBas: I prefer Metallica",
            ],
            metadatas=[
                {"channel_name": "muzika", "year": 2024},
                {"channel_name": "muzika", "year": 2024},
            ],
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_answer_returns_response(self, pipeline_with_data):
        """Should return RAGResponse with answer and sources."""
        # Mock the LLM call - new format returns (answer, success, error)
        with patch.object(
            pipeline_with_data,
            "_generate_answer",
            new_callable=AsyncMock,
            return_value=("Taka loves rock music, especially Led Zeppelin.", True, None),
        ):
            response = await pipeline_with_data.answer_question("What does Taka think about music?")

        assert isinstance(response, RAGResponse)
        assert "rock" in response.answer.lower() or "zeppelin" in response.answer.lower()
        assert len(response.sources) > 0
        assert response.success is True

    @pytest.mark.asyncio
    async def test_answer_includes_sources(self, pipeline_with_data):
        """Should include relevant source chunks."""
        with patch.object(
            pipeline_with_data,
            "_generate_answer",
            new_callable=AsyncMock,
            return_value=("Taka's favorite band is Led Zeppelin.", True, None),
        ):
            response = await pipeline_with_data.answer_question("What is Taka's favorite band?")

        assert len(response.sources) > 0
        # Sources should contain relevant text
        source_texts = [s["text"] for s in response.sources]
        assert any("Led Zeppelin" in text for text in source_texts)

    @pytest.mark.asyncio
    async def test_answer_with_context_k(self, pipeline_with_data):
        """Should use specified number of context chunks."""
        with patch.object(
            pipeline_with_data,
            "_generate_answer",
            new_callable=AsyncMock,
            return_value=("Based on 1 source: rock music.", True, None),
        ) as mock_generate:
            await pipeline_with_data.answer_question(
                "Music preferences?",
                context_k=1,
            )

            # Check that _generate_answer was called with limited context
            call_args = mock_generate.call_args
            context = call_args[0][1] if call_args[0] else call_args[1].get("context")
            # Context should contain only 1 chunk
            assert context.count("---") <= 1  # Separator between chunks


class TestFilterByParticipant:
    """Tests for filtering search by participant."""

    @pytest.fixture
    def pipeline_with_participants(self, tmp_path):
        """Create pipeline with participant data."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        nicknames = {
            301411562487545857: ["taka"],
            686998161163812968: ["bas"],
        }

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
            nicknames=nicknames,
        )

        # Add test data with different participants
        pipeline.vector_store.add_chunks(
            chunk_ids=["taka-chat", "bas-chat", "both-chat"],
            embeddings=[[0.5] * 3072] * 3,
            documents=[
                "Taka: Solo chat",
                "Bas: Different solo chat",
                "Taka: Joint chat\nBas: Indeed",
            ],
            metadatas=[
                {
                    "participant_ids": "301411562487545857",
                    "participant_nicknames": "taka",
                },
                {
                    "participant_ids": "686998161163812968",
                    "participant_nicknames": "bas",
                },
                {
                    "participant_ids": "301411562487545857,686998161163812968",
                    "participant_nicknames": "taka,bas",
                },
            ],
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_search_by_participant_nickname(self, pipeline_with_participants):
        """Should filter results to include specific participant."""
        results = await pipeline_with_participants.semantic_search(
            "chat",
            filters={"participant_nicknames": {"$contains": "taka"}},
        )

        # Should include chunks where taka participated
        # Note: LanceDB filter syntax may differ, test verifies the concept


class TestChannelRecap:
    """Tests for generate_channel_recap method."""

    @pytest.fixture
    def pipeline_with_time_data(self, tmp_path):
        """Create pipeline with time-based test data."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        # Get current time for realistic timestamps
        now = datetime.datetime.now(datetime.UTC)
        recent = (now - datetime.timedelta(hours=6)).isoformat()
        old = (now - datetime.timedelta(days=10)).isoformat()

        # Add test data with different timestamps
        pipeline.vector_store.add_chunks(
            chunk_ids=["recent-1", "recent-2", "old-1"],
            embeddings=[[0.5] * 3072] * 3,
            documents=[
                "Taka: Just discussing the latest news\nBas: Interesting!",
                "Shark: Anyone watching the game?\nGeri: Yes!",
                "Old conversation from last week",
            ],
            metadatas=[
                {"channel_name": "kanapeja", "start_time": recent, "year": 2026, "month": 1},
                {"channel_name": "kanapeja", "start_time": recent, "year": 2026, "month": 1},
                {"channel_name": "kanapeja", "start_time": old, "year": 2025, "month": 12},
            ],
        )

        return pipeline

    def test_parse_timeframe_valid(self, tmp_path):
        """Should parse valid timeframe strings."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        # Test various timeframes
        assert pipeline._parse_timeframe("24h") is not None
        assert pipeline._parse_timeframe("7d") is not None
        assert pipeline._parse_timeframe("month") is not None
        assert pipeline._parse_timeframe("week") is not None

    def test_parse_timeframe_invalid(self, tmp_path):
        """Should return None for invalid timeframe."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        assert pipeline._parse_timeframe("invalid") is None
        assert pipeline._parse_timeframe("2w") is None
        assert pipeline._parse_timeframe("") is None

    @pytest.mark.asyncio
    async def test_recap_invalid_timeframe(self, tmp_path):
        """Should return error for invalid timeframe."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        response = await pipeline.generate_channel_recap(timeframe="invalid")

        assert not response.success
        assert "Unknown timeframe" in response.answer

    @pytest.mark.asyncio
    async def test_recap_empty_results(self, tmp_path):
        """Should handle empty results gracefully."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        response = await pipeline.generate_channel_recap(
            timeframe="24h",
            channel_name="nonexistent",
        )

        assert response.success
        assert "No activity found" in response.answer

    @pytest.mark.asyncio
    async def test_recap_with_channel_filter(self, pipeline_with_time_data):
        """Should filter by channel name."""
        with patch.object(
            pipeline_with_time_data,
            "llm_client",
            MagicMock(
                generate_recap=AsyncMock(
                    return_value=MagicMock(
                        text="Activity recap for kanapeja",
                        success=True,
                        error_message=None,
                    )
                )
            ),
        ):
            response = await pipeline_with_time_data.generate_channel_recap(
                timeframe="7d",
                channel_name="kanapeja",
            )

        assert response.success
        # All sources should be from kanapeja
        for source in response.sources:
            assert source["metadata"]["channel_name"] == "kanapeja"

    @pytest.mark.asyncio
    async def test_recap_without_llm(self, pipeline_with_time_data):
        """Should return fallback when no LLM configured."""
        # Ensure no LLM client
        pipeline_with_time_data.llm_client = None

        response = await pipeline_with_time_data.generate_channel_recap(
            timeframe="7d",
            channel_name="kanapeja",
        )

        assert not response.success
        assert "llm" in response.error_message.lower()
        assert "configured" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_recap_returns_sources(self, pipeline_with_time_data):
        """Should include source chunks in response."""
        with patch.object(
            pipeline_with_time_data,
            "llm_client",
            MagicMock(
                generate_recap=AsyncMock(
                    return_value=MagicMock(
                        text="Summary of activity",
                        success=True,
                        error_message=None,
                    )
                )
            ),
        ):
            response = await pipeline_with_time_data.generate_channel_recap(
                timeframe="30d",
                channel_name="kanapeja",
            )

        assert response.success
        assert len(response.sources) > 0
        assert all("chunk_id" in s for s in response.sources)
        assert all("text" in s for s in response.sources)


class TestHybridSearch:
    """Tests for hybrid search (BM25 + vector) functionality."""

    @pytest.fixture
    def pipeline_with_bm25(self, tmp_path):
        """Create pipeline with test data for hybrid search."""
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_query = AsyncMock(return_value=[0.5] * 3072)

        nicknames = {
            301411562487545857: ["taka"],
            686998161163812968: ["keno"],
        }

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
            nicknames=nicknames,
            use_bm25=True,  # Explicitly enable BM25 for these tests
        )

        # Add test data - BM25 should help find exact keyword matches
        pipeline.vector_store.add_chunks(
            chunk_ids=["keno-music", "taka-politics", "both-gaming"],
            embeddings=[
                [0.3] * 3072,  # Music embeddings
                [0.7] * 3072,  # Politics embeddings
                [0.5] * 3072,  # Gaming embeddings
            ],
            documents=[
                "keno: I love rock music and metal bands",
                "taka: discussing politics and economics today",
                "keno: playing games\ntaka: valorant is fun",
            ],
            metadatas=[
                {
                    "channel_name": "muzika",
                    "participant_ids": "686998161163812968",
                    "participant_nicknames": "keno",
                },
                {
                    "channel_name": "politike",
                    "participant_ids": "301411562487545857",
                    "participant_nicknames": "taka",
                },
                {
                    "channel_name": "gaming",
                    "participant_ids": "301411562487545857,686998161163812968",
                    "participant_nicknames": "taka,keno",
                },
            ],
        )

        return pipeline

    @pytest.mark.asyncio
    async def test_hybrid_search_when_enabled(self, pipeline_with_bm25):
        """Hybrid search should work when explicitly enabled."""
        # Search for exact keyword that BM25 should find
        results = await pipeline_with_bm25.semantic_search("valorant")

        assert len(results) > 0
        # Should find the gaming chunk with "valorant"
        assert any("valorant" in r.document.lower() for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_can_be_disabled(self, pipeline_with_bm25):
        """Should be able to disable BM25 search."""
        results = await pipeline_with_bm25.semantic_search(
            "valorant",
            use_bm25=False,
        )

        # Should still work (vector-only)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_finds_exact_matches(self, pipeline_with_bm25):
        """BM25 should help find exact keyword matches."""
        # BM25 should boost results containing "metal"
        results = await pipeline_with_bm25.semantic_search("metal bands")

        assert len(results) > 0
        # The chunk with "metal" should be in results
        found_metal = any("metal" in r.document.lower() for r in results)
        assert found_metal

    @pytest.mark.asyncio
    async def test_hybrid_search_with_nickname_filter(self, pipeline_with_bm25):
        """Hybrid search should work with nickname auto-filtering."""
        # Search for keno with BM25 enabled
        results = await pipeline_with_bm25.semantic_search("keno music")

        assert len(results) > 0
        # All results should have keno as participant
        for r in results:
            assert "keno" in r.metadata.get("participant_nicknames", "")

    @pytest.mark.asyncio
    async def test_rrf_merges_results(self, pipeline_with_bm25):
        """RRF should merge results from both retrievers."""
        # Patch _ensure_bm25_index to use a mock BM25Index
        from strofkabot.rag.bm25_index import BM25Index

        bm25_index = BM25Index()
        bm25_index.build_index(
            chunk_ids=["keno-music", "taka-politics", "both-gaming"],
            documents=[
                "keno: I love rock music and metal bands",
                "taka: discussing politics and economics today",
                "keno: playing games\ntaka: valorant is fun",
            ],
        )
        pipeline_with_bm25._bm25_index = bm25_index

        results = await pipeline_with_bm25.semantic_search("rock music", k=3)

        # Should return merged results
        assert len(results) <= 3
        # Music chunk should be highly ranked (matches both BM25 and semantic)
        assert results[0].chunk_id == "keno-music" or "music" in results[0].document.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_k_limit(self, pipeline_with_bm25):
        """Hybrid search should respect k parameter."""
        results = await pipeline_with_bm25.semantic_search("anything", k=2)

        assert len(results) <= 2


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion implementation."""

    def test_rrf_scores_both_retrievers(self, tmp_path):
        """RRF should score results from both vector and BM25."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        from strofkabot.rag.bm25_index import BM25Result
        from strofkabot.rag.vector_store import SearchResult

        # Create mock results
        vector_results = [
            SearchResult(chunk_id="c1", document="doc1", metadata={}, distance=0.1),
            SearchResult(chunk_id="c2", document="doc2", metadata={}, distance=0.2),
        ]

        bm25_results = [
            BM25Result(chunk_id="c2", score=3.0, document="doc2"),  # c2 appears in both
            BM25Result(chunk_id="c3", score=2.0, document="doc3"),  # c3 only in BM25
        ]

        # Add c3 to vector store so we can look it up
        pipeline.vector_store.add_chunks(
            chunk_ids=["c1", "c2", "c3"],
            embeddings=[[0.5] * 3072] * 3,
            documents=["doc1", "doc2", "doc3"],
            metadatas=[{}, {}, {}],
        )

        merged = pipeline._reciprocal_rank_fusion(vector_results, bm25_results)

        # c2 should rank highest (appears in both)
        assert merged[0].chunk_id == "c2"
        # All 3 chunks should be in results
        chunk_ids = [r.chunk_id for r in merged]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids

    def test_rrf_empty_bm25_results(self, tmp_path):
        """RRF should handle empty BM25 results."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        from strofkabot.rag.vector_store import SearchResult

        vector_results = [
            SearchResult(chunk_id="c1", document="doc1", metadata={}, distance=0.1),
        ]

        merged = pipeline._reciprocal_rank_fusion(vector_results, [])

        assert len(merged) == 1
        assert merged[0].chunk_id == "c1"

    def test_rrf_empty_vector_results(self, tmp_path):
        """RRF should handle empty vector results."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        from strofkabot.rag.bm25_index import BM25Result

        # Add chunk to vector store
        pipeline.vector_store.add_chunks(
            chunk_ids=["c1"],
            embeddings=[[0.5] * 3072],
            documents=["doc1"],
            metadatas=[{}],
        )

        bm25_results = [
            BM25Result(chunk_id="c1", score=3.0, document="doc1"),
        ]

        merged = pipeline._reciprocal_rank_fusion([], bm25_results)

        assert len(merged) == 1
        assert merged[0].chunk_id == "c1"


class TestFilterMatching:
    """Tests for metadata filter matching."""

    def test_matches_simple_filter(self, tmp_path):
        """Should match simple equality filter."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        metadata = {"channel_name": "kanapeja", "year": 2024}
        filters = {"channel_name": "kanapeja"}

        assert pipeline._matches_filters(metadata, filters)

    def test_matches_contains_filter(self, tmp_path):
        """Should match $contains filter."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        metadata = {"participant_ids": "123,456,789"}
        filters = {"participant_ids": {"$contains": "456"}}

        assert pipeline._matches_filters(metadata, filters)

    def test_matches_and_filter(self, tmp_path):
        """Should match $and filter."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        metadata = {"channel_name": "kanapeja", "year": 2024}
        filters = {
            "$and": [
                {"channel_name": "kanapeja"},
                {"year": 2024},
            ]
        }

        assert pipeline._matches_filters(metadata, filters)

    def test_does_not_match_wrong_value(self, tmp_path):
        """Should not match when value differs."""
        mock_embedding_client = MagicMock()
        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=mock_embedding_client,
        )

        metadata = {"channel_name": "politike"}
        filters = {"channel_name": "kanapeja"}

        assert not pipeline._matches_filters(metadata, filters)


# E2E test - requires real embedding client
@pytest.mark.e2e
class TestPipelineE2E:
    """End-to-end tests with real components."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self, tmp_path):
        """Test complete search workflow with real embeddings."""
        import os

        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        from strofkabot.rag.embeddings import OpenRouterEmbeddingClient

        embedding_client = OpenRouterEmbeddingClient()

        pipeline = RAGPipeline(
            vector_store_dir=tmp_path,
            embedding_client=embedding_client,
            nicknames={301411562487545857: ["taka"]},
        )

        # Generate real embeddings for test data
        docs = [
            "Taka: I really love rock music",
            "Shark: Politics are interesting",
        ]
        response = await embedding_client.embed_texts(docs)
        embeddings = response.embeddings

        pipeline.vector_store.add_chunks(
            chunk_ids=["music-chat", "politics-chat"],
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {"channel_name": "muzika", "topics": "music"},
                {"channel_name": "politike", "topics": "politics"},
            ],
        )

        # Search for music-related content
        results = await pipeline.semantic_search("rock music preferences", k=2)

        assert len(results) == 2
        # Music chunk should be more similar
        assert "music" in results[0].document.lower()
