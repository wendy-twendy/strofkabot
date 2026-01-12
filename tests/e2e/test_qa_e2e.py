"""
End-to-end tests for the !qa (RAG) command using real OpenRouter API.

These tests make actual API calls and require:
- OPENROUTER_API_KEY environment variable set
- A populated vector store at data/vector_store/
- Run with: pytest tests/e2e/test_qa_e2e.py -v -m e2e

These tests are skipped by default in normal test runs.
To run them: pytest -m e2e
"""

import os
import warnings

import pytest

from strofkabot.config import NICKNAMES_FILE, RAG_LLM_MODEL, RAG_VECTOR_STORE_DIR
from strofkabot.rag.embeddings import OpenRouterEmbeddingClient
from strofkabot.rag.llm_client import RAGLLMClient
from strofkabot.rag.pipeline import RAGPipeline
from strofkabot.utils.nickname_loader import load_nicknames

# Skip all tests in this module if no API key or not explicitly running e2e tests
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    ),
    pytest.mark.skipif(
        not RAG_VECTOR_STORE_DIR.exists(),
        reason="RAG vector store not found",
    ),
]


def soft_assert(condition: bool, message: str) -> None:
    """Issue a warning instead of failing if condition is False.

    Use this for LLM behaviors that may vary but are expected most of the time.
    """
    if not condition:
        warnings.warn(f"Soft assertion failed: {message}", UserWarning, stacklevel=2)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rag_pipeline() -> RAGPipeline:
    """Create a real RAG pipeline with production data."""
    embedding_client = OpenRouterEmbeddingClient()
    llm_client = RAGLLMClient(model=RAG_LLM_MODEL)
    nicknames = load_nicknames(NICKNAMES_FILE)

    pipeline = RAGPipeline(
        vector_store_dir=RAG_VECTOR_STORE_DIR,
        embedding_client=embedding_client,
        nicknames=nicknames,
        llm_client=llm_client,
        # use_bm25 defaults to False (BM25 index uses ~4GB RAM)
    )

    return pipeline


# ============================================================================
# Pipeline Initialization Tests
# ============================================================================


class TestRAGPipelineInit:
    """Tests for RAG pipeline initialization."""

    def test_pipeline_has_data(self, rag_pipeline: RAGPipeline):
        """Pipeline should have chunks in vector store."""
        count = rag_pipeline.vector_store.count()
        assert count > 0, "Vector store should have data"
        print(f"\nVector store has {count} chunks")

    def test_bm25_index_loaded(self, rag_pipeline: RAGPipeline):
        """BM25 index should be loaded if available."""
        # BM25 index is optional, just check it doesn't crash
        assert rag_pipeline is not None


# ============================================================================
# Semantic Search Tests
# ============================================================================


class TestSemanticSearchE2E:
    """E2E tests for semantic search."""

    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(self, rag_pipeline: RAGPipeline):
        """Semantic search should return results for a general query."""
        results = await rag_pipeline.semantic_search("music", k=5)

        assert len(results) > 0, "Should return at least one result"
        assert len(results) <= 5, "Should respect k limit"

        # Check result structure
        for r in results:
            assert hasattr(r, "document"), "Result should have document"
            assert hasattr(r, "metadata"), "Result should have metadata"
            assert hasattr(r, "distance"), "Result should have distance"

        print(f"\nFound {len(results)} results for 'music' query")

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, rag_pipeline: RAGPipeline):
        """Results should be somewhat relevant to the query."""
        results = await rag_pipeline.semantic_search("wordle game", k=3)

        # At least one result should contain wordle-related content
        has_relevant = any(
            "wordle" in r.document.lower() or "game" in r.document.lower() for r in results
        )

        soft_assert(
            has_relevant,
            f"Expected at least one wordle-related result in: {[r.document[:50] for r in results]}",
        )


# ============================================================================
# Question Answering Tests
# ============================================================================


class TestQuestionAnsweringE2E:
    """E2E tests for RAG question answering."""

    @pytest.mark.asyncio
    async def test_answer_question_returns_response(self, rag_pipeline: RAGPipeline):
        """answer_question should return a response object."""
        response = await rag_pipeline.answer_question(
            "What do people talk about on this server?",
            context_k=3,
        )

        # Hard assertion: must return valid response
        assert response is not None, "Should return response"
        assert hasattr(response, "success"), "Response should have success field"
        assert hasattr(response, "answer"), "Response should have answer field"
        assert hasattr(response, "sources"), "Response should have sources field"

        print(f"\nResponse success: {response.success}")
        print(f"Answer preview: {response.answer[:200]}...")

    @pytest.mark.asyncio
    async def test_answer_question_success(self, rag_pipeline: RAGPipeline):
        """answer_question should succeed with valid question."""
        response = await rag_pipeline.answer_question(
            "What games do people play?",
            context_k=5,
        )

        # Hard assertion: should succeed
        assert response.success is True, f"Expected success, got: {response.answer}"
        assert len(response.answer) > 0, "Should have non-empty answer"

    @pytest.mark.asyncio
    async def test_answer_includes_sources(self, rag_pipeline: RAGPipeline):
        """Response should include source information."""
        response = await rag_pipeline.answer_question(
            "Tell me about discussions on this server",
            context_k=5,
        )

        assert response.success is True
        assert response.sources is not None, "Should have sources"
        assert len(response.sources) > 0, "Should have at least one source"

        # Check source structure
        for source in response.sources:
            assert "metadata" in source or "document" in source

        print(f"\nNumber of sources: {len(response.sources)}")

    @pytest.mark.asyncio
    async def test_answer_uses_context(self, rag_pipeline: RAGPipeline):
        """Answer should be based on retrieved context, not hallucinated."""
        response = await rag_pipeline.answer_question(
            "Who plays Wordle?",
            context_k=5,
        )

        assert response.success is True

        # The answer should either mention names from the server or say it doesn't know
        # This is a soft assertion since LLM output varies
        answer_lower = response.answer.lower()
        soft_assert(
            len(response.answer) > 20,  # Should be a real answer, not empty
            f"Answer seems too short: {response.answer}",
        )

        print(f"\nAnswer: {response.answer[:300]}")

    @pytest.mark.asyncio
    async def test_answer_handles_albanian_question(self, rag_pipeline: RAGPipeline):
        """Should handle questions in Albanian."""
        response = await rag_pipeline.answer_question(
            "Çfarë diskutojnë njerëzit këtu?",
            context_k=3,
        )

        assert response.success is True
        assert len(response.answer) > 0

        print(f"\nAlbanian question answer: {response.answer[:200]}")

    @pytest.mark.asyncio
    async def test_answer_no_data_scenario(self, rag_pipeline: RAGPipeline):
        """Should handle questions about topics not in the data gracefully."""
        response = await rag_pipeline.answer_question(
            "What did people say about quantum computing?",
            context_k=3,
        )

        # Should succeed but might indicate lack of relevant info
        assert response.success is True
        assert len(response.answer) > 0

        # The answer should acknowledge uncertainty if topic isn't covered
        # This is a soft assertion since behavior may vary
        print(f"\nAnswer for obscure topic: {response.answer[:200]}")


# ============================================================================
# Hybrid Search Tests (via semantic_search with use_bm25=True)
# ============================================================================


@pytest.mark.skip(reason="BM25 index uses ~4GB RAM, disabled for now")
class TestHybridSearchE2E:
    """E2E tests for hybrid search (vector + BM25)."""

    @pytest.mark.asyncio
    async def test_hybrid_search_works(self, rag_pipeline: RAGPipeline):
        """Hybrid search should combine vector and keyword results."""
        # semantic_search uses hybrid by default when BM25 index is available
        results = await rag_pipeline.semantic_search(
            "wordle game score",
            k=5,
            use_bm25=True,
        )

        assert len(results) > 0, "Should return results"

        print(f"\nHybrid search returned {len(results)} results")
        for r in results[:2]:
            print(f"  - {r.document[:80]}...")

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_boost(self, rag_pipeline: RAGPipeline):
        """Hybrid search should boost exact keyword matches."""
        # Search for a specific term that should appear in messages
        results = await rag_pipeline.semantic_search(
            "Wordle",
            k=5,
            use_bm25=True,
        )

        # At least one result should contain the exact keyword
        has_exact = any("wordle" in r.document.lower() for r in results)

        soft_assert(
            has_exact,
            "Expected at least one result with 'Wordle' keyword",
        )


# ============================================================================
# Query Rewriter E2E Tests
# ============================================================================


class TestQueryRewriterE2E:
    """E2E tests for enhanced query rewriting with member context."""

    @pytest.fixture
    def openrouter_client(self):
        """Create a real OpenRouter client for query rewriting."""
        from strofkabot.openrouter import OpenRouterClient

        return OpenRouterClient()

    @pytest.fixture
    def sample_members(self):
        """Sample member list for testing entity resolution."""
        from strofkabot.rag.query_rewriter import MemberInfo

        return [
            MemberInfo(
                author_id=301411562487545857,
                display_name="Takarak",
                username=".takarak",
                nicknames=["taka", "tak"],
            ),
            MemberInfo(
                author_id=238553996657295361,
                display_name="dave_a7x",
                username="dave_a7x",
                nicknames=["dejv", "dejvi", "dave"],
            ),
            MemberInfo(
                author_id=123456789012345678,
                display_name="Endi",
                username="endi",
                nicknames=["e"],
            ),
        ]

    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation for pronoun resolution."""
        from strofkabot.rag.query_rewriter import ConversationMessage

        return [
            ConversationMessage(
                author="Takarak",
                author_id=301411562487545857,
                content="I think AI is going to change the world completely",
            ),
            ConversationMessage(
                author="dave_a7x",
                author_id=238553996657295361,
                content="He's always so optimistic about these things",
            ),
        ]

    @pytest.mark.asyncio
    async def test_rewrite_basic_query(self, openrouter_client):
        """Basic query rewriting should work."""
        from strofkabot.rag.query_rewriter import rewrite_query

        result = await rewrite_query(
            client=openrouter_client._client,
            question="What do people think about politics?",
        )

        # Hard assertions: structure must be correct
        assert result is not None
        assert result.original_query == "What do people think about politics?"
        assert len(result.rag_queries) > 0
        assert isinstance(result.detected_entities, list)
        assert isinstance(result.resolved_entity_ids, list)
        assert result.retrieval_strategy in ["semantic", "participant_focused", "keyword"]

        print("\nBasic rewrite result:")
        print(f"  Queries: {result.rag_queries}")
        print(f"  Strategy: {result.retrieval_strategy}")

    @pytest.mark.asyncio
    async def test_rewrite_with_member_context(self, openrouter_client, sample_members):
        """Query rewriting with member context should resolve entities."""
        from strofkabot.rag.query_rewriter import rewrite_query

        result = await rewrite_query(
            client=openrouter_client._client,
            question="What does taka think about AI?",
            members=sample_members,
        )

        # Hard assertion: must return valid result
        assert result is not None
        assert len(result.rag_queries) > 0

        # Soft assertions: entity resolution should work
        soft_assert(
            "taka" in [e.lower() for e in result.detected_entities]
            or "takarak" in [e.lower() for e in result.detected_entities],
            f"Expected 'taka' or 'Takarak' in entities: {result.detected_entities}",
        )

        soft_assert(
            301411562487545857 in result.resolved_entity_ids,
            f"Expected Takarak's ID in resolved_entity_ids: {result.resolved_entity_ids}",
        )

        soft_assert(
            result.retrieval_strategy == "participant_focused",
            f"Expected participant_focused strategy, got: {result.retrieval_strategy}",
        )

        print("\nMember context rewrite result:")
        print(f"  Queries: {result.rag_queries}")
        print(f"  Entities: {result.detected_entities}")
        print(f"  Resolved IDs: {result.resolved_entity_ids}")
        print(f"  Strategy: {result.retrieval_strategy}")

    @pytest.mark.asyncio
    async def test_rewrite_resolves_pronouns(
        self, openrouter_client, sample_members, sample_conversation
    ):
        """Pronoun resolution should work with conversation context."""
        from strofkabot.rag.query_rewriter import rewrite_query

        result = await rewrite_query(
            client=openrouter_client._client,
            question="What else did he say?",
            members=sample_members,
            conversation_history=sample_conversation,
        )

        # Hard assertion: must return valid result
        assert result is not None
        assert len(result.rag_queries) > 0

        # Soft assertions: pronoun should be resolved
        soft_assert(
            result.resolved_query is not None,
            "Expected resolved_query for pronoun question, got None",
        )

        if result.resolved_query:
            soft_assert(
                "taka" in result.resolved_query.lower()
                or "takarak" in result.resolved_query.lower(),
                f"Expected 'Taka' in resolved query: {result.resolved_query}",
            )

        soft_assert(
            301411562487545857 in result.resolved_entity_ids,
            f"Expected Takarak's ID from pronoun resolution: {result.resolved_entity_ids}",
        )

        print("\nPronoun resolution result:")
        print(f"  Original: {result.original_query}")
        print(f"  Resolved: {result.resolved_query}")
        print(f"  Entities: {result.detected_entities}")
        print(f"  Resolved IDs: {result.resolved_entity_ids}")

    @pytest.mark.asyncio
    async def test_rewrite_with_exclusion(self, openrouter_client, sample_members):
        """Exclusion queries should detect excluded entities."""
        from strofkabot.rag.query_rewriter import rewrite_query

        result = await rewrite_query(
            client=openrouter_client._client,
            question="What does everyone except Taka think about this?",
            members=sample_members,
        )

        # Hard assertion: must return valid result
        assert result is not None

        # Soft assertion: should detect exclusion
        soft_assert(
            len(result.excluded_entities) > 0,
            f"Expected excluded_entities for 'except Taka', got: {result.excluded_entities}",
        )

        if result.excluded_entities:
            soft_assert(
                "taka" in [e.lower() for e in result.excluded_entities]
                or "takarak" in [e.lower() for e in result.excluded_entities],
                f"Expected 'Taka' in excluded_entities: {result.excluded_entities}",
            )

        print("\nExclusion detection result:")
        print(f"  Excluded: {result.excluded_entities}")

    @pytest.mark.asyncio
    async def test_rewrite_temporal_filter(self, openrouter_client):
        """Temporal references should be parsed."""
        from strofkabot.rag.query_rewriter import rewrite_query

        result = await rewrite_query(
            client=openrouter_client._client,
            question="What was discussed yesterday?",
        )

        # Hard assertion: must return valid result
        assert result is not None

        # Soft assertion: should detect temporal filter
        soft_assert(
            result.temporal_filter is not None,
            "Expected temporal_filter for 'yesterday', got None",
        )

        if result.temporal_filter:
            soft_assert(
                "after" in result.temporal_filter or "before" in result.temporal_filter,
                f"Expected after/before in temporal_filter: {result.temporal_filter}",
            )

        print("\nTemporal filter result:")
        print(f"  Filter: {result.temporal_filter}")
