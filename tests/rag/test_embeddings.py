# Tests for OpenRouterEmbeddingClient (TDD - write tests first)

"""Tests for embedding client functionality.

These tests verify the OpenRouter embedding client:
- Single text embedding returns 3072-dim vector
- Batch embedding handles multiple texts
- Empty list returns empty embeddings
- Albanian/special characters handled correctly
- API errors return success=False response
- Missing API key raises ValueError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.rag.embeddings import EmbeddingResponse, OpenRouterEmbeddingClient


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse dataclass."""

    def test_creation_success(self):
        """EmbeddingResponse should be creatable with success=True."""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            success=True,
            error_message=None,
            tokens_used=10,
        )

        assert response.success is True
        assert response.error_message is None
        assert len(response.embeddings) == 1
        assert response.tokens_used == 10

    def test_creation_failure(self):
        """EmbeddingResponse should be creatable with success=False."""
        response = EmbeddingResponse(
            embeddings=[],
            success=False,
            error_message="API error",
            tokens_used=0,
        )

        assert response.success is False
        assert response.error_message == "API error"
        assert len(response.embeddings) == 0


class TestOpenRouterEmbeddingClientInit:
    """Tests for client initialization."""

    def test_missing_api_key_raises(self):
        """Should raise ValueError if no API key provided or in env."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterEmbeddingClient()

    def test_init_with_explicit_key(self):
        """Should accept explicit API key."""
        with patch("strofkabot.rag.embeddings.AsyncOpenAI"):
            client = OpenRouterEmbeddingClient(api_key="test-key")
            assert client is not None

    def test_init_from_env(self):
        """Should read API key from environment."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
            with patch("strofkabot.rag.embeddings.AsyncOpenAI"):
                client = OpenRouterEmbeddingClient()
                assert client is not None


class TestEmbedTexts:
    """Tests for embed_texts method."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked AsyncOpenAI."""
        with patch("strofkabot.rag.embeddings.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            client = OpenRouterEmbeddingClient(api_key="test-key")
            yield client, mock_instance

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_client):
        """Should return 3072-dim vector for single text."""
        client, mock_openai = mock_client

        # Mock the embeddings.create response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 3072
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 5

        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        response = await client.embed_texts(["Hello world"])

        assert response.success is True
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3072
        assert response.tokens_used == 5

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_client):
        """Should handle batch of 100 texts."""
        client, mock_openai = mock_client

        # Mock response for 100 texts
        mock_embeddings = []
        for i in range(100):
            mock_emb = MagicMock()
            mock_emb.embedding = [float(i) / 100] * 3072
            mock_embeddings.append(mock_emb)

        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_response.usage.total_tokens = 500

        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        texts = [f"Text {i}" for i in range(100)]
        response = await client.embed_texts(texts)

        assert response.success is True
        assert len(response.embeddings) == 100
        for emb in response.embeddings:
            assert len(emb) == 3072

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, mock_client):
        """Should return empty embeddings for empty input."""
        client, mock_openai = mock_client

        response = await client.embed_texts([])

        assert response.success is True
        assert len(response.embeddings) == 0
        # Should not call API for empty list
        mock_openai.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_with_albanian_text(self, mock_client):
        """Should handle Albanian text with special characters."""
        client, mock_openai = mock_client

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.5] * 3072
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 15

        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        # Albanian text with special chars
        albanian_text = "Ç'kemi? Si jeni sot? Shqipëria është e bukur."
        response = await client.embed_texts([albanian_text])

        assert response.success is True
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3072

    @pytest.mark.asyncio
    async def test_error_response_on_api_failure(self, mock_client):
        """Should return success=False on API error."""
        client, mock_openai = mock_client

        mock_openai.embeddings.create = AsyncMock(side_effect=Exception("API rate limit exceeded"))

        response = await client.embed_texts(["Some text"])

        assert response.success is False
        assert "rate limit" in response.error_message.lower()
        assert len(response.embeddings) == 0

    @pytest.mark.asyncio
    async def test_batching_splits_large_requests(self, mock_client):
        """Should split requests larger than batch_size."""
        client, mock_openai = mock_client

        # Create mock responses for two batches
        def create_mock_response(num_texts):
            embeddings = []
            for _ in range(num_texts):
                mock_emb = MagicMock()
                mock_emb.embedding = [0.1] * 3072
                embeddings.append(mock_emb)
            mock_resp = MagicMock()
            mock_resp.data = embeddings
            mock_resp.usage.total_tokens = num_texts * 5
            return mock_resp

        # Will be called twice: once with 100, once with 50
        mock_openai.embeddings.create = AsyncMock(
            side_effect=[create_mock_response(100), create_mock_response(50)]
        )

        texts = [f"Text {i}" for i in range(150)]
        response = await client.embed_texts(texts, batch_size=100)

        assert response.success is True
        assert len(response.embeddings) == 150
        assert mock_openai.embeddings.create.call_count == 2


class TestEmbedQuery:
    """Tests for embed_query convenience method."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked AsyncOpenAI."""
        with patch("strofkabot.rag.embeddings.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            client = OpenRouterEmbeddingClient(api_key="test-key")
            yield client, mock_instance

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self, mock_client):
        """embed_query should return a single 3072-dim vector."""
        client, mock_openai = mock_client

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.25] * 3072
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 3

        mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

        vector = await client.embed_query("What does taka think about music?")

        assert len(vector) == 3072
        assert all(v == 0.25 for v in vector)

    @pytest.mark.asyncio
    async def test_embed_query_raises_on_error(self, mock_client):
        """embed_query should raise exception on API error."""
        client, mock_openai = mock_client

        mock_openai.embeddings.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await client.embed_query("Some query")


class TestParallelEmbedding:
    """Tests for parallel embedding with retry logic."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked AsyncOpenAI."""
        with patch("strofkabot.rag.embeddings.AsyncOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            client = OpenRouterEmbeddingClient(api_key="test-key")
            yield client, mock_instance

    @pytest.mark.asyncio
    async def test_parallel_execution_uses_semaphore(self, mock_client):
        """Parallel execution should respect max_concurrent limit."""
        client, mock_openai = mock_client

        # Track concurrent calls
        import asyncio

        concurrent_count = 0
        max_concurrent_seen = 0

        async def slow_embed(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate API latency
            concurrent_count -= 1

            mock_emb = MagicMock()
            mock_emb.embedding = [0.1] * 3072
            mock_resp = MagicMock()
            mock_resp.data = [mock_emb] * len(args[0] if args else kwargs.get("input", []))
            mock_resp.usage.total_tokens = 10
            return mock_resp

        mock_openai.embeddings.create = slow_embed

        # 500 texts = 5 batches of 100
        texts = [f"Text {i}" for i in range(500)]
        response = await client.embed_texts(texts, batch_size=100, max_concurrent=2)

        assert response.success is True
        assert len(response.embeddings) == 500
        # Should never exceed max_concurrent
        assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mock_client):
        """Should retry on transient errors with exponential backoff."""
        client, mock_openai = mock_client

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            # Third call succeeds
            mock_emb = MagicMock()
            mock_emb.embedding = [0.1] * 3072
            mock_resp = MagicMock()
            mock_resp.data = [mock_emb]
            mock_resp.usage.total_tokens = 5
            return mock_resp

        mock_openai.embeddings.create = failing_then_success

        response = await client.embed_texts(
            ["Test text"],
            batch_size=100,
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )

        assert response.success is True
        assert len(response.embeddings) == 1
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_embeddings(self, mock_client):
        """Partial failure should return successful embeddings with failed indices."""
        client, mock_openai = mock_client

        batch_call_count = 0

        async def partial_failure(*args, **kwargs):
            nonlocal batch_call_count
            batch_call_count += 1
            if batch_call_count == 2:
                raise Exception("Batch 2 failed")
            # Other batches succeed
            input_texts = kwargs.get("input", args[0] if args else [])
            mock_embs = []
            for _ in input_texts:
                mock_emb = MagicMock()
                mock_emb.embedding = [0.1] * 3072
                mock_embs.append(mock_emb)
            mock_resp = MagicMock()
            mock_resp.data = mock_embs
            mock_resp.usage.total_tokens = len(input_texts) * 5
            return mock_resp

        mock_openai.embeddings.create = partial_failure

        # 300 texts = 3 batches, batch 2 will fail
        texts = [f"Text {i}" for i in range(300)]
        response = await client.embed_texts(
            texts,
            batch_size=100,
            max_concurrent=10,
            max_retries=1,  # No retries to force failure
            retry_delay=0.01,
        )

        assert response.success is False
        assert len(response.embeddings) == 200  # Batches 1 and 3 succeeded
        assert len(response.failed_indices) == 100  # Batch 2 failed
        assert 100 in response.failed_indices  # First index of batch 2
        assert 199 in response.failed_indices  # Last index of batch 2

    @pytest.mark.asyncio
    async def test_order_preserved_in_parallel(self, mock_client):
        """Results should be in original order despite parallel execution."""
        client, mock_openai = mock_client

        import asyncio
        import random

        async def random_delay_embed(*args, **kwargs):
            # Random delay to shuffle completion order
            await asyncio.sleep(random.uniform(0.001, 0.01))
            input_texts = kwargs.get("input", args[0] if args else [])
            mock_embs = []
            for text in input_texts:
                mock_emb = MagicMock()
                # Encode text index in embedding for verification
                idx = int(text.split()[1])
                mock_emb.embedding = [float(idx)] * 3072
                mock_embs.append(mock_emb)
            mock_resp = MagicMock()
            mock_resp.data = mock_embs
            mock_resp.usage.total_tokens = 10
            return mock_resp

        mock_openai.embeddings.create = random_delay_embed

        texts = [f"Text {i}" for i in range(500)]
        response = await client.embed_texts(texts, batch_size=100, max_concurrent=50)

        assert response.success is True
        assert len(response.embeddings) == 500
        # Verify order: embedding[i] should have value i in all positions
        for i, emb in enumerate(response.embeddings):
            assert emb[0] == float(i), f"Order mismatch at index {i}"

    @pytest.mark.asyncio
    async def test_failed_indices_tracked_after_all_retries_exhausted(self, mock_client):
        """Should track failed indices when all retries exhausted."""
        client, mock_openai = mock_client

        async def always_fail(*args, **kwargs):
            raise Exception("Permanent failure")

        mock_openai.embeddings.create = always_fail

        response = await client.embed_texts(
            ["Text 1", "Text 2", "Text 3"],
            batch_size=100,
            max_retries=3,
            retry_delay=0.01,
        )

        assert response.success is False
        assert len(response.embeddings) == 0
        assert response.failed_indices == [0, 1, 2]
        assert "Permanent failure" in response.error_message

    @pytest.mark.asyncio
    async def test_new_parameters_have_defaults(self, mock_client):
        """New parallel parameters should have sensible defaults."""
        client, mock_openai = mock_client

        mock_emb = MagicMock()
        mock_emb.embedding = [0.1] * 3072
        mock_resp = MagicMock()
        mock_resp.data = [mock_emb]
        mock_resp.usage.total_tokens = 5
        mock_openai.embeddings.create = AsyncMock(return_value=mock_resp)

        # Should work without specifying new parameters
        response = await client.embed_texts(["Test"])

        assert response.success is True
        assert len(response.embeddings) == 1


# E2E test - requires real API key
@pytest.mark.e2e
class TestEmbeddingsE2E:
    """End-to-end tests with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_embed_real_text(self):
        """Should get real embedding from OpenRouter API."""
        import os

        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterEmbeddingClient()
        response = await client.embed_texts(["Hello, this is a test message."])

        assert response.success is True
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3072
        assert response.tokens_used > 0

        # Embeddings should be normalized floats
        emb = response.embeddings[0]
        assert all(isinstance(v, float) for v in emb)
        assert all(-10 < v < 10 for v in emb)  # Reasonable range

    @pytest.mark.asyncio
    async def test_embed_query_real(self):
        """Should get real embedding for a query."""
        import os

        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        client = OpenRouterEmbeddingClient()
        vector = await client.embed_query("What does taka think about music?")

        assert len(vector) == 3072
        assert all(isinstance(v, float) for v in vector)
