"""OpenRouter embedding client for RAG.

Uses OpenRouter API with google/gemini-embedding-001 model
to generate 3072-dimensional embeddings for text.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """Response from embedding API call."""

    embeddings: list[list[float]] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None
    tokens_used: int = 0
    failed_indices: list[int] = field(default_factory=list)


class OpenRouterEmbeddingClient:
    """Client for generating embeddings via OpenRouter API.

    Uses google/gemini-embedding-001 which produces 3072-dimensional vectors.
    """

    model = "google/gemini-embedding-001"
    dimension = 3072

    def __init__(self, api_key: str | None = None):
        """Initialize the embedding client.

        Args:
            api_key: OpenRouter API key. If not provided, reads from
                     OPENROUTER_API_KEY environment variable.

        Raises:
            ValueError: If no API key provided or found in environment.
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set " "and no api_key was provided"
            )

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=60.0,
        )

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
        max_concurrent: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of texts with parallel processing.

        Args:
            texts: List of texts to embed.
            batch_size: Maximum texts per API call (default 100).
            max_concurrent: Maximum concurrent requests (default 50).
            max_retries: Number of retry attempts for failed batches (default 3).
            retry_delay: Base delay in seconds for exponential backoff (default 1.0).

        Returns:
            EmbeddingResponse with embeddings and metadata.
        """
        if not texts:
            return EmbeddingResponse(embeddings=[], success=True, tokens_used=0)

        # Split into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Rate limiter
        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_batch_with_retry(
            batch_idx: int, batch: list[str]
        ) -> tuple[int, list[list[float]] | None, str | None, int]:
            """Embed a single batch with retry logic.

            Returns:
                (batch_idx, embeddings or None, error or None, tokens_used)
            """
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        response = await self._client.embeddings.create(
                            model=self.model,
                            input=batch,
                        )
                        embeddings = [item.embedding for item in response.data]
                        return (batch_idx, embeddings, None, response.usage.total_tokens)

                    except Exception as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            # Exponential backoff: delay * 2^attempt
                            delay = retry_delay * (2**attempt)
                            logger.warning(
                                f"Batch {batch_idx} failed (attempt {attempt + 1}), "
                                f"retrying in {delay}s: {e}"
                            )
                            await asyncio.sleep(delay)

                # All retries exhausted
                logger.error(f"Batch {batch_idx} failed after {max_retries} attempts: {last_error}")
                return (batch_idx, None, last_error, 0)

        # Launch all batches in parallel (rate-limited by semaphore)
        tasks = [embed_batch_with_retry(i, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        # Reassemble in order
        all_embeddings: list[list[float]] = []
        failed_indices: list[int] = []
        errors: list[str] = []
        total_tokens = 0

        for batch_idx, embeddings, error, tokens in sorted(results, key=lambda x: x[0]):
            total_tokens += tokens
            if embeddings is not None:
                all_embeddings.extend(embeddings)
            else:
                # Mark failed batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                failed_indices.extend(range(start_idx, end_idx))
                errors.append(f"Batch {batch_idx}: {error}")

        success = len(failed_indices) == 0
        error_message = "; ".join(errors) if errors else None

        return EmbeddingResponse(
            embeddings=all_embeddings,
            success=success,
            error_message=error_message,
            tokens_used=total_tokens,
            failed_indices=failed_indices,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string.

        This is a convenience method for search queries.
        Unlike embed_texts, it raises on error rather than
        returning an error response.

        Args:
            query: The query text to embed.

        Returns:
            3072-dimensional embedding vector.

        Raises:
            Exception: If API call fails.
        """
        response = await self._client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding
