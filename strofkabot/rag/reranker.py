"""Reranking module for improving RAG search relevance.

Provides LLM-based reranking to improve the ordering of search results
beyond simple embedding similarity.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Lightweight model for reranking (fast and cheap)
DEFAULT_RERANK_MODEL = "google/gemini-2.0-flash-lite-001"


@dataclass
class RerankResult:
    """Result from reranking operation."""

    original_index: int
    score: float
    chunk_id: str


class LLMReranker:
    """LLM-based reranker for search results.

    Uses a lightweight LLM to score relevance of each result to the query,
    then reorders by score.

    Args:
        model: Model ID to use for reranking.
    """

    def __init__(self, model: str = DEFAULT_RERANK_MODEL):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
        )
        self.model = model

    async def rerank(
        self,
        query: str,
        results: list,
        top_k: int | None = None,
    ) -> list:
        """Rerank search results by relevance to query.

        Args:
            query: The search query.
            results: List of SearchResult objects to rerank.
            top_k: Keep only top k results after reranking (None = keep all).

        Returns:
            Reranked list of SearchResult objects.
        """
        if not results:
            return results

        if len(results) <= 1:
            return results

        # Score each result
        scores = await self._score_results(query, results)

        # Pair results with scores and sort
        scored_results = list(zip(results, scores, strict=True))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Extract reranked results
        reranked = [r for r, _ in scored_results]

        # Apply top_k limit if specified
        if top_k is not None and top_k < len(reranked):
            reranked = reranked[:top_k]

        return reranked

    async def _score_results(
        self,
        query: str,
        results: list,
    ) -> list[float]:
        """Score each result's relevance to the query.

        Args:
            query: The search query.
            results: List of SearchResult objects.

        Returns:
            List of relevance scores (0-1).
        """
        # Format results for scoring
        result_texts = []
        for i, r in enumerate(results):
            # Smart truncation: keep start and end for context
            doc = self._truncate_document(r.document, max_chars=1000)
            result_texts.append(f"[{i}] {doc}")

        results_str = "\n\n".join(result_texts)

        # Check token limit before calling LLM (rough estimate: ~8k token limit for context)
        estimated_tokens = self._estimate_tokens(results_str) + self._estimate_tokens(query) + 100
        max_context_tokens = 7000
        if estimated_tokens > max_context_tokens:
            logger.warning(
                f"Prompt too large ({estimated_tokens} est. tokens), falling back to similarity scores"
            )
            return [r.similarity for r in results]

        prompt = f"""Score the relevance of each document to the query on a scale of 0-10.

Query: {query}

Documents:
{results_str}

Return ONLY a JSON array of scores in order, like: [8, 5, 9, 3, 7]
Each score should be an integer from 0 (completely irrelevant) to 10 (highly relevant).
"""

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a relevance scoring assistant. Score documents by how well they answer the query. Output only the JSON array of scores.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                extra_body={
                    "HTTP-Referer": "https://github.com/strofkabot",
                    "X-Title": "StrofkaBot RAG Reranker",
                },
            )

            text = response.choices[0].message.content
            if text:
                # Parse JSON array from response
                scores = self._parse_scores(text, len(results))
                # Normalize to 0-1 range
                return [s / 10.0 for s in scores]

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using original order")

        # Fallback: use original similarity scores
        return [r.similarity for r in results]

    def _parse_scores(self, text: str, expected_count: int) -> list[float]:
        """Parse scores from LLM response.

        Args:
            text: LLM response text.
            expected_count: Expected number of scores.

        Returns:
            List of scores.
        """
        # Try to extract JSON array from response
        text = text.strip()

        # Handle markdown code blocks
        if "```" in text:
            # Extract content between code blocks
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            scores = json.loads(text)
            if isinstance(scores, list) and len(scores) == expected_count:
                return [float(s) for s in scores]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find numbers in the text
        numbers = re.findall(r"\d+\.?\d*", text)
        if len(numbers) >= expected_count:
            return [float(n) for n in numbers[:expected_count]]

        # Fallback: return uniform scores
        logger.warning(f"Could not parse scores from: {text[:100]}")
        return [5.0] * expected_count

    def _truncate_document(self, doc: str, max_chars: int = 1000) -> str:
        """Smart truncation preserving start and end context.

        Args:
            doc: Document text to truncate.
            max_chars: Maximum characters to keep.

        Returns:
            Truncated document with start and end preserved.
        """
        if len(doc) <= max_chars:
            return doc

        # Keep 60% from start, 40% from end
        start_chars = int(max_chars * 0.6)
        end_chars = max_chars - start_chars - 5  # Reserve space for " ... "

        return doc[:start_chars] + " ... " + doc[-end_chars:]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (approx 4 chars per token).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4


class SimpleReranker:
    """Simple keyword-based reranker (no API calls).

    Boosts results that contain query keywords in their text.
    Useful as a fallback when LLM reranking is not available.
    """

    # Stop words for filtering (English + Albanian)
    STOP_WORDS = {
        # English
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "it",
        "this",
        "that",
        "with",
        # Albanian
        "dhe",
        "eshte",
        "jane",
        "ne",
        "per",
        "me",
        "te",
        "se",
        "ka",
        "nga",
        "si",
        "por",
        "do",
        "nje",
        "kjo",
        "ajo",
        "ky",
        "ai",
    }

    async def rerank(
        self,
        query: str,
        results: list,
        top_k: int | None = None,
    ) -> list:
        """Rerank results based on keyword matching.

        Args:
            query: The search query.
            results: List of SearchResult objects.
            top_k: Keep only top k results.

        Returns:
            Reranked list of SearchResult objects.
        """
        if not results:
            return results

        # Extract query keywords with proper tokenization (handles punctuation)
        query_words = set(re.findall(r"\w+", query.lower()))
        # Remove stop words
        query_words -= self.STOP_WORDS

        if not query_words:
            return results[:top_k] if top_k else results

        # Pre-compile regex patterns for word boundary matching
        word_patterns = {w: re.compile(rf"\b{re.escape(w)}\b") for w in query_words}

        # Score each result by keyword overlap
        scored_results = []
        for r in results:
            doc_lower = r.document.lower()
            # Count keyword matches using word boundaries (not substring)
            matches = sum(1 for w, pattern in word_patterns.items() if pattern.search(doc_lower))
            # Combine with original similarity
            combined_score = r.similarity + (matches * 0.1)
            scored_results.append((r, combined_score))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Extract results
        reranked = [r for r, _ in scored_results]

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
