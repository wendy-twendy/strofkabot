"""BM25 keyword search index for hybrid retrieval.

This module provides a BM25 index that complements vector search
for hybrid retrieval using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    """Result from BM25 search."""

    chunk_id: str
    score: float
    document: str


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Simple tokenization that works well for Albanian and English:
    - Lowercase
    - Split on word boundaries
    - Keep tokens with 2+ characters

    Args:
        text: Text to tokenize.

    Returns:
        List of tokens.
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """BM25 keyword search index with persistence support.

    Provides keyword-based search to complement vector search.
    Uses BM25Okapi algorithm from rank-bm25 library.

    Example:
        >>> index = BM25Index()
        >>> index.build_index(["id1", "id2"], ["hello world", "goodbye world"])
        >>> results = index.search("hello", k=1)
        >>> results[0].chunk_id
        'id1'
    """

    def __init__(self) -> None:
        """Initialize empty BM25 index."""
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._documents: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, chunk_ids: list[str], documents: list[str]) -> None:
        """Build BM25 index from documents.

        Args:
            chunk_ids: Unique identifiers for each document.
            documents: Document texts to index.

        Raises:
            ValueError: If chunk_ids and documents have different lengths.
        """
        if len(chunk_ids) != len(documents):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and documents ({len(documents)}) "
                "must have the same length"
            )

        self._chunk_ids = list(chunk_ids)
        self._documents = list(documents)
        self._tokenized_corpus = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        """Search the index for documents matching the query.

        Args:
            query: Search query string.
            k: Maximum number of results to return.

        Returns:
            List of BM25Result objects sorted by score (descending).

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not query.strip():
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Get top k results with non-zero scores
        scored_docs = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = scored_docs[:k]

        return [
            BM25Result(
                chunk_id=self._chunk_ids[i],
                score=score,
                document=self._documents[i],
            )
            for i, score in top_k
        ]

    def add_documents(self, chunk_ids: list[str], documents: list[str]) -> None:
        """Add new documents to the index.

        Rebuilds the entire index with existing + new documents.
        For large incremental updates, consider batching.

        Args:
            chunk_ids: Unique identifiers for new documents.
            documents: New document texts to add.

        Raises:
            ValueError: If chunk_ids and documents have different lengths.
        """
        if len(chunk_ids) != len(documents):
            raise ValueError(
                f"chunk_ids ({len(chunk_ids)}) and documents ({len(documents)}) "
                "must have the same length"
            )

        # Add to existing data
        self._chunk_ids.extend(chunk_ids)
        self._documents.extend(documents)
        new_tokenized = [tokenize(doc) for doc in documents]
        self._tokenized_corpus.extend(new_tokenized)

        # Rebuild BM25 index
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def save(self, path: Path) -> None:
        """Save the index to disk.

        Args:
            path: File path to save the index.
        """
        data = {
            "chunk_ids": self._chunk_ids,
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load an index from disk.

        Args:
            path: File path to load the index from.

        Returns:
            BM25Index instance with loaded data.

        Raises:
            FileNotFoundError: If the path doesn't exist.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls()
        index._chunk_ids = data["chunk_ids"]
        index._documents = data["documents"]
        index._tokenized_corpus = data["tokenized_corpus"]
        index._bm25 = BM25Okapi(index._tokenized_corpus)
        return index

    def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._chunk_ids)

    def get_chunk_ids(self) -> list[str]:
        """Return all chunk IDs in the index."""
        return list(self._chunk_ids)
