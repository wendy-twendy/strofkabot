"""Tests for BM25Index keyword search functionality.

These tests verify the BM25 index:
- Building index from documents
- Keyword search and scoring
- Tokenization for Albanian and English
- Persistence (save/load)
- Empty queries and edge cases
- Incremental document addition
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from strofkabot.rag.bm25_index import BM25Index, BM25Result, tokenize


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self):
        """Should split text into lowercase tokens."""
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_removes_short_tokens(self):
        """Should remove tokens with 1 character."""
        tokens = tokenize("I am a test")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "test" in tokens

    def test_handles_punctuation(self):
        """Should handle punctuation correctly."""
        tokens = tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens

    def test_albanian_text(self):
        """Should handle Albanian characters."""
        tokens = tokenize("Mirëdita, si jeni sot?")
        assert "mirëdita" in tokens
        assert "si" in tokens
        assert "jeni" in tokens
        assert "sot" in tokens

    def test_empty_string(self):
        """Should return empty list for empty string."""
        tokens = tokenize("")
        assert tokens == []

    def test_preserves_numbers(self):
        """Should keep numeric tokens."""
        tokens = tokenize("Test 123 value")
        assert "test" in tokens
        assert "123" in tokens
        assert "value" in tokens


class TestBM25Result:
    """Tests for BM25Result dataclass."""

    def test_creation(self):
        """Should create BM25Result with all fields."""
        result = BM25Result(
            chunk_id="chunk-123",
            score=2.5,
            document="Some text content",
        )
        assert result.chunk_id == "chunk-123"
        assert result.score == 2.5
        assert result.document == "Some text content"


class TestBM25IndexInit:
    """Tests for BM25Index initialization."""

    def test_empty_index(self):
        """Should start with empty index."""
        index = BM25Index()
        assert index.count() == 0
        assert index.get_chunk_ids() == []

    def test_search_without_build_raises(self):
        """Should raise error if searching before building."""
        index = BM25Index()
        with pytest.raises(RuntimeError, match="Index not built"):
            index.search("test query")


class TestBM25IndexBuild:
    """Tests for building BM25 index."""

    def test_build_index(self):
        """Should build index from documents."""
        index = BM25Index()
        index.build_index(
            chunk_ids=["c1", "c2", "c3"],
            documents=["hello world", "goodbye world", "hello there"],
        )
        assert index.count() == 3
        assert set(index.get_chunk_ids()) == {"c1", "c2", "c3"}

    def test_mismatched_lengths_raises(self):
        """Should raise error if chunk_ids and documents have different lengths."""
        index = BM25Index()
        with pytest.raises(ValueError, match="same length"):
            index.build_index(
                chunk_ids=["c1", "c2"],
                documents=["only one document"],
            )


class TestBM25IndexSearch:
    """Tests for BM25 search functionality."""

    @pytest.fixture
    def sample_index(self):
        """Create a sample index for testing."""
        index = BM25Index()
        index.build_index(
            chunk_ids=["c1", "c2", "c3", "c4"],
            documents=[
                "keno is talking about music and rock bands",
                "taka loves politics and debates about economics",
                "keno and taka discussing football together",
                "random conversation about weather today",
            ],
        )
        return index

    def test_basic_search(self, sample_index):
        """Should find documents matching query."""
        results = sample_index.search("music", k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "c1"  # Contains "music"

    def test_search_returns_sorted_by_score(self, sample_index):
        """Results should be sorted by score descending."""
        # Search for common word that appears in multiple docs
        results = sample_index.search("talking about", k=10)
        if len(results) >= 2:
            # Check scores are descending
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_search_limits_results(self, sample_index):
        """Should respect k parameter."""
        results = sample_index.search("the", k=2)
        assert len(results) <= 2

    def test_search_empty_query(self, sample_index):
        """Should return empty list for empty query."""
        results = sample_index.search("")
        assert results == []

    def test_search_whitespace_query(self, sample_index):
        """Should return empty list for whitespace-only query."""
        results = sample_index.search("   ")
        assert results == []

    def test_search_no_matches(self, sample_index):
        """Should return empty list when no documents match."""
        results = sample_index.search("xyznonexistent")
        assert results == []

    def test_search_multiple_terms(self, sample_index):
        """Should handle multi-word queries."""
        results = sample_index.search("keno music", k=5)
        assert len(results) > 0
        # Document with both terms should rank higher
        assert results[0].chunk_id == "c1"

    def test_search_case_insensitive(self, sample_index):
        """Search should be case-insensitive."""
        results_lower = sample_index.search("music")
        results_upper = sample_index.search("MUSIC")
        assert len(results_lower) == len(results_upper)
        if results_lower and results_upper:
            assert results_lower[0].chunk_id == results_upper[0].chunk_id

    def test_result_includes_document(self, sample_index):
        """Results should include original document text."""
        results = sample_index.search("music")
        assert len(results) > 0
        assert "music" in results[0].document.lower()


class TestBM25IndexPersistence:
    """Tests for saving and loading BM25 index."""

    def test_save_and_load(self):
        """Should persist and restore index correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bm25_index.pkl"

            # Create and save index
            index = BM25Index()
            index.build_index(
                chunk_ids=["c1", "c2"],
                documents=["hello world", "goodbye world"],
            )
            index.save(path)

            # Load into new instance
            loaded_index = BM25Index.load(path)

            assert loaded_index.count() == 2
            assert set(loaded_index.get_chunk_ids()) == {"c1", "c2"}

    def test_loaded_index_search_works(self):
        """Loaded index should be searchable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bm25_index.pkl"

            # Use 4+ documents to avoid BM25 IDF edge cases
            index = BM25Index()
            index.build_index(
                chunk_ids=["c1", "c2", "c3", "c4"],
                documents=[
                    "hello wonderful beautiful world today",
                    "goodbye cruel harsh world forever",
                    "another random document here",
                    "yet more text content stuff",
                ],
            )
            index.save(path)

            loaded_index = BM25Index.load(path)
            results = loaded_index.search("wonderful beautiful")

            assert len(results) >= 1
            assert results[0].chunk_id == "c1"

    def test_load_nonexistent_raises(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            BM25Index.load(Path("/nonexistent/path/index.pkl"))

    def test_save_creates_parent_dirs(self):
        """Should create parent directories when saving."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "dirs" / "bm25_index.pkl"

            index = BM25Index()
            index.build_index(chunk_ids=["c1"], documents=["test"])
            index.save(path)

            assert path.exists()


class TestBM25IndexAddDocuments:
    """Tests for incremental document addition."""

    def test_add_documents(self):
        """Should add new documents to existing index."""
        index = BM25Index()
        index.build_index(
            chunk_ids=["c1"],
            documents=["hello world"],
        )
        assert index.count() == 1

        index.add_documents(
            chunk_ids=["c2", "c3"],
            documents=["goodbye world", "hello there"],
        )
        assert index.count() == 3
        assert set(index.get_chunk_ids()) == {"c1", "c2", "c3"}

    def test_add_documents_searchable(self):
        """Added documents should be searchable."""
        # Start with 3 documents to avoid BM25 IDF edge cases
        index = BM25Index()
        index.build_index(
            chunk_ids=["c1", "c2", "c3"],
            documents=[
                "hello wonderful amazing world today",
                "random text content here now",
                "more filler documents needed",
            ],
        )

        index.add_documents(
            chunk_ids=["c4"],
            documents=["goodbye special unique friend forever"],
        )

        results = index.search("special unique")
        assert len(results) >= 1
        assert results[0].chunk_id == "c4"

    def test_add_documents_mismatched_raises(self):
        """Should raise error if lengths don't match."""
        index = BM25Index()
        index.build_index(chunk_ids=["c1"], documents=["test"])

        with pytest.raises(ValueError, match="same length"):
            index.add_documents(chunk_ids=["c2", "c3"], documents=["only one"])


class TestBM25IndexAlbanianContent:
    """Tests for Albanian language content."""

    @pytest.fixture
    def albanian_index(self):
        """Create index with Albanian content."""
        index = BM25Index()
        index.build_index(
            chunk_ids=["c1", "c2", "c3"],
            documents=[
                "taka po flet për politikën dhe ekonominë",
                "keno dhe bas po diskutojnë muzikën rock",
                "jordi po tregon histori të vjetra",
            ],
        )
        return index

    def test_search_albanian_terms(self, albanian_index):
        """Should find Albanian terms."""
        results = albanian_index.search("politikën")
        assert len(results) > 0
        assert results[0].chunk_id == "c1"

    def test_search_nicknames(self, albanian_index):
        """Should find user nicknames."""
        results = albanian_index.search("keno")
        assert len(results) > 0
        assert results[0].chunk_id == "c2"

    def test_search_multiple_albanian_terms(self, albanian_index):
        """Should handle multiple Albanian terms."""
        results = albanian_index.search("muzikën rock")
        assert len(results) > 0
        assert results[0].chunk_id == "c2"
