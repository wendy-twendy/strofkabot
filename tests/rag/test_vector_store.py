# Tests for VectorStore (TDD - write tests first)

"""Tests for ChromaDB vector store functionality.

These tests verify the VectorStore wrapper:
- Adding chunks with embeddings and metadata
- Searching by embedding similarity
- Filtering by metadata (channel, year, etc.)
- Persistence across restarts
- Empty collection handling
- Duplicate ID handling
"""

from __future__ import annotations

import pytest

from strofkabot.rag.vector_store import SearchResult, VectorStore


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_creation(self):
        """SearchResult should be creatable with all fields."""
        result = SearchResult(
            chunk_id="chunk-123",
            document="Some conversation text",
            metadata={"channel_name": "kanapeja", "year": 2024},
            distance=0.15,
        )

        assert result.chunk_id == "chunk-123"
        assert result.document == "Some conversation text"
        assert result.metadata["channel_name"] == "kanapeja"
        assert result.distance == 0.15

    def test_similarity_property(self):
        """similarity should be 1 - distance."""
        result = SearchResult(
            chunk_id="chunk-123",
            document="text",
            metadata={},
            distance=0.25,
        )
        assert result.similarity == pytest.approx(0.75)


class TestVectorStoreInit:
    """Tests for VectorStore initialization."""

    def test_creates_persist_directory(self, tmp_path):
        """Should create persist directory if it doesn't exist."""
        persist_dir = tmp_path / "chromadb"
        assert not persist_dir.exists()

        store = VectorStore(persist_dir=persist_dir)

        assert persist_dir.exists()

    def test_uses_default_collection_name(self, tmp_path):
        """Should use default collection name 'strofka_messages'."""
        store = VectorStore(persist_dir=tmp_path)
        assert store.collection_name == "strofka_messages"

    def test_custom_collection_name(self, tmp_path):
        """Should accept custom collection name."""
        store = VectorStore(persist_dir=tmp_path, collection_name="test_collection")
        assert store.collection_name == "test_collection"


class TestAddChunks:
    """Tests for add_chunks method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary VectorStore."""
        return VectorStore(persist_dir=tmp_path)

    def test_add_single_chunk(self, store):
        """Should add a single chunk with embedding and metadata."""
        chunk_ids = ["chunk-1"]
        embeddings = [[0.1] * 3072]  # 3072-dim vector
        documents = ["Takarak: hey everyone\nbasstein: hello!"]
        metadatas = [
            {
                "channel_id": 123,
                "channel_name": "kanapeja",
                "year": 2024,
                "month": 11,
                "participant_names": "Takarak,basstein",
                "message_count": 2,
            }
        ]

        store.add_chunks(chunk_ids, embeddings, documents, metadatas)

        # Verify it was added
        assert store.count() == 1

    def test_add_multiple_chunks(self, store):
        """Should add multiple chunks in batch."""
        chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        embeddings = [[0.1] * 3072, [0.2] * 3072, [0.3] * 3072]
        documents = ["Text 1", "Text 2", "Text 3"]
        metadatas = [
            {"channel_name": "kanapeja", "year": 2024},
            {"channel_name": "kanapeja", "year": 2024},
            {"channel_name": "shitpost", "year": 2025},
        ]

        store.add_chunks(chunk_ids, embeddings, documents, metadatas)

        assert store.count() == 3

    def test_add_empty_list(self, store):
        """Should handle empty lists gracefully."""
        store.add_chunks([], [], [], [])
        assert store.count() == 0

    def test_duplicate_ids_update(self, store):
        """Adding same chunk_id should update existing."""
        chunk_ids = ["chunk-1"]
        embeddings = [[0.1] * 3072]
        documents = ["Original text"]
        metadatas = [{"channel_name": "kanapeja"}]

        store.add_chunks(chunk_ids, embeddings, documents, metadatas)
        assert store.count() == 1

        # Add same ID with different content
        documents = ["Updated text"]
        store.add_chunks(chunk_ids, embeddings, documents, metadatas)

        # Count should still be 1 (upsert behavior)
        assert store.count() == 1


class TestSearch:
    """Tests for search method."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a VectorStore with test data."""
        store = VectorStore(persist_dir=tmp_path)

        # Create distinct embeddings for different topics
        chunk_ids = ["music-1", "politics-1", "gaming-1"]

        # Music embedding - high values in first half
        music_emb = [1.0] * 1536 + [0.1] * 1536

        # Politics embedding - high values in second half
        politics_emb = [0.1] * 1536 + [1.0] * 1536

        # Gaming embedding - medium values everywhere
        gaming_emb = [0.5] * 3072

        embeddings = [music_emb, politics_emb, gaming_emb]

        documents = [
            "Takarak: I love rock music\nbasstein: me too, especially metal",
            "shark: the election results are crazy\ngeri: politics is complicated",
            "jezi: anyone want to play valorant?\ntaka: sure, I'm in",
        ]

        metadatas = [
            {"channel_name": "muzika", "year": 2024, "topics": "music,rock,metal"},
            {"channel_name": "politike", "year": 2024, "topics": "politics,elections"},
            {"channel_name": "gaming", "year": 2025, "topics": "gaming,valorant"},
        ]

        store.add_chunks(chunk_ids, embeddings, documents, metadatas)
        return store

    def test_search_returns_results(self, populated_store):
        """Should return SearchResult objects."""
        # Query with music-like embedding
        query_embedding = [1.0] * 1536 + [0.1] * 1536

        results = populated_store.search(query_embedding, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_finds_most_similar(self, populated_store):
        """Should return most similar chunk first."""
        # Query similar to music embedding
        query_embedding = [0.9] * 1536 + [0.1] * 1536

        results = populated_store.search(query_embedding, k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "music-1"

    def test_search_respects_k_limit(self, populated_store):
        """Should return at most k results."""
        query_embedding = [0.5] * 3072

        results = populated_store.search(query_embedding, k=2)

        assert len(results) == 2

    def test_search_empty_collection(self, tmp_path):
        """Should return empty list for empty collection."""
        store = VectorStore(persist_dir=tmp_path)
        query_embedding = [0.5] * 3072

        results = store.search(query_embedding, k=10)

        assert results == []


class TestSearchWithFilters:
    """Tests for filtered search."""

    @pytest.fixture
    def store_with_metadata(self, tmp_path):
        """Create a VectorStore with diverse metadata."""
        store = VectorStore(persist_dir=tmp_path)

        chunk_ids = ["2024-jan", "2024-dec", "2025-jan"]
        # All embeddings similar so we test filter, not similarity
        embeddings = [[0.5] * 3072] * 3

        documents = ["January 2024", "December 2024", "January 2025"]

        metadatas = [
            {"channel_name": "kanapeja", "year": 2024, "month": 1},
            {"channel_name": "politike", "year": 2024, "month": 12},
            {"channel_name": "kanapeja", "year": 2025, "month": 1},
        ]

        store.add_chunks(chunk_ids, embeddings, documents, metadatas)
        return store

    def test_filter_by_channel(self, store_with_metadata):
        """Should filter by channel_name."""
        query_embedding = [0.5] * 3072

        results = store_with_metadata.search(
            query_embedding,
            k=10,
            where={"channel_name": "kanapeja"},
        )

        assert len(results) == 2
        assert all(r.metadata["channel_name"] == "kanapeja" for r in results)

    def test_filter_by_year(self, store_with_metadata):
        """Should filter by year."""
        query_embedding = [0.5] * 3072

        results = store_with_metadata.search(
            query_embedding,
            k=10,
            where={"year": 2024},
        )

        assert len(results) == 2
        assert all(r.metadata["year"] == 2024 for r in results)

    def test_filter_combined(self, store_with_metadata):
        """Should combine multiple filter conditions."""
        query_embedding = [0.5] * 3072

        results = store_with_metadata.search(
            query_embedding,
            k=10,
            where={
                "$and": [
                    {"channel_name": "kanapeja"},
                    {"year": 2024},
                ]
            },
        )

        assert len(results) == 1
        assert results[0].chunk_id == "2024-jan"

    def test_filter_no_matches(self, store_with_metadata):
        """Should return empty when filter matches nothing."""
        query_embedding = [0.5] * 3072

        results = store_with_metadata.search(
            query_embedding,
            k=10,
            where={"year": 2020},
        )

        assert results == []


class TestPersistence:
    """Tests for data persistence across restarts."""

    def test_data_persists(self, tmp_path):
        """Data should persist after store is recreated."""
        persist_dir = tmp_path / "chromadb"

        # Create store and add data
        store1 = VectorStore(persist_dir=persist_dir)
        store1.add_chunks(
            chunk_ids=["chunk-1"],
            embeddings=[[0.5] * 3072],
            documents=["Test document"],
            metadatas=[{"channel_name": "test"}],
        )
        assert store1.count() == 1

        # Delete store object
        del store1

        # Recreate store from same persist_dir
        store2 = VectorStore(persist_dir=persist_dir)

        # Data should still be there
        assert store2.count() == 1

        # Should be searchable
        results = store2.search([0.5] * 3072, k=1)
        assert len(results) == 1
        assert results[0].document == "Test document"


class TestGetChunk:
    """Tests for retrieving chunks by ID."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a VectorStore with test data."""
        store = VectorStore(persist_dir=tmp_path)
        store.add_chunks(
            chunk_ids=["chunk-1", "chunk-2"],
            embeddings=[[0.1] * 3072, [0.2] * 3072],
            documents=["Document 1", "Document 2"],
            metadatas=[
                {"channel_name": "kanapeja"},
                {"channel_name": "politike"},
            ],
        )
        return store

    def test_get_existing_chunk(self, store):
        """Should retrieve chunk by ID."""
        result = store.get_chunk("chunk-1")

        assert result is not None
        assert result["id"] == "chunk-1"
        assert result["document"] == "Document 1"
        assert result["metadata"]["channel_name"] == "kanapeja"

    def test_get_nonexistent_chunk(self, store):
        """Should return None for nonexistent ID."""
        result = store.get_chunk("nonexistent-id")
        assert result is None


class TestDeleteChunks:
    """Tests for deleting chunks."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a VectorStore with test data."""
        store = VectorStore(persist_dir=tmp_path)
        store.add_chunks(
            chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
            embeddings=[[0.1] * 3072, [0.2] * 3072, [0.3] * 3072],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[{"x": 1}, {"x": 2}, {"x": 3}],
        )
        return store

    def test_delete_single_chunk(self, store):
        """Should delete a single chunk by ID."""
        assert store.count() == 3

        store.delete_chunks(["chunk-2"])

        assert store.count() == 2
        assert store.get_chunk("chunk-2") is None

    def test_delete_multiple_chunks(self, store):
        """Should delete multiple chunks."""
        store.delete_chunks(["chunk-1", "chunk-3"])

        assert store.count() == 1
        assert store.get_chunk("chunk-2") is not None

    def test_delete_nonexistent_chunk(self, store):
        """Should handle deletion of nonexistent ID gracefully."""
        store.delete_chunks(["nonexistent"])
        assert store.count() == 3  # No change


class TestClearCollection:
    """Tests for clearing the entire collection."""

    def test_clear_removes_all(self, tmp_path):
        """Should remove all chunks from collection."""
        store = VectorStore(persist_dir=tmp_path)
        store.add_chunks(
            chunk_ids=["a", "b", "c"],
            embeddings=[[0.1] * 3072] * 3,
            documents=["1", "2", "3"],
            metadatas=[{}] * 3,
        )
        assert store.count() == 3

        store.clear()

        assert store.count() == 0


# E2E test - requires lancedb package
@pytest.mark.e2e
class TestVectorStoreE2E:
    """End-to-end tests with real LanceDB."""

    def test_full_workflow(self, tmp_path):
        """Test complete add/search/filter workflow."""
        store = VectorStore(persist_dir=tmp_path)

        # Add chunks
        store.add_chunks(
            chunk_ids=["music-chat", "politics-chat"],
            embeddings=[
                [1.0] * 1536 + [0.0] * 1536,  # Music embedding
                [0.0] * 1536 + [1.0] * 1536,  # Politics embedding
            ],
            documents=[
                "Taka: rock music is the best\nBas: I prefer metal",
                "Shark: election results\nGeri: politics discussion",
            ],
            metadatas=[
                {
                    "channel_name": "muzika",
                    "year": 2024,
                    "participant_names": "Taka,Bas",
                },
                {
                    "channel_name": "politike",
                    "year": 2024,
                    "participant_names": "Shark,Geri",
                },
            ],
        )

        # Search for music
        music_query = [0.9] * 1536 + [0.1] * 1536
        results = store.search(music_query, k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "music-chat"
        assert results[0].similarity > results[1].similarity

        # Filter by channel
        filtered = store.search(
            music_query,
            k=2,
            where={"channel_name": "politike"},
        )
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "politics-chat"
