"""
Tests for the MemoryStore class.
"""

import datetime
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from strofkabot.memory_store import Memory, MemoryStore


@pytest.fixture
def temp_memories_dir(tmp_path: Path) -> Path:
    """Provide a temporary memories directory for testing."""
    return tmp_path / "memories"


@pytest.fixture
async def memory_store(
    temp_memories_dir: Path,
) -> AsyncGenerator[MemoryStore, None]:
    """Fixture providing an initialized MemoryStore instance."""
    store = MemoryStore(temp_memories_dir)
    await store.initialize()
    yield store


class TestMemoryStoreInitialization:
    """Tests for MemoryStore initialization and directory creation."""

    async def test_initialize_creates_directories(self, temp_memories_dir: Path):
        """Verify that initialize() creates the required directories."""
        store = MemoryStore(temp_memories_dir)
        await store.initialize()

        assert temp_memories_dir.exists()
        assert (temp_memories_dir / "users").exists()

    async def test_initialize_is_idempotent(self, temp_memories_dir: Path):
        """Verify that multiple initialize calls don't fail."""
        store = MemoryStore(temp_memories_dir)
        await store.initialize()
        await store.initialize()

        assert temp_memories_dir.exists()

    async def test_server_memories_file_created_on_first_add(
        self, memory_store: MemoryStore, temp_memories_dir: Path
    ):
        """Verify that server.json is created when adding first server memory."""
        memory = Memory(
            text="Test server memory",
            category="general",
            importance=5,
        )
        await memory_store.add_server_memory(memory)

        assert (temp_memories_dir / "server.json").exists()


class TestMemoryDataclass:
    """Tests for Memory dataclass defaults and serialization."""

    def test_memory_defaults(self):
        """Test that Memory has correct default values."""
        memory = Memory(text="Test", category="general", importance=5)

        assert memory.text == "Test"
        assert memory.category == "general"
        assert memory.importance == 5
        assert memory.created_at is not None
        assert memory.last_accessed is None
        assert memory.access_count == 0

    def test_memory_to_dict(self):
        """Test Memory serialization to dict."""
        memory = Memory(
            text="Test",
            category="preferences",
            importance=8,
            created_at=datetime.datetime(2025, 1, 9, 12, 0, 0, tzinfo=datetime.UTC),
        )
        data = memory.to_dict()

        assert data["text"] == "Test"
        assert data["category"] == "preferences"
        assert data["importance"] == 8
        assert data["created_at"] == "2025-01-09T12:00:00+00:00"
        assert data["last_accessed"] is None
        assert data["access_count"] == 0

    def test_memory_from_dict(self):
        """Test Memory deserialization from dict."""
        data = {
            "text": "Test memory",
            "category": "facts",
            "importance": 7,
            "created_at": "2025-01-09T12:00:00+00:00",
            "last_accessed": "2025-01-09T14:00:00+00:00",
            "access_count": 3,
        }
        memory = Memory.from_dict(data)

        assert memory.text == "Test memory"
        assert memory.category == "facts"
        assert memory.importance == 7
        assert memory.access_count == 3


class TestUserMemoryCRUD:
    """Tests for user memory create and read operations."""

    async def test_add_user_memory(self, memory_store: MemoryStore, temp_memories_dir: Path):
        """Test adding a single user memory."""
        user_id = 12345
        memory = Memory(
            text="Prefers vegetarian food",
            category="preferences",
            importance=8,
        )
        await memory_store.add_user_memory(user_id, memory)

        # Check file was created
        user_file = temp_memories_dir / "users" / f"{user_id}.json"
        assert user_file.exists()

        # Check content
        with open(user_file) as f:
            data = json.load(f)
        assert data["user_id"] == user_id
        assert len(data["memories"]) == 1
        assert data["memories"][0]["text"] == "Prefers vegetarian food"

    async def test_add_user_memory_with_username(
        self, memory_store: MemoryStore, temp_memories_dir: Path
    ):
        """Test adding a user memory with username stores the name."""
        user_id = 12345
        user_name = "Alice"
        memory = Memory(
            text="Prefers tea over coffee",
            category="preferences",
            importance=6,
        )
        await memory_store.add_user_memory(user_id, memory, user_name)

        # Check file contains username
        user_file = temp_memories_dir / "users" / f"{user_id}.json"
        with open(user_file) as f:
            data = json.load(f)
        assert data["user_id"] == user_id
        assert data["user_name"] == "Alice"
        assert len(data["memories"]) == 1

    async def test_username_updates_on_new_memory(
        self, memory_store: MemoryStore, temp_memories_dir: Path
    ):
        """Test that username gets updated when adding new memories."""
        user_id = 12345
        memory1 = Memory(text="First memory", category="general", importance=5)
        memory2 = Memory(text="Second memory", category="general", importance=5)

        # Add first memory with original name
        await memory_store.add_user_memory(user_id, memory1, "OldName")

        # Add second memory with updated name
        await memory_store.add_user_memory(user_id, memory2, "NewName")

        # Check username was updated
        user_file = temp_memories_dir / "users" / f"{user_id}.json"
        with open(user_file) as f:
            data = json.load(f)
        assert data["user_name"] == "NewName"
        assert len(data["memories"]) == 2

    async def test_add_multiple_user_memories(self, memory_store: MemoryStore):
        """Test adding multiple memories for the same user."""
        user_id = 12345
        memories = [
            Memory(text="Likes Python", category="interests", importance=6),
            Memory(text="Works as engineer", category="facts", importance=7),
            Memory(text="Prefers dark mode", category="preferences", importance=5),
        ]
        for mem in memories:
            await memory_store.add_user_memory(user_id, mem)

        result = await memory_store.get_user_memories(user_id)
        assert len(result) == 3

    async def test_get_user_memories_empty(self, memory_store: MemoryStore):
        """Test getting memories for user with no memories returns empty list."""
        result = await memory_store.get_user_memories(999999)
        assert result == []

    async def test_get_user_memories_sorted_by_importance(self, memory_store: MemoryStore):
        """Test that memories are returned sorted by importance (desc)."""
        user_id = 12345
        memories = [
            Memory(text="Low importance", category="general", importance=3),
            Memory(text="High importance", category="general", importance=9),
            Memory(text="Medium importance", category="general", importance=6),
        ]
        for mem in memories:
            await memory_store.add_user_memory(user_id, mem)

        result = await memory_store.get_user_memories(user_id)

        assert result[0].text == "High importance"
        assert result[1].text == "Medium importance"
        assert result[2].text == "Low importance"

    async def test_get_user_memories_with_limit(self, memory_store: MemoryStore):
        """Test limiting number of returned memories."""
        user_id = 12345
        for i in range(10):
            mem = Memory(text=f"Memory {i}", category="general", importance=i)
            await memory_store.add_user_memory(user_id, mem)

        result = await memory_store.get_user_memories(user_id, limit=3)

        assert len(result) == 3
        # Should be the top 3 by importance
        assert result[0].importance == 9
        assert result[1].importance == 8
        assert result[2].importance == 7


class TestServerMemoryCRUD:
    """Tests for server memory create and read operations."""

    async def test_add_server_memory(self, memory_store: MemoryStore, temp_memories_dir: Path):
        """Test adding a single server memory."""
        memory = Memory(
            text="Running joke about burned pasta",
            category="jokes",
            importance=7,
        )
        await memory_store.add_server_memory(memory)

        # Check file was created
        server_file = temp_memories_dir / "server.json"
        assert server_file.exists()

        # Check content
        with open(server_file) as f:
            data = json.load(f)
        assert len(data["memories"]) == 1
        assert data["memories"][0]["text"] == "Running joke about burned pasta"

    async def test_add_multiple_server_memories(self, memory_store: MemoryStore):
        """Test adding multiple server memories."""
        memories = [
            Memory(text="Joke 1", category="jokes", importance=6),
            Memory(text="Event 1", category="events", importance=8),
            Memory(text="Meme 1", category="memes", importance=5),
        ]
        for mem in memories:
            await memory_store.add_server_memory(mem)

        result = await memory_store.get_server_memories()
        assert len(result) == 3

    async def test_get_server_memories_empty(
        self, memory_store: MemoryStore, temp_memories_dir: Path
    ):
        """Test getting server memories when file doesn't exist returns empty list."""
        result = await memory_store.get_server_memories()
        assert result == []

    async def test_get_server_memories_sorted_by_importance(self, memory_store: MemoryStore):
        """Test that server memories are returned sorted by importance (desc)."""
        memories = [
            Memory(text="Low", category="general", importance=2),
            Memory(text="High", category="general", importance=9),
            Memory(text="Medium", category="general", importance=5),
        ]
        for mem in memories:
            await memory_store.add_server_memory(mem)

        result = await memory_store.get_server_memories()

        assert result[0].text == "High"
        assert result[1].text == "Medium"
        assert result[2].text == "Low"

    async def test_get_server_memories_with_limit(self, memory_store: MemoryStore):
        """Test limiting number of returned server memories."""
        for i in range(10):
            mem = Memory(text=f"Memory {i}", category="general", importance=i)
            await memory_store.add_server_memory(mem)

        result = await memory_store.get_server_memories(limit=3)

        assert len(result) == 3
        assert result[0].importance == 9


class TestDuplicateDetection:
    """Tests for duplicate memory detection."""

    async def test_is_duplicate_exact_match(self, memory_store: MemoryStore):
        """Test that exact duplicate is detected."""
        user_id = 12345
        memory = Memory(text="Prefers vegetarian food", category="preferences", importance=8)
        await memory_store.add_user_memory(user_id, memory)

        is_dup = await memory_store.is_duplicate_user_memory(user_id, "Prefers vegetarian food")
        assert is_dup is True

    async def test_is_duplicate_high_similarity(self, memory_store: MemoryStore):
        """Test that high similarity text is detected as duplicate."""
        user_id = 12345
        memory = Memory(
            text="Prefers vegetarian",
            category="preferences",
            importance=8,
        )
        await memory_store.add_user_memory(user_id, memory)

        # Very similar text - 2 of 3 words match = 0.66 Jaccard, but
        # let's test with higher overlap: 3 words, 2 match, union 4 = 0.5
        # Need to craft it better: "Prefers vegetarian meals" vs "Prefers vegetarian"
        # Union: {"prefers", "vegetarian", "meals"} = 3
        # Intersection: {"prefers", "vegetarian"} = 2
        # 2/3 = 0.66, still below 0.7

        # Better: use almost identical text with just one word different
        await memory_store.add_user_memory(
            user_id,
            Memory(text="Alice loves cooking pasta dishes", category="interests", importance=7),
        )

        # 4 out of 5 words match = 4/6 = 0.66... need more overlap
        # "Alice loves cooking pasta" - 4 words
        # vs original 5 words, intersection 4, union 5 = 4/5 = 0.8
        is_dup = await memory_store.is_duplicate_user_memory(user_id, "Alice loves cooking pasta")
        assert is_dup is True

    async def test_is_not_duplicate_different_text(self, memory_store: MemoryStore):
        """Test that different text is not detected as duplicate."""
        user_id = 12345
        memory = Memory(text="Prefers vegetarian food", category="preferences", importance=8)
        await memory_store.add_user_memory(user_id, memory)

        is_dup = await memory_store.is_duplicate_user_memory(
            user_id, "Works as a software engineer"
        )
        assert is_dup is False

    async def test_is_duplicate_server_memory(self, memory_store: MemoryStore):
        """Test duplicate detection for server memories."""
        memory = Memory(
            text="Running joke about burned pasta",
            category="jokes",
            importance=7,
        )
        await memory_store.add_server_memory(memory)

        is_dup = await memory_store.is_duplicate_server_memory("Running joke about burned pasta")
        assert is_dup is True

        is_not_dup = await memory_store.is_duplicate_server_memory("Monthly game nights")
        assert is_not_dup is False


class TestTextSimilarity:
    """Tests for text similarity calculation."""

    def test_similarity_identical(self, memory_store: MemoryStore):
        """Test that identical texts have similarity of 1.0."""
        sim = memory_store._text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_similarity_completely_different(self, memory_store: MemoryStore):
        """Test that completely different texts have similarity of 0.0."""
        sim = memory_store._text_similarity("hello world", "foo bar")
        assert sim == 0.0

    def test_similarity_partial_overlap(self, memory_store: MemoryStore):
        """Test partial word overlap gives intermediate similarity."""
        sim = memory_store._text_similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_similarity_case_insensitive(self, memory_store: MemoryStore):
        """Test that similarity is case-insensitive."""
        sim = memory_store._text_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_similarity_empty_strings(self, memory_store: MemoryStore):
        """Test that empty strings return 0.0 similarity."""
        assert memory_store._text_similarity("", "hello") == 0.0
        assert memory_store._text_similarity("hello", "") == 0.0
        assert memory_store._text_similarity("", "") == 0.0


class TestMarkAccessed:
    """Tests for updating memory access statistics."""

    async def test_mark_user_memory_accessed(self, memory_store: MemoryStore):
        """Test that marking a memory as accessed updates stats."""
        user_id = 12345
        memory = Memory(
            text="Test memory",
            category="general",
            importance=5,
        )
        await memory_store.add_user_memory(user_id, memory)

        await memory_store.mark_user_memory_accessed(user_id, "Test memory")

        memories = await memory_store.get_user_memories(user_id)
        assert memories[0].access_count == 1
        assert memories[0].last_accessed is not None

    async def test_mark_multiple_accesses(self, memory_store: MemoryStore):
        """Test that multiple accesses increment the counter."""
        user_id = 12345
        memory = Memory(text="Test memory", category="general", importance=5)
        await memory_store.add_user_memory(user_id, memory)

        await memory_store.mark_user_memory_accessed(user_id, "Test memory")
        await memory_store.mark_user_memory_accessed(user_id, "Test memory")
        await memory_store.mark_user_memory_accessed(user_id, "Test memory")

        memories = await memory_store.get_user_memories(user_id)
        assert memories[0].access_count == 3

    async def test_mark_server_memory_accessed(self, memory_store: MemoryStore):
        """Test marking server memory as accessed."""
        memory = Memory(text="Server memory", category="general", importance=5)
        await memory_store.add_server_memory(memory)

        await memory_store.mark_server_memory_accessed("Server memory")

        memories = await memory_store.get_server_memories()
        assert memories[0].access_count == 1


class TestPruning:
    """Tests for memory pruning logic."""

    async def test_prune_user_memories_keeps_limit(self, memory_store: MemoryStore):
        """Test that pruning keeps only the top N memories by importance."""
        user_id = 12345
        # Add 25 memories (more than the 20 limit)
        for i in range(25):
            mem = Memory(text=f"Memory {i}", category="general", importance=i % 10)
            await memory_store.add_user_memory(user_id, mem)

        await memory_store.prune_user_memories(user_id, limit=20)

        memories = await memory_store.get_user_memories(user_id)
        assert len(memories) == 20

    async def test_prune_keeps_highest_importance(self, memory_store: MemoryStore):
        """Test that pruning keeps the highest importance memories."""
        user_id = 12345
        # Add memories with varying importance
        for i in range(10):
            mem = Memory(text=f"Low {i}", category="general", importance=1)
            await memory_store.add_user_memory(user_id, mem)
        for i in range(5):
            mem = Memory(text=f"High {i}", category="general", importance=9)
            await memory_store.add_user_memory(user_id, mem)

        await memory_store.prune_user_memories(user_id, limit=5)

        memories = await memory_store.get_user_memories(user_id)
        assert len(memories) == 5
        # All remaining should be high importance
        for mem in memories:
            assert mem.importance == 9

    async def test_prune_server_memories(self, memory_store: MemoryStore):
        """Test pruning server memories."""
        # Add 60 memories (more than the 50 limit)
        for i in range(60):
            mem = Memory(text=f"Memory {i}", category="general", importance=i % 10)
            await memory_store.add_server_memory(mem)

        await memory_store.prune_server_memories(limit=50)

        memories = await memory_store.get_server_memories()
        assert len(memories) == 50

    async def test_prune_does_nothing_under_limit(self, memory_store: MemoryStore):
        """Test that pruning doesn't remove anything if under limit."""
        user_id = 12345
        for i in range(5):
            mem = Memory(text=f"Memory {i}", category="general", importance=5)
            await memory_store.add_user_memory(user_id, mem)

        await memory_store.prune_user_memories(user_id, limit=20)

        memories = await memory_store.get_user_memories(user_id)
        assert len(memories) == 5


class TestMemoryTextLength:
    """Tests for memory text length enforcement."""

    async def test_memory_text_truncated_if_too_long(self, memory_store: MemoryStore):
        """Test that long memory text is truncated."""
        user_id = 12345
        long_text = "x" * 300  # Over 200 char limit
        memory = Memory(text=long_text, category="general", importance=5)

        await memory_store.add_user_memory(user_id, memory)

        memories = await memory_store.get_user_memories(user_id)
        assert len(memories[0].text) == 200
