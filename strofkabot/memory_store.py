"""File-based memory storage for the !ask command."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import aiofiles

from strofkabot.config import MEMORY_TEXT_MAX_LENGTH

logger = logging.getLogger(__name__)

# Similarity threshold for duplicate detection (0.7 allows for some variation)
DUPLICATE_SIMILARITY_THRESHOLD = 0.7


@dataclass
class Memory:
    """A single memory entry."""

    text: str
    category: str
    importance: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    access_count: int = 0

    def to_dict(self) -> dict:
        """Serialize memory to dictionary."""
        return {
            "text": self.text,
            "category": self.category,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        """Deserialize memory from dictionary."""
        created_at = datetime.fromisoformat(data["created_at"])
        last_accessed = None
        if data.get("last_accessed"):
            last_accessed = datetime.fromisoformat(data["last_accessed"])

        return cls(
            text=data["text"],
            category=data["category"],
            importance=data["importance"],
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data.get("access_count", 0),
        )


class MemoryStore:
    """File-based storage for user and server memories."""

    def __init__(self, memories_dir: Path):
        """Initialize the memory store.

        Args:
            memories_dir: Directory to store memory JSON files.
        """
        self.memories_dir = memories_dir
        self.users_dir = memories_dir / "users"
        self.server_file = memories_dir / "server.json"

    async def initialize(self) -> None:
        """Create necessary directories if they don't exist."""
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.users_dir.mkdir(exist_ok=True)

    def _get_user_file(self, user_id: int) -> Path:
        """Get the file path for a user's memories."""
        return self.users_dir / f"{user_id}.json"

    async def _read_user_data(self, user_id: int) -> dict:
        """Read user memory data from file."""
        user_file = self._get_user_file(user_id)
        if not user_file.exists():
            return {"user_id": user_id, "user_name": None, "memories": []}

        async with aiofiles.open(user_file) as f:
            content = await f.read()
            data = json.loads(content)
            # Ensure user_name field exists for older files
            if "user_name" not in data:
                data["user_name"] = None
            return data

    async def _write_user_data(self, user_id: int, data: dict) -> None:
        """Write user memory data to file."""
        user_file = self._get_user_file(user_id)
        async with aiofiles.open(user_file, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def _read_server_data(self) -> dict:
        """Read server memory data from file."""
        if not self.server_file.exists():
            return {"memories": []}

        async with aiofiles.open(self.server_file) as f:
            content = await f.read()
            return json.loads(content)

    async def _write_server_data(self, data: dict) -> None:
        """Write server memory data to file."""
        async with aiofiles.open(self.server_file, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def add_user_memory(
        self, user_id: int, memory: Memory, user_name: str | None = None
    ) -> None:
        """Add a memory for a specific user.

        Args:
            user_id: Discord user ID.
            memory: The memory to add.
            user_name: Optional display name for the user (stored in file for readability).
        """
        # Truncate text if too long
        if len(memory.text) > MEMORY_TEXT_MAX_LENGTH:
            memory.text = memory.text[:MEMORY_TEXT_MAX_LENGTH]

        data = await self._read_user_data(user_id)
        # Update user_name if provided (allows updating outdated names)
        if user_name:
            data["user_name"] = user_name
        data["memories"].append(memory.to_dict())
        await self._write_user_data(user_id, data)

        logger.debug(f"Added user memory for {user_id}: {memory.text[:50]}...")

    async def add_server_memory(self, memory: Memory) -> None:
        """Add a server-wide memory.

        Args:
            memory: The memory to add.
        """
        # Truncate text if too long
        if len(memory.text) > MEMORY_TEXT_MAX_LENGTH:
            memory.text = memory.text[:MEMORY_TEXT_MAX_LENGTH]

        data = await self._read_server_data()
        data["memories"].append(memory.to_dict())
        await self._write_server_data(data)

        logger.debug(f"Added server memory: {memory.text[:50]}...")

    async def get_user_memories(self, user_id: int, limit: int | None = None) -> list[Memory]:
        """Get memories for a specific user, sorted by importance.

        Args:
            user_id: Discord user ID.
            limit: Optional maximum number of memories to return.

        Returns:
            List of Memory objects, sorted by importance (descending).
        """
        data = await self._read_user_data(user_id)
        memories = [Memory.from_dict(m) for m in data["memories"]]

        # Sort by importance (desc), then by access_count (desc), then by created_at (desc)
        memories.sort(
            key=lambda m: (m.importance, m.access_count, m.created_at),
            reverse=True,
        )

        if limit:
            memories = memories[:limit]

        return memories

    async def get_server_memories(self, limit: int | None = None) -> list[Memory]:
        """Get server-wide memories, sorted by importance.

        Args:
            limit: Optional maximum number of memories to return.

        Returns:
            List of Memory objects, sorted by importance (descending).
        """
        data = await self._read_server_data()
        memories = [Memory.from_dict(m) for m in data["memories"]]

        # Sort by importance (desc), then by access_count (desc), then by created_at (desc)
        memories.sort(
            key=lambda m: (m.importance, m.access_count, m.created_at),
            reverse=True,
        )

        if limit:
            memories = memories[:limit]

        return memories

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate word overlap similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    async def is_duplicate_user_memory(self, user_id: int, text: str) -> bool:
        """Check if a similar memory already exists for a user.

        Args:
            user_id: Discord user ID.
            text: The memory text to check.

        Returns:
            True if a duplicate exists, False otherwise.
        """
        memories = await self.get_user_memories(user_id)

        for mem in memories:
            if self._text_similarity(text, mem.text) >= DUPLICATE_SIMILARITY_THRESHOLD:
                return True

        return False

    async def is_duplicate_server_memory(self, text: str) -> bool:
        """Check if a similar server memory already exists.

        Args:
            text: The memory text to check.

        Returns:
            True if a duplicate exists, False otherwise.
        """
        memories = await self.get_server_memories()

        for mem in memories:
            if self._text_similarity(text, mem.text) >= DUPLICATE_SIMILARITY_THRESHOLD:
                return True

        return False

    async def mark_user_memory_accessed(self, user_id: int, text: str) -> None:
        """Mark a user memory as accessed, updating stats.

        Args:
            user_id: Discord user ID.
            text: The memory text to mark as accessed.
        """
        data = await self._read_user_data(user_id)

        for mem_dict in data["memories"]:
            if mem_dict["text"] == text:
                mem_dict["last_accessed"] = datetime.now(UTC).isoformat()
                mem_dict["access_count"] = mem_dict.get("access_count", 0) + 1
                break

        await self._write_user_data(user_id, data)

    async def mark_server_memory_accessed(self, text: str) -> None:
        """Mark a server memory as accessed, updating stats.

        Args:
            text: The memory text to mark as accessed.
        """
        data = await self._read_server_data()

        for mem_dict in data["memories"]:
            if mem_dict["text"] == text:
                mem_dict["last_accessed"] = datetime.now(UTC).isoformat()
                mem_dict["access_count"] = mem_dict.get("access_count", 0) + 1
                break

        await self._write_server_data(data)

    async def prune_user_memories(self, user_id: int, limit: int = 20) -> None:
        """Prune user memories to keep only the top N by importance.

        Args:
            user_id: Discord user ID.
            limit: Maximum number of memories to keep.
        """
        data = await self._read_user_data(user_id)
        memories = [Memory.from_dict(m) for m in data["memories"]]

        if len(memories) <= limit:
            return

        # Sort by importance (desc), access_count (desc), created_at (desc)
        memories.sort(
            key=lambda m: (m.importance, m.access_count, m.created_at),
            reverse=True,
        )

        # Keep only top N
        pruned = memories[:limit]
        data["memories"] = [m.to_dict() for m in pruned]

        await self._write_user_data(user_id, data)
        logger.info(f"Pruned user {user_id} memories: {len(memories)} -> {limit}")

    async def prune_server_memories(self, limit: int = 50) -> None:
        """Prune server memories to keep only the top N by importance.

        Args:
            limit: Maximum number of memories to keep.
        """
        data = await self._read_server_data()
        memories = [Memory.from_dict(m) for m in data["memories"]]

        if len(memories) <= limit:
            return

        # Sort by importance (desc), access_count (desc), created_at (desc)
        memories.sort(
            key=lambda m: (m.importance, m.access_count, m.created_at),
            reverse=True,
        )

        # Keep only top N
        pruned = memories[:limit]
        data["memories"] = [m.to_dict() for m in pruned]

        await self._write_server_data(data)
        logger.info(f"Pruned server memories: {len(memories)} -> {limit}")
