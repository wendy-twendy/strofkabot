"""
Integration tests for the memory system.

These tests verify the full flow from memory extraction through storage,
including the tool calling interface and CRUD operations.
"""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.memory_store import Memory, MemoryStore
from strofkabot.openrouter import OpenRouterClient

# ============================================================================
# Fixtures
# ============================================================================


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


@pytest.fixture
def openrouter_client():
    """Create an OpenRouter client with mocked API key."""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return OpenRouterClient()


def make_tool_call(name: str, arguments: dict) -> MagicMock:
    """Helper to create a mock tool call."""
    tool_call = MagicMock()
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def make_api_response_with_tools(tool_calls: list) -> MagicMock:
    """Helper to create a mock API response with tool calls."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].message.content = None
    return response


def make_api_response_no_tools() -> MagicMock:
    """Helper to create a mock API response with no tool calls."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = None
    response.choices[0].message.content = None
    return response


# ============================================================================
# Test Memory Extraction Flow
# ============================================================================


class TestMemoryExtractionFlow:
    """Tests for the full memory extraction flow."""

    @pytest.mark.asyncio
    async def test_extract_and_save_user_memory(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting a user memory via tool calling and saving it."""
        user_id = 12345
        user_name = "Alice"

        # Mock the API to return a save_user_memory tool call
        tool_call = make_tool_call(
            "save_user_memory",
            {
                "user_id": user_id,
                "memory_text": "Alice is a software engineer",
                "category": "facts",
                "importance": 8,
                "confidence": 1.0,
                "tags": ["career", "tech"],
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Extract memories
        context_messages = [
            {
                "author": "Alice",
                "author_id": user_id,
                "content": "I work as a software engineer at Google",
                "timestamp": "12:00",
            }
        ]
        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="What do you do for work?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
        )

        # Verify extraction result
        assert len(extracted["user_memories"]) == 1
        mem_data = extracted["user_memories"][0]
        assert mem_data["user_id"] == user_id
        assert mem_data["memory_text"] == "Alice is a software engineer"
        assert mem_data["category"] == "facts"
        assert mem_data["importance"] == 8
        assert mem_data["confidence"] == 1.0
        assert mem_data["tags"] == ["career", "tech"]

        # Now save it to the store
        memory = Memory(
            text=mem_data["memory_text"],
            category=mem_data["category"],
            importance=mem_data["importance"],
            confidence=mem_data["confidence"],
            tags=mem_data["tags"],
        )
        await memory_store.add_user_memory(user_id, memory, user_name)

        # Verify it was saved
        saved_memories = await memory_store.get_user_memories(user_id)
        assert len(saved_memories) == 1
        assert saved_memories[0].text == "Alice is a software engineer"
        assert saved_memories[0].confidence == 1.0
        assert saved_memories[0].tags == ["career", "tech"]

    @pytest.mark.asyncio
    async def test_extract_and_save_server_memory(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting a server memory via tool calling and saving it."""
        # Mock the API to return a save_server_memory tool call
        tool_call = make_tool_call(
            "save_server_memory",
            {
                "memory_text": "The community has a tradition of Friday movie nights",
                "category": "knowledge",
                "importance": 8,
                "confidence": 0.9,
                "tags": ["traditions", "events"],
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Extract memories
        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="What traditions does this server have?",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
        )

        # Verify extraction result
        assert len(extracted["server_memories"]) == 1
        mem_data = extracted["server_memories"][0]
        assert "Friday movie nights" in mem_data["memory_text"]

        # Save it
        memory = Memory(
            text=mem_data["memory_text"],
            category=mem_data["category"],
            importance=mem_data["importance"],
            confidence=mem_data["confidence"],
            tags=mem_data["tags"],
        )
        await memory_store.add_server_memory(memory)

        # Verify it was saved
        saved_memories = await memory_store.get_server_memories()
        assert len(saved_memories) == 1
        assert "Friday movie nights" in saved_memories[0].text

    @pytest.mark.asyncio
    async def test_extract_no_memories_for_casual_chat(self, openrouter_client: OpenRouterClient):
        """Test that casual chat doesn't extract any memories."""
        # Mock the API to return no tool calls
        mock_response = make_api_response_no_tools()
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        extracted = await openrouter_client.extract_memories(
            context_messages=[
                {"author": "Bob", "author_id": 1, "content": "lol nice", "timestamp": "12:00"}
            ],
            question="haha",
            response_text="",
            user_id=1,
            user_name="Bob",
            known_users={"bob": 1},
        )

        assert len(extracted["user_memories"]) == 0
        assert len(extracted["server_memories"]) == 0
        assert len(extracted["user_updates"]) == 0
        assert len(extracted["server_updates"]) == 0
        assert len(extracted["user_invalidations"]) == 0
        assert len(extracted["server_invalidations"]) == 0

    @pytest.mark.asyncio
    async def test_extract_rejects_unknown_user_id(self, openrouter_client: OpenRouterClient):
        """Test that memories for unknown user IDs are rejected."""
        # Mock tool call with unknown user_id
        tool_call = make_tool_call(
            "save_user_memory",
            {
                "user_id": 99999,  # Not in known_users
                "memory_text": "Some fact",
                "category": "facts",
                "importance": 5,
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="test",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},  # Only user 1 is known
        )

        # Should be rejected because user_id 99999 is not known
        assert len(extracted["user_memories"]) == 0

    @pytest.mark.asyncio
    async def test_extract_rejects_bot_related_server_memory(
        self, openrouter_client: OpenRouterClient
    ):
        """Test that server memories about the bot are rejected."""
        # Mock tool call with bot-related content
        tool_call = make_tool_call(
            "save_server_memory",
            {
                "memory_text": "Llumi is the server's AI assistant",
                "category": "knowledge",
                "importance": 8,
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="test",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
        )

        # Should be rejected because it mentions "Llumi"
        assert len(extracted["server_memories"]) == 0


# ============================================================================
# Test Memory Update Flow
# ============================================================================


class TestMemoryUpdateFlow:
    """Tests for the memory update flow via tool calling."""

    @pytest.mark.asyncio
    async def test_extract_user_memory_update(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting and applying a user memory update."""
        user_id = 12345
        user_name = "Alice"

        # First, add an existing memory
        original = Memory(text="Alice lives in New York", category="facts", importance=7)
        await memory_store.add_user_memory(user_id, original, user_name)

        # Mock the API to return an update_user_memory tool call
        tool_call = make_tool_call(
            "update_user_memory",
            {
                "user_id": user_id,
                "old_memory_match": "New York",
                "new_memory_text": "Alice lives in Berlin",
                "category": "facts",
                "importance": 7,
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Pass existing memories to extraction
        existing_user_memories = await memory_store.get_user_memories(user_id)

        extracted = await openrouter_client.extract_memories(
            context_messages=[
                {
                    "author": "Alice",
                    "author_id": user_id,
                    "content": "I moved to Berlin last month!",
                    "timestamp": "12:00",
                }
            ],
            question="How's the new place?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
            existing_user_memories=existing_user_memories,
        )

        # Verify extraction result
        assert len(extracted["user_updates"]) == 1
        update = extracted["user_updates"][0]
        assert update["user_id"] == user_id
        assert update["old_match"] == "New York"
        assert update["new_memory_text"] == "Alice lives in Berlin"

        # Apply the update
        new_memory = Memory(
            text=update["new_memory_text"],
            category=update["category"],
            importance=update["importance"],
        )
        result = await memory_store.update_user_memory(user_id, update["old_match"], new_memory)

        assert result is True

        # Verify the update was applied
        memories = await memory_store.get_user_memories(user_id)
        assert len(memories) == 1
        assert memories[0].text == "Alice lives in Berlin"

    @pytest.mark.asyncio
    async def test_extract_server_memory_update(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting and applying a server memory update."""
        # First, add an existing memory
        original = Memory(
            text="The server was founded in 2020",
            category="knowledge",
            importance=8,
        )
        await memory_store.add_server_memory(original)

        # Mock the API to return an update_server_memory tool call
        tool_call = make_tool_call(
            "update_server_memory",
            {
                "old_memory_match": "founded in 2020",
                "new_memory_text": "The server was founded in 2019",
                "category": "knowledge",
                "importance": 8,
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        existing_server_memories = await memory_store.get_server_memories()

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="When was the server created?",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
            existing_server_memories=existing_server_memories,
        )

        # Verify extraction result
        assert len(extracted["server_updates"]) == 1
        update = extracted["server_updates"][0]

        # Apply the update
        new_memory = Memory(
            text=update["new_memory_text"],
            category=update["category"],
            importance=update["importance"],
        )
        result = await memory_store.update_server_memory(update["old_match"], new_memory)

        assert result is True

        # Verify the update was applied
        memories = await memory_store.get_server_memories()
        assert len(memories) == 1
        assert "2019" in memories[0].text


# ============================================================================
# Test Memory Invalidation Flow
# ============================================================================


class TestMemoryInvalidationFlow:
    """Tests for the memory invalidation flow via tool calling."""

    @pytest.mark.asyncio
    async def test_extract_user_memory_invalidation(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting and applying a user memory invalidation."""
        user_id = 12345
        user_name = "Alice"

        # First, add an existing memory
        original = Memory(text="Alice likes jazz music", category="preferences", importance=6)
        await memory_store.add_user_memory(user_id, original, user_name)

        # Mock the API to return an invalidate_user_memory tool call
        tool_call = make_tool_call(
            "invalidate_user_memory",
            {
                "user_id": user_id,
                "memory_text_match": "jazz",
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        existing_user_memories = await memory_store.get_user_memories(user_id)

        extracted = await openrouter_client.extract_memories(
            context_messages=[
                {
                    "author": "Alice",
                    "author_id": user_id,
                    "content": "I don't really like jazz anymore",
                    "timestamp": "12:00",
                }
            ],
            question="What music do you like now?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
            existing_user_memories=existing_user_memories,
        )

        # Verify extraction result
        assert len(extracted["user_invalidations"]) == 1
        inv = extracted["user_invalidations"][0]
        assert inv["user_id"] == user_id
        assert inv["text_match"] == "jazz"

        # Apply the invalidation
        result = await memory_store.invalidate_user_memory(user_id, inv["text_match"])

        assert result is True

        # Verify the memory was removed
        memories = await memory_store.get_user_memories(user_id)
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_extract_server_memory_invalidation(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting and applying a server memory invalidation."""
        # First, add an existing memory
        original = Memory(
            text="The server has a running joke about burned pasta",
            category="jokes",
            importance=7,
        )
        await memory_store.add_server_memory(original)

        # Mock the API to return an invalidate_server_memory tool call
        tool_call = make_tool_call(
            "invalidate_server_memory",
            {
                "memory_text_match": "burned pasta",
            },
        )
        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        existing_server_memories = await memory_store.get_server_memories()

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="That joke is old, right?",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
            existing_server_memories=existing_server_memories,
        )

        # Verify extraction result
        assert len(extracted["server_invalidations"]) == 1
        inv = extracted["server_invalidations"][0]

        # Apply the invalidation
        result = await memory_store.invalidate_server_memory(inv["text_match"])

        assert result is True

        # Verify the memory was removed
        memories = await memory_store.get_server_memories()
        assert len(memories) == 0


# ============================================================================
# Test Multiple Operations in One Extraction
# ============================================================================


class TestMultipleOperations:
    """Tests for extracting multiple memory operations at once."""

    @pytest.mark.asyncio
    async def test_extract_multiple_user_memories(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting multiple user memories at once."""
        user_id = 12345

        # Mock multiple tool calls
        tool_calls = [
            make_tool_call(
                "save_user_memory",
                {
                    "user_id": user_id,
                    "memory_text": "Alice is a software engineer",
                    "category": "facts",
                    "importance": 8,
                },
            ),
            make_tool_call(
                "save_user_memory",
                {
                    "user_id": user_id,
                    "memory_text": "Alice prefers dark mode",
                    "category": "preferences",
                    "importance": 5,
                },
            ),
        ]
        mock_response = make_api_response_with_tools(tool_calls)
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="Tell me about yourself",
            response_text="",
            user_id=user_id,
            user_name="Alice",
            known_users={"alice": user_id},
        )

        assert len(extracted["user_memories"]) == 2

    @pytest.mark.asyncio
    async def test_extract_save_update_and_invalidate_together(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test extracting a save, update, and invalidate in one call."""
        user_id = 12345

        # Add some existing memories
        await memory_store.add_user_memory(
            user_id, Memory(text="Lives in NYC", category="facts", importance=7)
        )
        await memory_store.add_user_memory(
            user_id, Memory(text="Likes coffee", category="preferences", importance=5)
        )

        # Mock multiple tool calls of different types
        tool_calls = [
            make_tool_call(
                "save_user_memory",
                {
                    "user_id": user_id,
                    "memory_text": "Started a new job",
                    "category": "events",
                    "importance": 7,
                },
            ),
            make_tool_call(
                "update_user_memory",
                {
                    "user_id": user_id,
                    "old_memory_match": "NYC",
                    "new_memory_text": "Lives in LA now",
                    "category": "facts",
                    "importance": 7,
                },
            ),
            make_tool_call(
                "invalidate_user_memory",
                {
                    "user_id": user_id,
                    "memory_text_match": "coffee",
                },
            ),
        ]
        mock_response = make_api_response_with_tools(tool_calls)
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        existing = await memory_store.get_user_memories(user_id)

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="Lots of changes!",
            response_text="",
            user_id=user_id,
            user_name="User",
            known_users={"user": user_id},
            existing_user_memories=existing,
        )

        # Verify all operation types were extracted
        assert len(extracted["user_memories"]) == 1
        assert len(extracted["user_updates"]) == 1
        assert len(extracted["user_invalidations"]) == 1

        # Apply all operations in order: invalidations first, then updates, then saves
        # (This is the order used in ai.py)

        # Invalidations
        for inv in extracted["user_invalidations"]:
            await memory_store.invalidate_user_memory(inv["user_id"], inv["text_match"])

        # Updates
        for upd in extracted["user_updates"]:
            new_memory = Memory(
                text=upd["new_memory_text"],
                category=upd["category"],
                importance=upd["importance"],
            )
            await memory_store.update_user_memory(upd["user_id"], upd["old_match"], new_memory)

        # Saves
        for mem_data in extracted["user_memories"]:
            memory = Memory(
                text=mem_data["memory_text"],
                category=mem_data["category"],
                importance=mem_data["importance"],
            )
            await memory_store.add_user_memory(mem_data["user_id"], memory)

        # Verify final state
        final_memories = await memory_store.get_user_memories(user_id)
        memory_texts = [m.text for m in final_memories]

        assert len(final_memories) == 2
        assert "Lives in LA now" in memory_texts  # Updated
        assert "Started a new job" in memory_texts  # New
        assert "coffee" not in str(memory_texts).lower()  # Invalidated


# ============================================================================
# Test Existing Memory Context
# ============================================================================


class TestExistingMemoryContext:
    """Tests for passing existing memories to extraction."""

    @pytest.mark.asyncio
    async def test_existing_memories_included_in_context(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that existing memories are included in the extraction prompt."""
        user_id = 12345
        user_name = "Alice"

        # Add existing memories
        await memory_store.add_user_memory(
            user_id,
            Memory(text="Alice likes Python", category="interests", importance=6),
            user_name,
        )
        await memory_store.add_server_memory(
            Memory(text="Server has game nights", category="knowledge", importance=7)
        )

        # Mock to capture the actual prompt
        mock_response = make_api_response_no_tools()
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        existing_user = await memory_store.get_user_memories(user_id)
        existing_server = await memory_store.get_server_memories()

        await openrouter_client.extract_memories(
            context_messages=[
                {"author": "Alice", "author_id": user_id, "content": "Hi", "timestamp": "12:00"}
            ],
            question="Hello",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
            existing_user_memories=existing_user,
            existing_server_memories=existing_server,
        )

        # Check that the call was made and examine the messages
        call_kwargs = openrouter_client._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]

        # The user content should include existing memories
        user_content = messages[1]["content"]
        assert "Already known about Alice" in user_content
        assert "Python" in user_content
        assert "Already known about this server" in user_content
        assert "game nights" in user_content


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in the memory extraction flow."""

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_empty_results(
        self, openrouter_client: OpenRouterClient
    ):
        """Test that API errors result in empty extraction results."""
        openrouter_client._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="test",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
        )

        assert extracted["user_memories"] == []
        assert extracted["server_memories"] == []
        assert extracted["user_updates"] == []
        assert extracted["server_updates"] == []
        assert extracted["user_invalidations"] == []
        assert extracted["server_invalidations"] == []

    @pytest.mark.asyncio
    async def test_malformed_tool_arguments_are_skipped(self, openrouter_client: OpenRouterClient):
        """Test that malformed tool call arguments are gracefully skipped."""
        # Create a tool call with invalid JSON
        tool_call = MagicMock()
        tool_call.function.name = "save_user_memory"
        tool_call.function.arguments = "not valid json"

        mock_response = make_api_response_with_tools([tool_call])
        openrouter_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        extracted = await openrouter_client.extract_memories(
            context_messages=[],
            question="test",
            response_text="",
            user_id=1,
            user_name="User",
            known_users={"user": 1},
        )

        # Should be empty since the tool call was invalid
        assert len(extracted["user_memories"]) == 0
