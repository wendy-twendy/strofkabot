"""
End-to-end tests for the memory system using real OpenRouter API.

These tests make actual API calls and require:
- OPENROUTER_API_KEY environment variable set
- Run with: pytest tests/e2e/test_memory_e2e.py -v -m e2e

These tests are skipped by default in normal test runs.
To run them: pytest -m e2e

Note: LLM behavior is non-deterministic. Some tests use soft assertions
(warnings) for behaviors that may vary, while hard assertions are used
for critical invariants (e.g., no extraction from casual chat).
"""

import os
import warnings
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from strofkabot.memory_store import Memory, MemoryStore
from strofkabot.openrouter import OpenRouterClient

# Skip all tests in this module if no API key or not explicitly running e2e tests
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
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
def openrouter_client() -> OpenRouterClient:
    """Create a real OpenRouter client."""
    return OpenRouterClient()


# ============================================================================
# E2E Tests - Memory Extraction
# ============================================================================


class TestMemoryExtractionE2E:
    """End-to-end tests for memory extraction using real API."""

    @pytest.mark.asyncio
    async def test_extracts_explicit_personal_fact(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that explicit personal facts are extracted (soft assertion).

        Note: The LLM may be conservative. This test verifies the API works
        and uses soft assertions for extraction behavior.
        """
        user_id = 12345
        user_name = "Alice"

        # Use very explicit, important personal information
        context_messages = [
            {
                "author": "Alice",
                "author_id": user_id,
                "content": (
                    "Just to let everyone know - I'm a professional software engineer "
                    "at Google and I've been coding for 10 years. This is important for "
                    "future conversations about tech topics."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="That's great! What kind of work do you do exactly?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
        )

        # Soft assertion - LLM may be conservative
        soft_assert(
            len(extracted["user_memories"]) >= 1,
            f"Expected extraction for explicit personal fact, got: {extracted}",
        )

        # If memories were extracted, verify content is relevant
        if extracted["user_memories"]:
            all_texts = [m["memory_text"].lower() for m in extracted["user_memories"]]
            combined = " ".join(all_texts)
            assert (
                "engineer" in combined or "software" in combined or "google" in combined
            ), f"Should mention software/engineer/Google, got: {all_texts}"

    @pytest.mark.asyncio
    async def test_extracts_clear_preference(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that clear preferences are extracted (soft assertion)."""
        user_id = 12345
        user_name = "Bob"

        # Use stronger, more permanent preference language
        context_messages = [
            {
                "author": "Bob",
                "author_id": user_id,
                "content": (
                    "I'm a strict vegetarian and have been for 15 years. "
                    "I never eat meat or fish. Please remember this for any "
                    "future food recommendations."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Any dietary restrictions I should know about?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"bob": user_id},
        )

        # Soft assertion - LLM may be conservative
        soft_assert(
            len(extracted["user_memories"]) >= 1,
            f"Expected extraction for clear preference, got: {extracted}",
        )

        # If memories were extracted, verify content is relevant
        if extracted["user_memories"]:
            all_texts = [m["memory_text"].lower() for m in extracted["user_memories"]]
            combined = " ".join(all_texts)
            assert (
                "vegetarian" in combined or "meat" in combined
            ), f"Should mention vegetarian/meat, got: {all_texts}"

    @pytest.mark.asyncio
    async def test_no_extraction_for_casual_chat(self, openrouter_client: OpenRouterClient):
        """Test that casual chat doesn't result in memory extraction."""
        user_id = 12345

        context_messages = [
            {
                "author": "User",
                "author_id": user_id,
                "content": "lol that's funny",
                "timestamp": "12:00",
                "is_bot": False,
            },
            {
                "author": "User",
                "author_id": user_id,
                "content": "haha yeah",
                "timestamp": "12:01",
                "is_bot": False,
            },
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="lmao",
            response_text="",
            user_id=user_id,
            user_name="User",
            known_users={"user": user_id},
        )

        # Should not extract anything meaningful from casual chat
        total_extracted = len(extracted["user_memories"]) + len(extracted["server_memories"])
        assert total_extracted == 0, f"Should not extract from casual chat, got: {extracted}"

    @pytest.mark.asyncio
    async def test_no_extraction_for_temporary_states(self, openrouter_client: OpenRouterClient):
        """Test that temporary states are not extracted."""
        user_id = 12345

        context_messages = [
            {
                "author": "User",
                "author_id": user_id,
                "content": "I'm so tired today, had a long day at work",
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="How are you?",
            response_text="",
            user_id=user_id,
            user_name="User",
            known_users={"user": user_id},
        )

        # Should not extract temporary states like "tired today"
        for mem in extracted["user_memories"]:
            text = mem["memory_text"].lower()
            assert "tired" not in text, f"Should not extract temporary state: {text}"


# ============================================================================
# E2E Tests - Memory Updates
# ============================================================================


class TestMemoryUpdateE2E:
    """End-to-end tests for memory updates using real API."""

    @pytest.mark.asyncio
    async def test_detects_location_change(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that location changes trigger updates or new memories (soft assertion)."""
        user_id = 12345
        user_name = "Alice"

        # Add existing memory
        await memory_store.add_user_memory(
            user_id,
            Memory(text="Alice lives in New York", category="facts", importance=7),
            user_name,
        )

        existing = await memory_store.get_user_memories(user_id)

        # Use explicit language about the change
        context_messages = [
            {
                "author": "Alice",
                "author_id": user_id,
                "content": (
                    "Big news everyone! I no longer live in New York - I permanently "
                    "moved to Berlin, Germany last month. Please update your records, "
                    "this is my new home now."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Wow, that's a big change! How's Berlin?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"alice": user_id},
            existing_user_memories=existing,
        )

        # Should either update the existing memory or invalidate + save new
        has_update = len(extracted["user_updates"]) > 0
        has_invalidation = len(extracted["user_invalidations"]) > 0
        has_new_memory = any(
            "berlin" in m["memory_text"].lower() for m in extracted["user_memories"]
        )

        # Soft assertion - LLM behavior may vary
        soft_assert(
            has_update or has_invalidation or has_new_memory,
            f"Expected location change detection. Got: {extracted}",
        )

    @pytest.mark.asyncio
    async def test_detects_preference_change(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that preference changes are detected (soft assertion)."""
        user_id = 12345
        user_name = "Bob"

        # Add existing memory
        await memory_store.add_user_memory(
            user_id,
            Memory(text="Bob prefers tea over coffee", category="preferences", importance=6),
            user_name,
        )

        existing = await memory_store.get_user_memories(user_id)

        # Use explicit contradiction language
        context_messages = [
            {
                "author": "Bob",
                "author_id": user_id,
                "content": (
                    "Update on my preferences: I no longer drink tea at all. "
                    "I've completely switched to coffee and drink 3 cups every day now. "
                    "Tea just doesn't do it for me anymore."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Wow, that's a change! What happened to the tea?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"bob": user_id},
            existing_user_memories=existing,
        )

        # Should detect the preference change
        has_update = len(extracted["user_updates"]) > 0
        has_invalidation = len(extracted["user_invalidations"]) > 0
        has_new_memory = any(
            "coffee" in m["memory_text"].lower() for m in extracted["user_memories"]
        )

        # Soft assertion - LLM behavior may vary
        soft_assert(
            has_update or has_invalidation or has_new_memory,
            f"Expected preference change detection. Got: {extracted}",
        )


# ============================================================================
# E2E Tests - Memory Invalidation
# ============================================================================


class TestMemoryInvalidationE2E:
    """End-to-end tests for memory invalidation using real API."""

    @pytest.mark.asyncio
    async def test_detects_explicit_negation(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test that explicit negations trigger invalidation (soft assertion)."""
        user_id = 12345
        user_name = "Carol"

        # Add existing memory
        await memory_store.add_user_memory(
            user_id,
            Memory(text="Carol plays guitar", category="interests", importance=6),
            user_name,
        )

        existing = await memory_store.get_user_memories(user_id)

        # Use very explicit negation language
        context_messages = [
            {
                "author": "Carol",
                "author_id": user_id,
                "content": (
                    "I need to correct something - I don't play guitar anymore and haven't "
                    "for years. I sold all my instruments. Please remove or update any "
                    "information saying I play guitar, it's no longer accurate."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Oh, that's a change! What happened to music?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"carol": user_id},
            existing_user_memories=existing,
        )

        # Should invalidate or update the guitar memory
        has_invalidation = len(extracted["user_invalidations"]) > 0
        has_update = len(extracted["user_updates"]) > 0

        # Soft assertion - LLM behavior may vary
        soft_assert(
            has_invalidation or has_update,
            f"Expected explicit negation detection. Got: {extracted}",
        )


# ============================================================================
# E2E Tests - Full Flow
# ============================================================================


class TestFullMemoryFlowE2E:
    """End-to-end tests for the complete memory flow."""

    @pytest.mark.asyncio
    async def test_complete_save_and_retrieve_flow(
        self, openrouter_client: OpenRouterClient, memory_store: MemoryStore
    ):
        """Test the complete flow: extract -> save -> retrieve (soft assertion)."""
        user_id = 12345
        user_name = "Dave"

        # Conversation with very explicit, important personal info
        context_messages = [
            {
                "author": "Dave",
                "author_id": user_id,
                "content": (
                    "For the record, here's my background: I'm a senior data scientist "
                    "with a PhD in Machine Learning. I've been working at a fintech startup "
                    "in San Francisco for 5 years. This is important context for future "
                    "technical discussions."
                ),
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        # Extract memories
        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Thanks for sharing! What's your area of expertise?",
            response_text="",
            user_id=user_id,
            user_name=user_name,
            known_users={"dave": user_id},
        )

        # Save extracted memories
        for mem_data in extracted["user_memories"]:
            memory = Memory(
                text=mem_data["memory_text"],
                category=mem_data["category"],
                importance=mem_data["importance"],
                confidence=mem_data.get("confidence", 1.0),
                tags=mem_data.get("tags", []),
            )
            await memory_store.add_user_memory(user_id, memory, user_name)

        # Retrieve and verify
        saved = await memory_store.get_user_memories(user_id)

        # Soft assertion - LLM may be conservative
        soft_assert(
            len(saved) >= 1,
            f"Expected at least one memory saved, got: {len(saved)}",
        )

        # If memories were saved, check content relevance
        if saved:
            all_texts = " ".join(m.text.lower() for m in saved)
            has_relevant_content = (
                "data scientist" in all_texts
                or "scientist" in all_texts
                or "startup" in all_texts
                or "san francisco" in all_texts
                or "machine learning" in all_texts
                or "phd" in all_texts
            )
            assert has_relevant_content, f"Should contain relevant info, got: {all_texts}"

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, openrouter_client: OpenRouterClient):
        """Test that confidence scores are assigned appropriately."""
        user_id = 12345

        # Explicit statement should have high confidence
        context_messages = [
            {
                "author": "User",
                "author_id": user_id,
                "content": "I am a vegetarian, I don't eat any meat at all.",
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="Any dietary restrictions?",
            response_text="",
            user_id=user_id,
            user_name="User",
            known_users={"user": user_id},
        )

        if extracted["user_memories"]:
            # Explicit statements should have high confidence
            for mem in extracted["user_memories"]:
                confidence = mem.get("confidence", 1.0)
                assert (
                    confidence >= 0.7
                ), f"Explicit statement should have confidence >= 0.7, got {confidence}"

    @pytest.mark.asyncio
    async def test_tags_are_extracted(self, openrouter_client: OpenRouterClient):
        """Test that relevant tags are extracted."""
        user_id = 12345

        context_messages = [
            {
                "author": "User",
                "author_id": user_id,
                "content": "I love programming in Python and JavaScript",
                "timestamp": "12:00",
                "is_bot": False,
            }
        ]

        extracted = await openrouter_client.extract_memories(
            context_messages=context_messages,
            question="What languages do you use?",
            response_text="",
            user_id=user_id,
            user_name="User",
            known_users={"user": user_id},
        )

        # Check if any memories have tags
        has_tags = any(len(mem.get("tags", [])) > 0 for mem in extracted["user_memories"])

        # Tags are optional, but if present they should be relevant
        if has_tags:
            all_tags = []
            for mem in extracted["user_memories"]:
                all_tags.extend(mem.get("tags", []))
            all_tags_lower = [t.lower() for t in all_tags]

            # Should have programming-related tags
            has_relevant_tag = any(
                tag in all_tags_lower
                for tag in ["programming", "python", "javascript", "coding", "tech"]
            )
            assert has_relevant_tag, f"Tags should be relevant, got: {all_tags}"
