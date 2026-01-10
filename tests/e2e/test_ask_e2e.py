"""
End-to-end tests for the !ask command using real OpenRouter API.

These tests make actual API calls and require:
- OPENROUTER_API_KEY environment variable set
- Run with: pytest tests/e2e/test_ask_e2e.py -v -m e2e

These tests are skipped by default in normal test runs.
To run them: pytest -m e2e

Note: LLM behavior is non-deterministic. Some tests use soft assertions
(warnings) for behaviors that may vary, while hard assertions are used
for critical invariants (e.g., API must return valid response).
"""

import os
import warnings

import pytest

from strofkabot.openrouter import OpenRouterClient
from strofkabot.utils import build_system_prompt

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
def openrouter_client() -> OpenRouterClient:
    """Create a real OpenRouter client."""
    return OpenRouterClient()


# ============================================================================
# Query Classification Tests
# ============================================================================


class TestQueryClassificationE2E:
    """E2E tests for query classification using real API."""

    # --- Search Detection ---

    @pytest.mark.asyncio
    async def test_search_enabled_for_current_events(self, openrouter_client: OpenRouterClient):
        """Current events should trigger web search."""
        metadata = await openrouter_client.classify_query(
            "What's the weather in Berlin right now?",
            context_messages=[],
        )
        soft_assert(
            metadata.search is True,
            f"Expected search=True for weather query, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_search_enabled_for_prices(self, openrouter_client: OpenRouterClient):
        """Price queries should trigger web search."""
        metadata = await openrouter_client.classify_query(
            "What's the current price of Bitcoin?",
            context_messages=[],
        )
        soft_assert(
            metadata.search is True,
            f"Expected search=True for price query, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_search_enabled_for_sports(self, openrouter_client: OpenRouterClient):
        """Sports scores should trigger web search."""
        metadata = await openrouter_client.classify_query(
            "Who won the latest Champions League final?",
            context_messages=[],
        )
        soft_assert(
            metadata.search is True,
            f"Expected search=True for sports query, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_search_disabled_for_general_knowledge(self, openrouter_client: OpenRouterClient):
        """General knowledge shouldn't trigger search."""
        metadata = await openrouter_client.classify_query(
            "What is photosynthesis?",
            context_messages=[],
        )
        soft_assert(
            metadata.search is False,
            f"Expected search=False for general knowledge, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_search_disabled_for_casual_chat(self, openrouter_client: OpenRouterClient):
        """Casual chat shouldn't trigger search."""
        metadata = await openrouter_client.classify_query(
            "lol that's funny",
            context_messages=[],
        )
        soft_assert(
            metadata.search is False,
            f"Expected search=False for casual chat, got {metadata}",
        )

    # --- Thinking/Reasoning Detection ---

    @pytest.mark.asyncio
    async def test_thinking_enabled_for_math(self, openrouter_client: OpenRouterClient):
        """Math problems should trigger reasoning."""
        metadata = await openrouter_client.classify_query(
            "Calculate 127 * 38 + 15",
            context_messages=[],
        )
        soft_assert(
            metadata.thinking is True,
            f"Expected thinking=True for math problem, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_thinking_enabled_for_logic(self, openrouter_client: OpenRouterClient):
        """Logic puzzles should trigger reasoning."""
        metadata = await openrouter_client.classify_query(
            "If all A are B and all B are C, are all A also C?",
            context_messages=[],
        )
        soft_assert(
            metadata.thinking is True,
            f"Expected thinking=True for logic puzzle, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_thinking_enabled_for_analysis(self, openrouter_client: OpenRouterClient):
        """Analysis requests should trigger reasoning."""
        metadata = await openrouter_client.classify_query(
            "Analyze the pros and cons of microservices architecture",
            context_messages=[],
        )
        soft_assert(
            metadata.thinking is True,
            f"Expected thinking=True for analysis request, got {metadata}",
        )

    @pytest.mark.asyncio
    async def test_thinking_disabled_for_simple_question(self, openrouter_client: OpenRouterClient):
        """Simple questions shouldn't need reasoning."""
        metadata = await openrouter_client.classify_query(
            "What color is the sky?",
            context_messages=[],
        )
        soft_assert(
            metadata.thinking is False,
            f"Expected thinking=False for simple question, got {metadata}",
        )

    # --- Query Type Detection ---

    @pytest.mark.asyncio
    async def test_query_type_factual(self, openrouter_client: OpenRouterClient):
        """Factual questions should be classified correctly."""
        metadata = await openrouter_client.classify_query(
            "What is the capital of France?",
            context_messages=[],
        )
        soft_assert(
            metadata.query_type == "factual",
            f"Expected query_type='factual', got {metadata.query_type}",
        )

    @pytest.mark.asyncio
    async def test_query_type_creative(self, openrouter_client: OpenRouterClient):
        """Creative requests should be classified correctly."""
        metadata = await openrouter_client.classify_query(
            "Write me a haiku about coding",
            context_messages=[],
        )
        soft_assert(
            metadata.query_type == "creative",
            f"Expected query_type='creative', got {metadata.query_type}",
        )

    @pytest.mark.asyncio
    async def test_query_type_technical(self, openrouter_client: OpenRouterClient):
        """Technical questions should be classified correctly."""
        metadata = await openrouter_client.classify_query(
            "How do I reverse a list in Python?",
            context_messages=[],
        )
        soft_assert(
            metadata.query_type == "technical",
            f"Expected query_type='technical', got {metadata.query_type}",
        )

    @pytest.mark.asyncio
    async def test_query_type_opinion(self, openrouter_client: OpenRouterClient):
        """Opinion questions should be classified correctly."""
        metadata = await openrouter_client.classify_query(
            "Should I learn Rust or Go?",
            context_messages=[],
        )
        soft_assert(
            metadata.query_type == "opinion",
            f"Expected query_type='opinion', got {metadata.query_type}",
        )

    # --- Language Detection ---

    @pytest.mark.asyncio
    async def test_language_detection_english(self, openrouter_client: OpenRouterClient):
        """English language should be detected."""
        metadata = await openrouter_client.classify_query(
            "What time is it?",
            context_messages=[],
        )
        soft_assert(
            metadata.language == "en",
            f"Expected language='en', got {metadata.language}",
        )

    @pytest.mark.asyncio
    async def test_language_detection_serbian(self, openrouter_client: OpenRouterClient):
        """Serbian language should be detected."""
        metadata = await openrouter_client.classify_query(
            "Koji je glavni grad Srbije?",
            context_messages=[],
        )
        soft_assert(
            metadata.language == "sr",
            f"Expected language='sr', got {metadata.language}",
        )

    @pytest.mark.asyncio
    async def test_language_detection_albanian(self, openrouter_client: OpenRouterClient):
        """Albanian language should be detected."""
        metadata = await openrouter_client.classify_query(
            "Cfare ore eshte tani?",
            context_messages=[],
        )
        soft_assert(
            metadata.language == "sq",
            f"Expected language='sq', got {metadata.language}",
        )


# ============================================================================
# Full Ask Response Tests
# ============================================================================


class TestAskResponseE2E:
    """E2E tests for complete ask_with_context flow."""

    @pytest.mark.asyncio
    async def test_simple_question_returns_success(self, openrouter_client: OpenRouterClient):
        """Simple question should return successful response."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        response = await openrouter_client.ask_with_context(
            question="What is 2 + 2?",
            system_prompt=system_prompt,
            context_messages=[],
        )

        # Hard assertion: must succeed
        assert response.success is True, f"Expected success, got error: {response.error_message}"
        assert len(response.text) > 0, "Expected non-empty response"

        # Soft assertion: should mention the answer
        soft_assert(
            "4" in response.text,
            f"Expected '4' in response: {response.text[:200]}",
        )

    @pytest.mark.asyncio
    async def test_context_awareness(self, openrouter_client: OpenRouterClient):
        """Response should reference provided conversation context."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        context = [
            {
                "author": "Alice",
                "content": "I just got a new cat named Whiskers!",
                "timestamp": "12:00",
            },
            {
                "author": "Bob",
                "content": "That's awesome! What color is it?",
                "timestamp": "12:01",
            },
            {
                "author": "Alice",
                "content": "It's orange with white spots",
                "timestamp": "12:02",
            },
        ]

        response = await openrouter_client.ask_with_context(
            question="What was Alice's cat called again?",
            system_prompt=system_prompt,
            context_messages=context,
        )

        # Hard assertion: must succeed
        assert response.success is True, f"Expected success, got error: {response.error_message}"

        # Soft assertion: should reference the cat's name from context
        soft_assert(
            "whiskers" in response.text.lower(),
            f"Expected 'Whiskers' in response: {response.text[:200]}",
        )

    @pytest.mark.asyncio
    async def test_web_search_flag_set_correctly(self, openrouter_client: OpenRouterClient):
        """Web search queries should set search_used flag."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        # Pre-classify to force search
        metadata = await openrouter_client.classify_query(
            "What is the current price of gold per ounce?",
            context_messages=[],
        )

        response = await openrouter_client.ask_with_context(
            question="What is the current price of gold per ounce?",
            system_prompt=system_prompt,
            context_messages=[],
            query_metadata=metadata,
        )

        # Hard assertion: must succeed
        assert response.success is True, f"Expected success, got error: {response.error_message}"

        # Hard check: search flag should match what was requested
        if metadata.search:
            assert (
                response.search_used is True
            ), "Expected search_used=True when metadata.search=True"

    @pytest.mark.asyncio
    async def test_web_search_returns_current_info(self, openrouter_client: OpenRouterClient):
        """Web search should return current/recent information."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        # Force search with explicit metadata
        from strofkabot.openrouter import QueryMetadata

        metadata = QueryMetadata(
            search=True,
            thinking=False,
            reasoning_effort="low",
            query_type="factual",
            key_topics=["bitcoin", "price"],
            suggested_response_style="brief",
            language="en",
            requires_citations=True,
        )

        response = await openrouter_client.ask_with_context(
            question="What is the current price of Bitcoin in USD?",
            system_prompt=system_prompt,
            context_messages=[],
            query_metadata=metadata,
        )

        # Hard assertion: must succeed
        assert response.success is True, f"Expected success, got error: {response.error_message}"

        # Soft assertion: should mention a dollar amount or price
        has_price_indicator = (
            "$" in response.text
            or "USD" in response.text
            or any(char.isdigit() for char in response.text)
        )
        soft_assert(
            has_price_indicator,
            f"Expected price information in response: {response.text[:300]}",
        )

    @pytest.mark.asyncio
    async def test_response_no_markdown_headers(self, openrouter_client: OpenRouterClient):
        """Casual responses should not use markdown headers."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        response = await openrouter_client.ask_with_context(
            question="Tell me a short joke",
            system_prompt=system_prompt,
            context_messages=[],
        )

        # Hard assertion: must succeed
        assert response.success is True, f"Expected success, got error: {response.error_message}"

        # Soft assertion: casual responses shouldn't have markdown formatting
        has_markdown_headers = "##" in response.text
        has_bold_labels = "**" in response.text and ":**" in response.text

        soft_assert(
            not has_markdown_headers and not has_bold_labels,
            f"Expected no markdown formatting in casual response: {response.text[:200]}",
        )

    @pytest.mark.asyncio
    async def test_model_info_returned(self, openrouter_client: OpenRouterClient):
        """Response should include model information."""
        system_prompt = build_system_prompt(
            guild_name="Test Guild",
            channel_name="test-channel",
            user_name="TestUser",
        )

        response = await openrouter_client.ask_with_context(
            question="Hello!",
            system_prompt=system_prompt,
            context_messages=[],
        )

        # Hard assertion: must succeed and have model info
        assert response.success is True
        assert response.model_used is not None, "Expected model_used to be set"
        assert len(response.model_used) > 0, "Expected non-empty model_used"
