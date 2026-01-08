"""Tests for OpenRouter client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strofkabot.openrouter_client import OpenRouterClient, OpenRouterResponse, QueryMetadata


def make_metadata(
    search: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "medium",
    query_type: str = "factual",
    key_topics: list[str] | None = None,
    suggested_response_style: str = "conversational",
    language: str = "en",
    requires_citations: bool = False,
    is_followup: bool = False,
) -> QueryMetadata:
    """Create a QueryMetadata instance with defaults."""
    return QueryMetadata(
        search=search,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        query_type=query_type,
        key_topics=key_topics or [],
        suggested_response_style=suggested_response_style,
        language=language,
        requires_citations=requires_citations,
        is_followup=is_followup,
    )


class TestQueryMetadata:
    """Tests for QueryMetadata dataclass."""

    def test_creates_metadata_with_all_fields(self):
        """Test that metadata is created with all fields."""
        metadata = QueryMetadata(
            search=True,
            thinking=True,
            reasoning_effort="high",
            query_type="technical",
            key_topics=["Python", "async"],
            suggested_response_style="detailed",
            language="sr",
            requires_citations=True,
            is_followup=True,
        )

        assert metadata.search is True
        assert metadata.thinking is True
        assert metadata.reasoning_effort == "high"
        assert metadata.query_type == "technical"
        assert metadata.key_topics == ["Python", "async"]
        assert metadata.suggested_response_style == "detailed"
        assert metadata.language == "sr"
        assert metadata.requires_citations is True
        assert metadata.is_followup is True


class TestOpenRouterResponse:
    """Tests for OpenRouterResponse dataclass."""

    def test_creates_successful_response(self):
        """Test creating a successful response."""
        response = OpenRouterResponse(
            text="Hello!",
            success=True,
            model_used="google/gemini-2.0-flash-exp:free",
        )

        assert response.text == "Hello!"
        assert response.success is True
        assert response.error_message is None

    def test_creates_error_response(self):
        """Test creating an error response."""
        response = OpenRouterResponse(
            text="",
            success=False,
            error_message="API error",
        )

        assert response.text == ""
        assert response.success is False
        assert response.error_message == "API error"


class TestOpenRouterClientInit:
    """Tests for OpenRouter client initialization."""

    def test_raises_without_api_key(self):
        """Test that ValueError is raised without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterClient()

    def test_initializes_with_api_key(self):
        """Test that client initializes with API key."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            client = OpenRouterClient()
            assert client._client is not None


class TestFormatContext:
    """Tests for _format_context method (XML format)."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterClient()

    def test_formats_simple_message(self, client):
        """Test formatting a simple message."""
        messages = [{"author": "Alice", "content": "Hello world", "timestamp": "12:00"}]

        result = client._format_context(messages)

        assert "<conversation>" in result
        assert "</conversation>" in result
        assert 'author="Alice"' in result
        assert 'time="12:00"' in result
        assert "Hello world" in result

    def test_formats_message_with_reply(self, client):
        """Test formatting a message that is a reply."""
        messages = [
            {
                "author": "Bob",
                "content": "I agree!",
                "timestamp": "12:05",
                "reply_to_author": "Alice",
            }
        ]

        result = client._format_context(messages)

        assert 'replying_to="Alice"' in result
        assert "I agree!" in result

    def test_formats_message_with_images(self, client):
        """Test formatting a message with image count."""
        messages = [
            {
                "author": "Carol",
                "content": "Check this out",
                "timestamp": "12:10",
                "image_count": 2,
            }
        ]

        result = client._format_context(messages)

        assert 'images="2"' in result

    def test_formats_multiple_messages(self, client):
        """Test formatting multiple messages."""
        messages = [
            {"author": "Alice", "content": "First message", "timestamp": "12:00"},
            {"author": "Bob", "content": "Second message", "timestamp": "12:01"},
            {"author": "Carol", "content": "Third message", "timestamp": "12:02"},
        ]

        result = client._format_context(messages)

        assert result.count("<message") == 3
        assert result.count("</message>") == 3
        assert "First message" in result
        assert "Second message" in result
        assert "Third message" in result


class TestBuildMessages:
    """Tests for _build_messages method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterClient()

    def test_builds_basic_messages(self, client):
        """Test building basic messages without context."""
        messages = client._build_messages(
            question="What is Python?",
            system_prompt="You are helpful.",
            context_messages=[],
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is Python?"

    def test_builds_messages_with_context(self, client):
        """Test building messages with context."""
        context = [{"author": "User", "content": "Previous message", "timestamp": "12:00"}]

        messages = client._build_messages(
            question="Follow up question",
            system_prompt="You are helpful.",
            context_messages=context,
        )

        # Should have: system, user (context), assistant (ack), user (question)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "<conversation>" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Follow up question"

    def test_builds_messages_with_url_context(self, client):
        """Test building messages with URL context."""
        url_context = (
            '<url_contents><url href="https://example.com">Article text</url></url_contents>'
        )

        messages = client._build_messages(
            question="What does the article say?",
            system_prompt="You are helpful.",
            context_messages=[],
            url_context=url_context,
        )

        # Should have: system, user (url_context), assistant (ack), user (question)
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert "url_contents" in messages[1]["content"]
        assert "Article text" in messages[1]["content"]

    def test_builds_messages_with_images(self, client):
        """Test building messages with images."""
        images = [
            {"data": "base64data", "mime_type": "image/jpeg"},
        ]

        messages = client._build_messages(
            question="What's in this image?",
            system_prompt="You are helpful.",
            context_messages=[],
            images=images,
        )

        assert len(messages) == 2
        # Question should be multimodal
        question_content = messages[1]["content"]
        assert isinstance(question_content, list)
        assert question_content[0]["type"] == "text"
        assert question_content[1]["type"] == "image_url"
        assert "data:image/jpeg;base64,base64data" in question_content[1]["image_url"]["url"]


class TestClassifyQuery:
    """Tests for classify_query method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterClient()

    @pytest.mark.asyncio
    async def test_classifies_simple_question(self, client):
        """Test classifying a simple factual question."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search": False,
                "thinking": False,
                "reasoning_effort": "minimal",
                "query_type": "factual",
                "key_topics": ["Python"],
                "suggested_response_style": "brief",
                "language": "en",
                "requires_citations": False,
                "is_followup": False,
            }
        )

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.classify_query("What is Python?", [])

        assert result.search is False
        assert result.thinking is False
        assert result.reasoning_effort == "minimal"
        assert result.query_type == "factual"
        assert "Python" in result.key_topics

    @pytest.mark.asyncio
    async def test_classifies_search_required_question(self, client):
        """Test classifying a question that needs search."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search": True,
                "thinking": False,
                "reasoning_effort": "low",
                "query_type": "factual",
                "key_topics": ["weather", "today"],
                "suggested_response_style": "brief",
                "language": "en",
                "requires_citations": True,
                "is_followup": False,
            }
        )

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.classify_query("What's the weather like today?", [])

        assert result.search is True
        assert result.requires_citations is True

    @pytest.mark.asyncio
    async def test_classifies_thinking_required_question(self, client):
        """Test classifying a question that needs thinking."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search": False,
                "thinking": True,
                "reasoning_effort": "high",
                "query_type": "technical",
                "key_topics": ["algorithm", "complexity"],
                "suggested_response_style": "detailed",
                "language": "en",
                "requires_citations": False,
                "is_followup": False,
            }
        )

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.classify_query("Explain the time complexity of quicksort", [])

        assert result.thinking is True
        assert result.reasoning_effort == "high"
        assert result.query_type == "technical"

    @pytest.mark.asyncio
    async def test_detects_followup_question(self, client):
        """Test detecting a follow-up question."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search": False,
                "thinking": False,
                "reasoning_effort": "low",
                "query_type": "factual",
                "key_topics": [],
                "suggested_response_style": "conversational",
                "language": "en",
                "requires_citations": False,
                "is_followup": True,
            }
        )

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        context = [{"author": "Alice", "content": "I love Python!", "timestamp": "12:00"}]
        result = await client.classify_query("What about JavaScript?", context)

        assert result.is_followup is True

    @pytest.mark.asyncio
    async def test_detects_non_english_language(self, client):
        """Test detecting non-English language."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "search": False,
                "thinking": False,
                "reasoning_effort": "minimal",
                "query_type": "factual",
                "key_topics": ["vreme"],
                "suggested_response_style": "brief",
                "language": "sr",
                "requires_citations": False,
                "is_followup": False,
            }
        )

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.classify_query("Koliko je sati?", [])

        assert result.language == "sr"

    @pytest.mark.asyncio
    async def test_handles_classification_failure(self, client):
        """Test that classification failure returns default metadata."""
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        result = await client.classify_query("Some question", [])

        # Should return default metadata
        assert result.search is False
        assert result.thinking is False
        assert result.reasoning_effort == "medium"
        assert result.query_type == "factual"
        assert result.language == "en"


class TestAskWithContext:
    """Tests for ask_with_context method."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterClient()

    @pytest.mark.asyncio
    async def test_successful_response(self, client):
        """Test a successful API response."""
        # Mock the classify_query result
        mock_classify_response = MagicMock()
        mock_classify_response.choices = [MagicMock()]
        mock_classify_response.choices[0].message.content = json.dumps(
            {
                "search": False,
                "thinking": False,
                "reasoning_effort": "medium",
                "query_type": "factual",
                "key_topics": [],
                "suggested_response_style": "conversational",
                "language": "en",
                "requires_citations": False,
                "is_followup": False,
            }
        )

        # Mock the actual response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is the AI response."

        client._client.chat.completions.create = AsyncMock(
            side_effect=[mock_classify_response, mock_response]
        )

        result = await client.ask_with_context(
            question="Hello",
            system_prompt="Be helpful",
            context_messages=[],
        )

        assert result.success is True
        assert result.text == "This is the AI response."

    @pytest.mark.asyncio
    async def test_uses_vision_model_for_images(self, client):
        """Test that vision model is used when images are present."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I can see the image."

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        images = [{"data": "base64data", "mime_type": "image/jpeg"}]

        result = await client.ask_with_context(
            question="What's in this image?",
            system_prompt="Be helpful",
            context_messages=[],
            images=images,
        )

        assert result.success is True
        # Should have called only once (no classification for images)
        assert client._client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_online_model_for_search(self, client):
        """Test that :online suffix is added when search is needed."""
        # Provide pre-computed metadata
        metadata = make_metadata(search=True)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Search results here."

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.ask_with_context(
            question="What's in the news today?",
            system_prompt="Be helpful",
            context_messages=[],
            query_metadata=metadata,
        )

        assert result.success is True
        assert result.search_used is True
        # Check that :online was in the model name
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert ":online" in call_kwargs["model"]

    @pytest.mark.asyncio
    async def test_includes_url_context(self, client):
        """Test that URL context is included in messages."""
        metadata = make_metadata()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Based on the article..."

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        url_context = (
            '<url_contents><url href="https://example.com">Article content</url></url_contents>'
        )

        result = await client.ask_with_context(
            question="Summarize this article",
            system_prompt="Be helpful",
            context_messages=[],
            query_metadata=metadata,
            url_context=url_context,
        )

        assert result.success is True
        # Verify URL context was passed
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # URL context should be in one of the user messages
        has_url_context = any("url_contents" in str(msg.get("content", "")) for msg in messages)
        assert has_url_context

    @pytest.mark.asyncio
    async def test_handles_api_error(self, client):
        """Test handling of API errors."""
        client._client.chat.completions.create = AsyncMock(side_effect=Exception("API unavailable"))

        result = await client.ask_with_context(
            question="Hello",
            system_prompt="Be helpful",
            context_messages=[],
        )

        assert result.success is False
        assert "API unavailable" in result.error_message

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, client):
        """Test handling of empty response from API."""
        metadata = make_metadata()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.ask_with_context(
            question="Hello",
            system_prompt="Be helpful",
            context_messages=[],
            query_metadata=metadata,
        )

        assert result.success is False
        assert "No text response" in result.error_message
