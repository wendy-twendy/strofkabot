"""Tests for RAG query rewriter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strofkabot.rag.query_rewriter import (
    ConversationMessage,
    MemberInfo,
    RewrittenQuery,
    format_conversation_xml,
    format_members_xml,
    rewrite_query,
)


@pytest.fixture
def mock_client():
    """Create a mock AsyncOpenAI client."""
    return AsyncMock()


@pytest.fixture
def sample_members():
    """Sample member list for testing."""
    return [
        MemberInfo(
            author_id=301411562487545857,
            display_name="Takarak",
            username=".takarak",
            nicknames=["taka"],
        ),
        MemberInfo(
            author_id=238553996657295361,
            display_name="dave_a7x",
            username="dave_a7x",
            nicknames=["dejv", "dejvi", "dave"],
        ),
    ]


@pytest.fixture
def sample_conversation():
    """Sample conversation history for testing."""
    return [
        ConversationMessage(
            author="Takarak",
            author_id=301411562487545857,
            content="I think AI is going to change everything",
        ),
        ConversationMessage(
            author="dave_a7x",
            author_id=238553996657295361,
            content="He's being too optimistic again",
        ),
    ]


class TestDataclasses:
    """Tests for dataclasses."""

    def test_member_info_creation(self):
        """Test MemberInfo creation."""
        member = MemberInfo(
            author_id=123456,
            display_name="Test User",
            username="testuser",
            nicknames=["test", "testy"],
        )
        assert member.author_id == 123456
        assert member.display_name == "Test User"
        assert member.username == "testuser"
        assert member.nicknames == ["test", "testy"]

    def test_conversation_message_creation(self):
        """Test ConversationMessage creation."""
        msg = ConversationMessage(
            author="Test User",
            author_id=123456,
            content="Hello world",
        )
        assert msg.author == "Test User"
        assert msg.author_id == 123456
        assert msg.content == "Hello world"


class TestRewrittenQuery:
    """Tests for RewrittenQuery dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating RewrittenQuery with all fields."""
        result = RewrittenQuery(
            original_query="What does Taka think?",
            rag_queries=["Taka opinions views", "Taka thoughts ideas"],
            detected_entities=["Taka"],
            resolved_entity_ids=[301411562487545857],
            temporal_filter={"after": "2026-01-10"},
            retrieval_strategy="participant_focused",
            excluded_entities=[],
            resolved_query="What does Takarak think?",
        )

        assert result.original_query == "What does Taka think?"
        assert result.rag_queries == ["Taka opinions views", "Taka thoughts ideas"]
        assert result.detected_entities == ["Taka"]
        assert result.resolved_entity_ids == [301411562487545857]
        assert result.temporal_filter == {"after": "2026-01-10"}
        assert result.retrieval_strategy == "participant_focused"
        assert result.resolved_query == "What does Takarak think?"

    def test_creation_with_defaults(self):
        """Test creating RewrittenQuery with default values."""
        result = RewrittenQuery(
            original_query="Test",
            rag_queries=["test query"],
            detected_entities=[],
        )

        assert result.temporal_filter is None
        assert result.resolved_entity_ids == []
        assert result.retrieval_strategy == "semantic"
        assert result.excluded_entities == []
        assert result.resolved_query is None


class TestXmlFormatting:
    """Tests for XML formatting helpers."""

    def test_format_members_xml(self, sample_members):
        """Test member list XML formatting."""
        xml = format_members_xml(sample_members)

        assert "<members>" in xml
        assert "</members>" in xml
        assert 'id="301411562487545857"' in xml
        assert 'd="Takarak"' in xml
        assert 'u=".takarak"' in xml
        assert 'n="taka"' in xml
        assert 'n="dejv,dejvi,dave"' in xml

    def test_format_members_xml_empty(self):
        """Test empty member list returns empty string."""
        assert format_members_xml(None) == ""
        assert format_members_xml([]) == ""

    def test_format_members_xml_escapes_special_chars(self):
        """Test XML special characters are escaped."""
        members = [
            MemberInfo(
                author_id=123,
                display_name='Test <User> & "Quotes"',
                username="test",
                nicknames=[],
            )
        ]
        xml = format_members_xml(members)

        assert "&lt;" in xml
        assert "&gt;" in xml
        assert "&amp;" in xml
        assert "&quot;" in xml

    def test_format_members_xml_respects_limit(self):
        """Test member limit is respected."""
        members = [
            MemberInfo(author_id=i, display_name=f"User{i}", username=f"user{i}", nicknames=[])
            for i in range(100)
        ]
        xml = format_members_xml(members, max_members=5)

        # Should only have 5 members
        assert xml.count("<m ") == 5

    def test_format_conversation_xml(self, sample_conversation):
        """Test conversation history XML formatting."""
        xml = format_conversation_xml(sample_conversation)

        assert "<conversation>" in xml
        assert "</conversation>" in xml
        assert 'author="Takarak"' in xml
        assert 'id="301411562487545857"' in xml
        assert "AI is going to change everything" in xml

    def test_format_conversation_xml_empty(self):
        """Test empty conversation returns empty string."""
        assert format_conversation_xml(None) == ""
        assert format_conversation_xml([]) == ""

    def test_format_conversation_xml_truncates_content(self):
        """Test long messages are truncated."""
        history = [
            ConversationMessage(
                author="Test",
                author_id=123,
                content="A" * 200,  # 200 chars, should be truncated to 100
            )
        ]
        xml = format_conversation_xml(history)

        # Content should be truncated
        assert "A" * 100 in xml
        assert "A" * 101 not in xml


@pytest.mark.asyncio
async def test_rewrite_basic_query(mock_client):
    """Test basic query rewriting."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["Taka AI opinions"], "detected_entities": ["Taka"], "resolved_entity_ids": [], "temporal_filter": null, "retrieval_strategy": "semantic", "excluded_entities": [], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "What does Taka think about AI?")

    assert result.original_query == "What does Taka think about AI?"
    assert result.rag_queries == ["Taka AI opinions"]
    assert result.detected_entities == ["Taka"]
    assert result.temporal_filter is None


@pytest.mark.asyncio
async def test_rewrite_with_member_context(mock_client, sample_members):
    """Test entity resolution using member list."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["Taka AI opinions"], "detected_entities": ["taka"], "resolved_entity_ids": [301411562487545857], "temporal_filter": null, "retrieval_strategy": "participant_focused", "excluded_entities": [], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "What does taka think?", members=sample_members)

    assert result.resolved_entity_ids == [301411562487545857]
    assert result.retrieval_strategy == "participant_focused"

    # Verify member context was passed to the model
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_prompt = call_kwargs["messages"][0]["content"]
    assert "<members>" in system_prompt
    assert "Takarak" in system_prompt


@pytest.mark.asyncio
async def test_rewrite_resolves_pronouns(mock_client, sample_conversation):
    """Test pronoun resolution from conversation context."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["Takarak AI opinions thoughts"], "detected_entities": ["Takarak"], "resolved_entity_ids": [301411562487545857], "temporal_filter": null, "retrieval_strategy": "participant_focused", "excluded_entities": [], "resolved_query": "What does Takarak think about AI?"}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(
        mock_client, "What does he think?", conversation_history=sample_conversation
    )

    assert result.resolved_query == "What does Takarak think about AI?"
    assert 301411562487545857 in result.resolved_entity_ids

    # Verify conversation context was passed to the model
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_prompt = call_kwargs["messages"][0]["content"]
    assert "<conversation>" in system_prompt
    assert "AI is going to change everything" in system_prompt


@pytest.mark.asyncio
async def test_rewrite_with_excluded_entities(mock_client, sample_members):
    """Test exclusion entity detection."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["AI opinions views"], "detected_entities": [], "resolved_entity_ids": [], "temporal_filter": null, "retrieval_strategy": "semantic", "excluded_entities": ["Taka"], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(
        mock_client,
        "What does everyone except Taka think about AI?",
        members=sample_members,
    )

    assert "Taka" in result.excluded_entities


@pytest.mark.asyncio
async def test_rewrite_with_multiple_queries(mock_client):
    """Test rewriting with multiple query variants."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["politics debate views", "political discussions opinions"], "detected_entities": [], "resolved_entity_ids": [], "temporal_filter": null, "retrieval_strategy": "semantic", "excluded_entities": [], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "What about politics?")

    assert len(result.rag_queries) == 2
    assert "politics debate views" in result.rag_queries
    assert "political discussions opinions" in result.rag_queries


@pytest.mark.asyncio
async def test_rewrite_with_temporal_filter(mock_client):
    """Test query with temporal reference."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["yesterday discussions"], "detected_entities": [], "resolved_entity_ids": [], "temporal_filter": {"after": "2026-01-11", "before": "2026-01-12"}, "retrieval_strategy": "semantic", "excluded_entities": [], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "What happened yesterday?")

    assert result.temporal_filter == {"after": "2026-01-11", "before": "2026-01-12"}


@pytest.mark.asyncio
async def test_rewrite_with_multiple_entities(mock_client):
    """Test detecting multiple entities."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["Taka Endi debate discussion"], "detected_entities": ["Taka", "Endi"], "resolved_entity_ids": [], "temporal_filter": null, "retrieval_strategy": "semantic", "excluded_entities": [], "resolved_query": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "What did Taka and Endi discuss?")

    assert result.detected_entities == ["Taka", "Endi"]


@pytest.mark.asyncio
async def test_rewrite_fallback_on_api_error(mock_client):
    """Test fallback to original query on API error."""
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    result = await rewrite_query(mock_client, "Test question")

    assert result.original_query == "Test question"
    assert result.rag_queries == ["Test question"]
    assert result.detected_entities == []
    assert result.resolved_entity_ids == []
    assert result.temporal_filter is None
    assert result.retrieval_strategy == "semantic"


@pytest.mark.asyncio
async def test_rewrite_fallback_on_empty_rag_queries(mock_client):
    """Test fallback when model returns empty rag_queries."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": [], "detected_entities": [], "resolved_entity_ids": [], "temporal_filter": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "Test question")

    # Should fall back to original query
    assert result.rag_queries == ["Test question"]


@pytest.mark.asyncio
async def test_rewrite_handles_invalid_json(mock_client):
    """Test handling of invalid JSON response."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="not valid json"))])
    )

    result = await rewrite_query(mock_client, "Test question")

    # Should fall back to original query
    assert result.rag_queries == ["Test question"]
    assert result.detected_entities == []
    assert result.resolved_entity_ids == []


@pytest.mark.asyncio
async def test_rewrite_uses_correct_model(mock_client):
    """Test that the correct model is used for rewriting."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["test"], "detected_entities": [], "resolved_entity_ids": [], "temporal_filter": null}'
                    )
                )
            ]
        )
    )

    await rewrite_query(mock_client, "Test question")

    # Verify the model used
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "google/gemini-2.0-flash-lite" in call_kwargs["model"]


@pytest.mark.asyncio
async def test_rewrite_parses_entity_ids_as_integers(mock_client):
    """Test that resolved_entity_ids are parsed as integers."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"rag_queries": ["test"], "detected_entities": ["Taka"], "resolved_entity_ids": ["301411562487545857", 238553996657295361], "temporal_filter": null}'
                    )
                )
            ]
        )
    )

    result = await rewrite_query(mock_client, "Test question")

    # Both string and int should be parsed as int
    assert result.resolved_entity_ids == [301411562487545857, 238553996657295361]
    assert all(isinstance(i, int) for i in result.resolved_entity_ids)
