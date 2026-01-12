"""Query rewriting for RAG retrieval.

Generates multiple query variants, parses temporal references,
detects entities, and resolves pronouns using conversation context.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from strofkabot.config import RAG_EXTRACTION_MODEL

logger = logging.getLogger(__name__)


@dataclass
class MemberInfo:
    """Compact member representation for the rewriter."""

    author_id: int
    display_name: str
    username: str
    nicknames: list[str]


@dataclass
class ConversationMessage:
    """Minimal message representation for conversation context."""

    author: str
    author_id: int
    content: str


@dataclass
class RewrittenQuery:
    """Result of query rewriting."""

    original_query: str
    rag_queries: list[str]
    detected_entities: list[str]
    resolved_entity_ids: list[int] = field(default_factory=list)
    temporal_filter: dict | None = None
    retrieval_strategy: str = "semantic"
    excluded_entities: list[str] = field(default_factory=list)
    resolved_query: str | None = None


REWRITER_PROMPT = """You are a query analyzer for Discord server history search.
Analyze the question and output JSON to improve semantic search.

Today's date: {current_date}
{member_context}
{conversation_context}
OUTPUT FIELDS:
1. rag_queries (array): 1-3 search query variants for semantic search:
   - Extract key entities + topic keywords
   - Include alternative phrasings
   - Simple queries need 1 variant, ambiguous queries need 2-3
   - Example: "What does Taka think about AI?" -> ["Taka AI opinions views", "Taka artificial intelligence thoughts"]

2. detected_entities (array): People/nicknames mentioned or resolved from pronouns:
   - Include names directly in the question
   - Resolve pronouns to who is being DISCUSSED, not just who spoke last
   - Example: If A says "X is great" and B says "He's right", then "he" refers to X (who B is talking about), not B
   - Match against member nicknames when possible

3. resolved_entity_ids (array): Numeric author IDs for detected entities:
   - Look up detected names/nicknames in the member list
   - Return the matching author_id values
   - Empty array if no matches found

4. temporal_filter (object|null): Time constraints if mentioned:
   - "yesterday" -> {{"after": "YYYY-MM-DD", "before": "YYYY-MM-DD"}}
   - "last week" -> {{"after": "YYYY-MM-DD"}}
   - "in January" -> {{"after": "2026-01-01", "before": "2026-02-01"}}
   - null if no temporal reference

5. retrieval_strategy (string): How to search:
   - "participant_focused": Question about specific person(s)
   - "semantic": General topic search
   - "keyword": Specific phrase/quote search

6. excluded_entities (array): People explicitly excluded:
   - For queries like "everyone except X" or "not including Y"
   - Empty array if no exclusions

7. resolved_query (string|null): Query with pronouns resolved:
   - Rewrite "he/she/they" with the name of who is being DISCUSSED (not just last speaker)
   - Rewrite "that/it" with the actual topic being discussed
   - Example: After "Taka: AI is cool" + "Dave: He's optimistic", "What did he say?" -> "What did Taka say?"
   - null if no pronouns to resolve

Output ONLY valid JSON."""


def format_members_xml(members: list[MemberInfo] | None, max_members: int = 60) -> str:
    """Format member list as compact XML for the prompt.

    Args:
        members: List of MemberInfo objects.
        max_members: Maximum members to include.

    Returns:
        XML string or empty string if no members.
    """
    if not members:
        return ""

    limited = members[:max_members]
    lines = ["<members>"]
    for m in limited:
        # Escape XML special chars in display names
        display = (
            m.display_name.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        username = (
            m.username.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        nicks = ",".join(m.nicknames) if m.nicknames else ""
        attrs = f'id="{m.author_id}" d="{display}" u="{username}"'
        if nicks:
            attrs += f' n="{nicks}"'
        lines.append(f"  <m {attrs}/>")
    lines.append("</members>")
    return "\n".join(lines)


def format_conversation_xml(
    history: list[ConversationMessage] | None,
    max_messages: int = 5,
) -> str:
    """Format conversation history as XML.

    Args:
        history: List of ConversationMessage objects.
        max_messages: Maximum messages to include.

    Returns:
        XML string or empty string if no history.
    """
    if not history:
        return ""

    recent = history[-max_messages:]
    lines = ["<conversation>"]
    for msg in recent:
        # Truncate and escape content
        content = msg.content[:100]
        content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        author = (
            msg.author.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        lines.append(f'  <msg author="{author}" id="{msg.author_id}">{content}</msg>')
    lines.append("</conversation>")
    return "\n".join(lines)


async def rewrite_query(
    client: AsyncOpenAI,
    question: str,
    members: list[MemberInfo] | None = None,
    conversation_history: list[ConversationMessage] | None = None,
) -> RewrittenQuery:
    """Rewrite a query for better RAG retrieval.

    Uses a lightweight model to generate multiple query variants,
    detect entities, resolve pronouns, and parse temporal references.

    Args:
        client: AsyncOpenAI client configured for OpenRouter.
        question: The user's question.
        members: Optional list of server members for entity resolution.
        conversation_history: Optional recent messages for pronoun resolution.

    Returns:
        RewrittenQuery with enhanced search parameters.
    """
    current_date = datetime.date.today().isoformat()

    # Build context sections
    member_context = format_members_xml(members)
    conversation_context = format_conversation_xml(conversation_history)

    # Build prompt with context
    prompt = REWRITER_PROMPT.replace("{current_date}", current_date)
    prompt = prompt.replace("{member_context}", member_context + "\n" if member_context else "")
    prompt = prompt.replace(
        "{conversation_context}", conversation_context + "\n" if conversation_context else ""
    )

    try:
        response = await client.chat.completions.create(
            model=RAG_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
            extra_body={
                "HTTP-Referer": "https://github.com/strofkabot",
                "X-Title": "StrofkaBot RAG",
            },
        )

        result = json.loads(response.choices[0].message.content)

        rag_queries = result.get("rag_queries", [])
        if not rag_queries:
            rag_queries = [question]

        # Parse resolved_entity_ids as integers
        raw_ids = result.get("resolved_entity_ids", [])
        resolved_ids = []
        for rid in raw_ids:
            try:
                resolved_ids.append(int(rid))
            except (ValueError, TypeError):
                pass

        return RewrittenQuery(
            original_query=question,
            rag_queries=rag_queries,
            detected_entities=result.get("detected_entities", []),
            resolved_entity_ids=resolved_ids,
            temporal_filter=result.get("temporal_filter"),
            retrieval_strategy=result.get("retrieval_strategy", "semantic"),
            excluded_entities=result.get("excluded_entities", []),
            resolved_query=result.get("resolved_query"),
        )

    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}, using original query")
        return RewrittenQuery(
            original_query=question,
            rag_queries=[question],
            detected_entities=[],
            resolved_entity_ids=[],
            temporal_filter=None,
            retrieval_strategy="semantic",
            excluded_entities=[],
            resolved_query=None,
        )
