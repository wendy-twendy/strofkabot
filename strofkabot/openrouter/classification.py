"""Query classification for determining search/thinking needs."""

import datetime
import json
import logging

from openai import AsyncOpenAI

from strofkabot.config import OPENROUTER_ROUTER_MODEL
from strofkabot.openrouter.models import QueryMetadata

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a query analyzer for a Discord server called "Strofka".
Analyze the question with its conversation context and output JSON.

CONTEXT ANALYSIS:
- What language is the user speaking?
- Who was mentioned recently in conversation? (for pronoun resolution)
- What topics were discussed? (for "that", "it" resolution)

QUERY CLASSIFICATION:
1. search (bool): Should web search be used? SET TRUE if:
   - Current events, news, weather, prices, sports scores
   - Questions about specific people, companies, products, or places
   - Time-sensitive info: dates, deadlines, "latest", "current", "recent"
   - Technical docs, API references, library versions, release dates
   - Scientific data, statistics, research findings
   - Geographic, demographic, or economic data
   - User explicitly asks to "search", "look up", or "find"
   SET FALSE for: casual chat, jokes, opinions, general knowledge, context-based questions

2. needs_rag (bool): Does this need server history/member info? SET TRUE if:
   - Questions about server members, users, or what someone said
   - "What did [person] say about...", "Has [person] ever..."
   - Server events, inside jokes, past conversations, group dynamics
   - Questions referencing "here", "this server", "our server"
   - Asking about opinions/positions of server members
   SET FALSE for: general knowledge, current events, technical help, creative tasks

3. resolved_query (string|null): If needs_rag=true, rewrite the query with:
   - Pronouns resolved: "he" → actual name from context
   - References resolved: "that" → actual topic from context
   - Keep original intent but make it explicit
   - Example: "What did he say?" → "What did Taka say about the debate?"
   - Set null if needs_rag=false or no pronouns to resolve

4. rag_queries (array|null): If needs_rag=true, provide 1-3 search query variants:
   - Variant 1: Key entities + topic (e.g., "Taka politics opinions")
   - Variant 2: Alternative phrasing (e.g., "Taka debate statements views")
   - Variant 3: Broader context if ambiguous (e.g., "political discussions Taka")
   - For simple queries, 1 variant is enough
   - For ambiguous/vague queries, use 2-3 variants
   - Set null if needs_rag=false

5. detected_entities (array|null): People/nicknames mentioned or referenced:
   - Include explicitly named people
   - Include resolved pronouns ("he" → include who "he" refers to)
   - Used for participant filtering in search
   - Set null if needs_rag=false or no entities detected

6. temporal_filter (object|null): Time constraints if mentioned:
   - "yesterday" → {{"after": "YYYY-MM-DD", "before": "YYYY-MM-DD"}}
   - "last week" → {{"after": "YYYY-MM-DD"}}
   - "in January" → {{"after": "2026-01-01", "before": "2026-02-01"}}
   - null if no temporal reference
   - Today's date: {current_date}

7. thinking (bool): Needs multi-step reasoning? (math, logic, "why", "analyze", comparisons)
8. reasoning_effort: How much thinking is needed?
   - "minimal": Simple facts, definitions
   - "low": Straightforward questions
   - "medium": Some analysis needed
   - "high": Complex reasoning, multiple factors
   - "xhigh": Deep analysis, proofs, complex math
9. query_type: Main category
   - "factual": Verifiable facts
   - "creative": Writing, ideas, brainstorming
   - "technical": Code, debugging, how-to
   - "opinion": Subjective advice
   - "comparison": Comparing options
10. key_topics: 2-3 main topics/entities mentioned (array of strings)
11. suggested_response_style:
   - "brief": Quick answer sufficient
   - "detailed": Thorough explanation needed
   - "step-by-step": Process/tutorial format
   - "conversational": Friendly chat style
   - "sarcastic": User is joking or being sarcastic, match their energy
12. language: ISO 639-1 code of user's question language (e.g., "en", "sq", "es")
13. requires_citations: true if factual claims need sources

Output ONLY valid JSON with all fields."""


def get_default_metadata() -> QueryMetadata:
    """Return default metadata for fallback scenarios."""
    return QueryMetadata(
        search=False,
        thinking=False,
        reasoning_effort="medium",
        query_type="factual",
        key_topics=[],
        suggested_response_style="conversational",
        language="en",
        requires_citations=False,
        needs_rag=False,
        rag_query=None,
        rag_queries=None,
        resolved_query=None,
        detected_entities=None,
        temporal_filter=None,
    )


async def classify_query(
    client: AsyncOpenAI,
    question: str,
    context_messages: list[dict],
) -> QueryMetadata:
    """Use a fast model to classify the query with rich metadata.

    Args:
        client: AsyncOpenAI client instance.
        question: The user's question.
        context_messages: List of context message dicts for better classification.

    Returns:
        QueryMetadata with classification results.
    """
    # Build context summary for the classifier
    context_summary = ""
    if context_messages:
        recent = context_messages[-5:]  # Last 5 messages
        context_lines = []
        for msg in recent:
            author = msg.get("author", "Unknown")
            content = msg.get("content", "")[:100]
            context_lines.append(f"{author}: {content}")
        context_summary = "\n".join(context_lines)

    user_content = f"Question: {question}"
    if context_summary:
        user_content = f"Recent conversation:\n{context_summary}\n\nQuestion: {question}"

    # Inject current date for temporal filter calculations
    current_date = datetime.date.today().isoformat()
    prompt = CLASSIFICATION_PROMPT.replace("{current_date}", current_date)

    try:
        response = await client.chat.completions.create(
            model=OPENROUTER_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # Build rag_queries from new field, falling back to legacy rag_query
        rag_queries = result.get("rag_queries")
        legacy_rag_query = result.get("rag_query")
        if not rag_queries and legacy_rag_query:
            rag_queries = [legacy_rag_query]

        metadata = QueryMetadata(
            search=result.get("search", False),
            thinking=result.get("thinking", False),
            reasoning_effort=result.get("reasoning_effort", "medium"),
            query_type=result.get("query_type", "factual"),
            key_topics=result.get("key_topics", []),
            suggested_response_style=result.get("suggested_response_style", "conversational"),
            language=result.get("language", "en"),
            requires_citations=result.get("requires_citations", False),
            needs_rag=result.get("needs_rag", False),
            rag_query=legacy_rag_query,
            rag_queries=rag_queries,
            resolved_query=result.get("resolved_query"),
            detected_entities=result.get("detected_entities"),
            temporal_filter=result.get("temporal_filter"),
        )

        logger.debug(
            "Router (%s): search=%s, thinking=%s, rag=%s, queries=%d, type=%s, style=%s",
            OPENROUTER_ROUTER_MODEL,
            metadata.search,
            metadata.thinking,
            metadata.needs_rag,
            len(metadata.rag_queries or []),
            metadata.query_type,
            metadata.suggested_response_style,
        )

        return metadata

    except Exception as e:
        logger.warning("Router failed (%s), using default metadata", e)
        return get_default_metadata()
