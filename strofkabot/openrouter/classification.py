"""Query classification for determining search/thinking needs."""

import json
import logging

from openai import AsyncOpenAI

from strofkabot.config import OPENROUTER_ROUTER_MODEL
from strofkabot.openrouter.models import QueryMetadata

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a query analyzer. Analyze the question with its conversation context and output JSON.

CONTEXT ANALYSIS:
- What language is the user speaking?
- Are there images/videos being referenced?

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
2. thinking (bool): Needs multi-step reasoning? (math, logic, "why", "analyze", comparisons)
3. reasoning_effort: How much thinking is needed?
   - "minimal": Simple facts, definitions
   - "low": Straightforward questions
   - "medium": Some analysis needed
   - "high": Complex reasoning, multiple factors
   - "xhigh": Deep analysis, proofs, complex math
4. query_type: Main category
   - "factual": Verifiable facts
   - "creative": Writing, ideas, brainstorming
   - "technical": Code, debugging, how-to
   - "opinion": Subjective advice
   - "comparison": Comparing options
5. key_topics: 2-3 main topics/entities mentioned (array of strings)
6. suggested_response_style:
   - "brief": Quick answer sufficient
   - "detailed": Thorough explanation needed
   - "step-by-step": Process/tutorial format
   - "conversational": Friendly chat style
   - "sarcastic": User is joking or being sarcastic, match their energy
7. language: ISO 639-1 code of user's quesion language (e.g., "en", "sr", "es")
8. requires_citations: true if factual claims need sources

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

    try:
        response = await client.chat.completions.create(
            model=OPENROUTER_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        metadata = QueryMetadata(
            search=result.get("search", False),
            thinking=result.get("thinking", False),
            reasoning_effort=result.get("reasoning_effort", "medium"),
            query_type=result.get("query_type", "factual"),
            key_topics=result.get("key_topics", []),
            suggested_response_style=result.get("suggested_response_style", "conversational"),
            language=result.get("language", "en"),
            requires_citations=result.get("requires_citations", False),
        )

        logger.debug(
            "Router (%s): search=%s, thinking=%s, type=%s, style=%s",
            OPENROUTER_ROUTER_MODEL,
            metadata.search,
            metadata.thinking,
            metadata.query_type,
            metadata.suggested_response_style,
        )

        return metadata

    except Exception as e:
        logger.warning("Router failed (%s), using default metadata", e)
        return get_default_metadata()
