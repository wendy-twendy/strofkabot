"""Query-focused extraction for RAG context.

Uses a lightweight model to extract only the relevant information
from retrieved chunks for a specific question.
"""

import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from strofkabot.config import RAG_EXTRACTION_MODEL
from strofkabot.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are extracting relevant information from Discord server history to answer a question.

TASK: Extract ONLY information relevant to answering the question.

GUIDELINES:
- Extract direct quotes when someone's opinion/statement is asked about
- Include speaker names and approximate dates for attribution
- Summarize discussions if asking about a topic
- If context doesn't contain relevant info, say so clearly
- Keep extraction focused and concise (max 500 tokens)
- Use same language as the question

FORMAT:
- Bullet points for multiple pieces of information
- Quote format: "[Name] (date): 'quoted text'"
- Be factual, don't add interpretation"""


@dataclass
class ExtractionResult:
    """Result from query-focused extraction."""

    text: str
    success: bool
    error_message: str | None = None
    sources_used: int = 0


async def extract_relevant_context(
    client: AsyncOpenAI,
    question: str,
    search_results: list[SearchResult],
    max_tokens: int = 500,
) -> ExtractionResult:
    """Extract query-relevant information from search results.

    Uses a lightweight model (Gemini 2.0 Flash Lite) to create a focused
    extraction containing only the information relevant to the question.

    Args:
        client: AsyncOpenAI client configured for OpenRouter.
        question: The user's original question.
        search_results: Retrieved chunks from semantic search.
        max_tokens: Maximum tokens for the extraction (default 500).

    Returns:
        ExtractionResult with focused context or error.
    """
    if not search_results:
        return ExtractionResult(
            text="",
            success=True,
            error_message="No search results to extract from",
            sources_used=0,
        )

    # Build context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        meta = result.metadata
        channel = meta.get("channel_name", "?")
        year = meta.get("year", "")
        month = meta.get("month", "")

        header = f"[Source {i}: #{channel}"
        if year:
            if isinstance(month, int):
                header += f", {year}-{month:02d}"
            else:
                header += f", {year}"
        header += "]"
        context_parts.append(f"{header}\n{result.document}")

    context = "\n\n---\n\n".join(context_parts)

    try:
        response = await client.chat.completions.create(
            model=RAG_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"<question>\n{question}\n</question>\n\n<context>\n{context}\n</context>\n\nExtract relevant information.",
                },
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            extra_body={
                "HTTP-Referer": "https://github.com/strofkabot",
                "X-Title": "StrofkaBot RAG",
            },
        )

        text = response.choices[0].message.content
        if text:
            return ExtractionResult(
                text=text.strip(),
                success=True,
                sources_used=len(search_results),
            )
        return ExtractionResult(
            text="",
            success=False,
            error_message="Empty response from extraction model",
            sources_used=len(search_results),
        )

    except Exception as e:
        logger.exception("RAG extraction error")
        return ExtractionResult(
            text="",
            success=False,
            error_message=str(e),
            sources_used=len(search_results),
        )
