"""LLM-based metadata extraction for conversation chunks.

Uses OpenRouter API with google/gemini-2.0-flash-lite-001 to extract:
- Topics discussed
- Sentiment (positive/negative/neutral/mixed)
- Conversation type (discussion/banter/question_answer/announcement/debate)
- Users mentioned (not speakers)
- Summary
- Key phrases for search
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from .schema import ExtractedMetadata

logger = logging.getLogger(__name__)

# Extraction prompt template
EXTRACTION_PROMPT = """Analyze this Discord conversation and extract metadata.

Participants (speakers): {participants}

Conversation:
{chunk_text}

Extract as JSON (no markdown, just raw JSON):
{{
    "mentioned_users": ["nickname or name of anyone talked ABOUT (not the speakers)"],
    "topics": ["main topics discussed, max 3"],
    "sentiment": "positive|negative|neutral|mixed",
    "conversation_type": "discussion|banter|question_answer|announcement|debate",
    "summary": "1-2 sentence summary of what was discussed",
    "key_phrases": ["3-5 important phrases for search"]
}}

Rules:
- mentioned_users: Only include people TALKED ABOUT, not the speakers themselves
- topics: Be specific (e.g., "football match" not just "sports")
- sentiment: Consider the overall emotional tone
- conversation_type: Choose the best fit
- summary: Brief, informative, third-person
- key_phrases: Important searchable terms from the conversation

Respond with ONLY the JSON object, no explanation."""


@dataclass
class ExtractionResponse:
    """Response from metadata extraction."""

    metadata: ExtractedMetadata | None
    success: bool
    error_message: str | None = None


class MetadataExtractor:
    """Extracts metadata from conversation chunks using LLM.

    Uses OpenRouter API with google/gemini-2.0-flash-lite-001 for fast,
    cheap extraction of topics, sentiment, and other metadata.
    """

    model = "google/gemini-2.0-flash-lite-001"

    def __init__(self, api_key: str | None = None):
        """Initialize the metadata extractor.

        Args:
            api_key: OpenRouter API key. If not provided, reads from
                     OPENROUTER_API_KEY environment variable.

        Raises:
            ValueError: If no API key provided or found in environment.
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set " "and no api_key was provided"
            )

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=60.0,
        )

    async def extract(self, chunk_text: str, participants: list[str]) -> ExtractionResponse:
        """Extract metadata from a conversation chunk.

        Args:
            chunk_text: The formatted conversation text.
            participants: List of participant names in the conversation.

        Returns:
            ExtractionResponse with extracted metadata or error.
        """
        try:
            prompt = EXTRACTION_PROMPT.format(
                participants=", ".join(participants),
                chunk_text=chunk_text,
            )

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON
            try:
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    # Extract JSON from code block
                    lines = content.split("\n")
                    json_lines = []
                    in_block = False
                    for line in lines:
                        if line.startswith("```"):
                            in_block = not in_block
                            continue
                        if in_block or not line.startswith("```"):
                            json_lines.append(line)
                    content = "\n".join(json_lines)

                data = json.loads(content)
                metadata = ExtractedMetadata.from_llm_response(data)

                return ExtractionResponse(
                    metadata=metadata,
                    success=True,
                )

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse LLM JSON response: %s", e)
                return ExtractionResponse(
                    metadata=None,
                    success=False,
                    error_message=f"JSON parse error: {e}",
                )

        except Exception as e:
            logger.exception("Metadata extraction API error")
            return ExtractionResponse(
                metadata=None,
                success=False,
                error_message=str(e),
            )

    async def extract_batch(
        self,
        chunks: list[tuple[str, list[str]]],
        max_concurrent: int = 5,
    ) -> list[ExtractionResponse]:
        """Extract metadata for multiple chunks concurrently.

        Args:
            chunks: List of (chunk_text, participants) tuples.
            max_concurrent: Maximum concurrent API calls.

        Returns:
            List of ExtractionResponse objects in same order as input.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(
            chunk_text: str, participants: list[str]
        ) -> ExtractionResponse:
            async with semaphore:
                return await self.extract(chunk_text, participants)

        tasks = [extract_with_semaphore(text, participants) for text, participants in chunks]

        return await asyncio.gather(*tasks)
