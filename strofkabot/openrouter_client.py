"""OpenRouter AI client for the !ask command with auto search/thinking detection."""

import json
import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from strofkabot.config import (
    OPENROUTER_INFERENCE_MODEL,
    OPENROUTER_ROUTER_MODEL,
    OPENROUTER_VISION_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class QueryMetadata:
    """Rich metadata from query classification."""

    search: bool
    thinking: bool
    reasoning_effort: str  # minimal, low, medium, high, xhigh
    query_type: str  # factual, creative, technical, opinion, comparison
    key_topics: list[str]
    suggested_response_style: str  # brief, detailed, step-by-step, conversational, sarcastic
    language: str  # ISO 639-1 code
    requires_citations: bool


@dataclass
class OpenRouterResponse:
    """Structured response from OpenRouter API."""

    text: str
    success: bool
    error_message: str | None = None
    model_used: str | None = None
    search_used: bool = False
    thinking_used: bool = False
    metadata: QueryMetadata | None = None


class OpenRouterClient:
    """Client for interacting with OpenRouter API with auto search/thinking detection."""

    def __init__(self):
        """Initialize the OpenRouter client.

        Raises:
            ValueError: If OPENROUTER_API_KEY environment variable is not set.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60.0,  # 60 second timeout to prevent hanging
        )

    async def classify_query(
        self,
        question: str,
        context_messages: list[dict] | None = None,
    ) -> QueryMetadata:
        """Classify a query to get rich metadata for the main model.

        Args:
            question: The user's question.
            context_messages: Optional list of context message dicts.

        Returns:
            QueryMetadata with classification results.
        """
        return await self._classify_query(question, context_messages or [])

    async def ask_with_context(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
        images: list[dict] | None = None,
        query_metadata: QueryMetadata | None = None,
        url_context: str | None = None,
    ) -> OpenRouterResponse:
        """Send a question to OpenRouter with conversation context.

        Args:
            question: The user's question.
            system_prompt: Discord-aware system instructions.
            context_messages: List of context message dicts with author, content, etc.
            images: Optional list of image dicts with data (base64) and mime_type.
            query_metadata: Optional pre-computed metadata (skips classification).
            url_context: Optional extracted URL content to include in context.

        Returns:
            OpenRouterResponse with the model's answer or error details.
        """
        try:
            # Use vision model for images, otherwise use classification
            if images:
                model = OPENROUTER_VISION_MODEL
                metadata = query_metadata or QueryMetadata(
                    search=False,
                    thinking=False,
                    reasoning_effort="medium",
                    query_type="factual",
                    key_topics=[],
                    suggested_response_style="conversational",
                    language="en",
                    requires_citations=False,
                )
            else:
                metadata = query_metadata or await self._classify_query(question, context_messages)
                model = OPENROUTER_INFERENCE_MODEL
                if metadata.search:
                    model = f"{model}:online"

            # Build messages in OpenAI format
            messages = self._build_messages(
                question, system_prompt, context_messages, images, url_context
            )

            # Build extra_body for reasoning
            extra_body = {
                "HTTP-Referer": "https://github.com/strofkabot",
                "X-Title": "StrofkaBot",
            }
            if metadata.thinking:
                extra_body["reasoning"] = {"effort": metadata.reasoning_effort}

            logger.info(
                "OpenRouter request: model=%s, search=%s, thinking=%s, effort=%s, type=%s",
                model,
                metadata.search,
                metadata.thinking,
                metadata.reasoning_effort,
                metadata.query_type,
            )

            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
                extra_body=extra_body,
            )

            text = response.choices[0].message.content
            if text:
                return OpenRouterResponse(
                    text=text,
                    success=True,
                    model_used=model,
                    search_used=metadata.search,
                    thinking_used=metadata.thinking,
                    metadata=metadata,
                )
            else:
                return OpenRouterResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=model,
                    metadata=metadata,
                )

        except Exception as e:
            logger.exception("OpenRouter API error")
            return OpenRouterResponse(
                text="",
                success=False,
                error_message=str(e),
            )

    async def _classify_query(
        self,
        question: str,
        context_messages: list[dict],
    ) -> QueryMetadata:
        """Use a fast model to classify the query with rich metadata.

        Args:
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

        classification_prompt = """You are a query analyzer. Analyze the question with its conversation context and output JSON.

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

        user_content = f"Question: {question}"
        if context_summary:
            user_content = f"Recent conversation:\n{context_summary}\n\nQuestion: {question}"

        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": classification_prompt},
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

    def _build_messages(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
        images: list[dict] | None = None,
        url_context: str | None = None,
    ) -> list[dict]:
        """Build the messages list for the OpenAI-compatible API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add context if provided
        if context_messages or url_context:
            context_parts = []
            if context_messages:
                context_parts.append(self._format_context(context_messages))
            if url_context:
                context_parts.append(url_context)
            context_text = "\n\n".join(context_parts)
            messages.append({"role": "user", "content": context_text})
            messages.append(
                {
                    "role": "assistant",
                    "content": "I can see the conversation history. How can I help?",
                }
            )

        # Add the actual question (with images if present)
        if images:
            # Multimodal format: text first, then images
            content = [{"type": "text", "text": question}]
            for img in images:
                data_url = f"data:{img['mime_type']};base64,{img['data']}"
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def _format_context(self, context_messages: list[dict]) -> str:
        """Format context messages into structured XML for the prompt."""
        parts = ["<conversation>"]

        for msg in context_messages:
            author = msg.get("author", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            reply_to = msg.get("reply_to_author")
            image_count = msg.get("image_count", 0)

            # Build message attributes
            attrs = [f'author="{author}"', f'time="{timestamp}"']
            if msg.get("is_bot"):
                attrs.append('is_me="true"')
            if reply_to:
                attrs.append(f'replying_to="{reply_to}"')
            if image_count > 0:
                attrs.append(f'images="{image_count}"')

            parts.append(f"  <message {' '.join(attrs)}>")
            parts.append(f"    {content}")
            parts.append("  </message>")

        parts.append("</conversation>")
        return "\n".join(parts)
