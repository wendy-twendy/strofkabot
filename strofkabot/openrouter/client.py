"""OpenRouter AI client for the !ask command with auto search/thinking detection."""

import logging
import os

from openai import AsyncOpenAI

from strofkabot.config import (
    OPENROUTER_INFERENCE_MODEL,
    OPENROUTER_VISION_MODEL,
)
from strofkabot.openrouter.classification import classify_query as _classify_query
from strofkabot.openrouter.classification import get_default_metadata
from strofkabot.openrouter.memory_extraction import (
    extract_memories as _extract_memories,
)
from strofkabot.openrouter.memory_extraction import (
    filter_relevant_memories as _filter_relevant_memories,
)
from strofkabot.openrouter.models import OpenRouterResponse, QueryMetadata

logger = logging.getLogger(__name__)


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
        return await _classify_query(self._client, question, context_messages or [])

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
                metadata = query_metadata or get_default_metadata()
            else:
                metadata = query_metadata or await _classify_query(
                    self._client, question, context_messages
                )
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

    async def filter_relevant_memories(
        self,
        question: str,
        memories: list,
    ) -> list[int]:
        """Filter memories to only those relevant to the current question.

        Args:
            question: The user's question.
            memories: List of Memory objects to filter.

        Returns:
            List of indices of relevant memories (0-indexed).
        """
        return await _filter_relevant_memories(self._client, question, memories)

    async def extract_memories(
        self,
        context_messages: list[dict],
        question: str,
        response_text: str,
        user_id: int,
        user_name: str,
        known_users: dict[str, int] | None = None,
        existing_user_memories: list | None = None,
        existing_server_memories: list | None = None,
    ) -> dict:
        """Extract new memories worth saving from a conversation using tool calling.

        Args:
            context_messages: List of context message dicts.
            question: The user's question.
            response_text: The AI's response (not used in extraction anymore).
            user_id: Discord ID of the user who asked.
            user_name: Display name of the user who asked.
            known_users: Mapping of display_name -> user_id for users in context.
            existing_user_memories: List of existing Memory objects for the user.
            existing_server_memories: List of existing server Memory objects.

        Returns:
            Dict with memory operations.
        """
        return await _extract_memories(
            self._client,
            context_messages,
            question,
            response_text,
            user_id,
            user_name,
            known_users,
            existing_user_memories,
            existing_server_memories,
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
