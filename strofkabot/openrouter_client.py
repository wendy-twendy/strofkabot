"""OpenRouter AI client for the !ask command with auto search/thinking detection."""

import json
import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from strofkabot.config import (
    OPENROUTER_INFERENCE_MODEL,
    OPENROUTER_ROUTER_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterResponse:
    """Structured response from OpenRouter API."""

    text: str
    success: bool
    error_message: str | None = None
    model_used: str | None = None
    search_used: bool = False
    thinking_used: bool = False


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
        )

    async def ask_with_context(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
        images: list[dict] | None = None,
    ) -> OpenRouterResponse:
        """Send a question to OpenRouter with conversation context.

        Args:
            question: The user's question.
            system_prompt: Discord-aware system instructions.
            context_messages: List of context message dicts with author, content, etc.
            images: Optional list of image dicts (NOT SUPPORTED - will return error).

        Returns:
            OpenRouterResponse with the model's answer or error details.
        """
        # Images not supported via OpenRouter yet
        if images:
            return OpenRouterResponse(
                text="",
                success=False,
                error_message="images_not_supported",
            )

        try:
            # Auto-detect if search/thinking are needed
            needs_search, needs_thinking = await self._detect_query_requirements(question)

            # Build messages in OpenAI format
            messages = self._build_messages(question, system_prompt, context_messages)

            # Determine model (add :online suffix for search)
            model = OPENROUTER_INFERENCE_MODEL
            if needs_search:
                model = f"{model}:online"

            # Build extra_body for reasoning
            extra_body = {
                "HTTP-Referer": "https://github.com/strofkabot",
                "X-Title": "StrofkaBot",
            }
            if needs_thinking:
                extra_body["reasoning"] = {"effort": "high"}

            logger.info(
                "OpenRouter request: model=%s, search=%s, thinking=%s",
                model,
                needs_search,
                needs_thinking,
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
                    search_used=needs_search,
                    thinking_used=needs_thinking,
                )
            else:
                return OpenRouterResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=model,
                )

        except Exception as e:
            logger.exception("OpenRouter API error")
            return OpenRouterResponse(
                text="",
                success=False,
                error_message=str(e),
            )

    async def _detect_query_requirements(
        self,
        question: str,
    ) -> tuple[bool, bool]:
        """Use a fast model to detect if search and/or thinking are needed.

        Returns:
            Tuple of (needs_search, needs_thinking).
        """
        classification_prompt = """You are a query classifier. Analyze the question and output JSON with two boolean fields.

SEARCH (true/false): Does this need CURRENT or RECENT information?
- true: news, current events, prices, weather, "latest", "today", "now", "recent", "this week"
- false: general knowledge, math, logic, history, programming, static facts

THINKING (true/false): Does this need COMPLEX REASONING or analysis?
- true: logic puzzles, multi-step problems, "why", "implications", "analyze", comparisons
- false: simple facts, definitions, "what is X", straightforward answers

Output ONLY valid JSON: {"search": true/false, "thinking": true/false}"""

        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": classification_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                max_tokens=50,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            needs_search = result.get("search", False)
            needs_thinking = result.get("thinking", False)

            logger.debug(
                "Router (%s): search=%s, thinking=%s",
                OPENROUTER_ROUTER_MODEL,
                needs_search,
                needs_thinking,
            )

            return needs_search, needs_thinking

        except Exception as e:
            logger.warning("Router failed (%s), defaulting to no search/thinking", e)
            return False, False

    def _build_messages(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
    ) -> list[dict]:
        """Build the messages list for the OpenAI-compatible API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add context if provided
        if context_messages:
            context_text = self._format_context(context_messages)
            messages.append({"role": "user", "content": context_text})
            messages.append(
                {
                    "role": "assistant",
                    "content": "I can see the conversation history. How can I help?",
                }
            )

        # Add the actual question
        messages.append({"role": "user", "content": question})

        return messages

    def _format_context(self, context_messages: list[dict]) -> str:
        """Format context messages into a single text block for the prompt."""
        parts = ["Recent conversation history:"]

        for msg in context_messages:
            author = msg.get("author", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            reply_info = ""
            if msg.get("reply_to_author"):
                reply_info = f" (replying to {msg['reply_to_author']})"

            image_count = msg.get("image_count", 0)
            image_info = f" [+{image_count} image(s)]" if image_count > 0 else ""

            parts.append(f"[{timestamp}] {author}{reply_info}: {content}{image_info}")

        return "\n".join(parts)
