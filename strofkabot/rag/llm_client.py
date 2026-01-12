"""LLM client for RAG answer generation.

Uses OpenRouter API with Gemini models for generating answers
from retrieved context.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Default model for answer generation
DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"


@dataclass
class LLMResponse:
    """Response from LLM answer generation."""

    text: str
    success: bool
    error_message: str | None = None
    model_used: str | None = None
    tokens_used: int | None = None


class RAGLLMClient:
    """LLM client for RAG answer generation.

    Uses OpenRouter API with OpenAI-compatible interface.

    Args:
        model: Model ID to use for generation.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=60.0,
        )
        self.model = model

    async def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate an answer to a question using retrieved context.

        Args:
            question: The user's question.
            context: Retrieved context from vector store.
            system_prompt: Optional custom system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the generated answer.
        """
        if system_prompt is None:
            system_prompt = self._default_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Based on the following conversation history from a Discord server, answer the question.

<context>
{context}
</context>

<question>
{question}
</question>

Answer based ONLY on what's in the context. If the context doesn't contain enough information, say so. Be concise but complete.""",
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "HTTP-Referer": "https://github.com/strofkabot",
                    "X-Title": "StrofkaBot RAG",
                },
            )

            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            if text:
                return LLMResponse(
                    text=text,
                    success=True,
                    model_used=self.model,
                    tokens_used=tokens,
                )
            else:
                return LLMResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=self.model,
                )

        except Exception as e:
            logger.exception("LLM generation error")
            return LLMResponse(
                text="",
                success=False,
                error_message=str(e),
            )

    async def generate_user_insight(
        self,
        user_name: str,
        user_nicknames: list[str],
        context: str,
        insight_type: str = "summary",
    ) -> LLMResponse:
        """Generate insights about a user based on their messages.

        Args:
            user_name: The user's display name.
            user_nicknames: List of nicknames for the user.
            context: Retrieved context containing the user's messages.
            insight_type: Type of insight ("summary", "topics", "personality").

        Returns:
            LLMResponse with the generated insight.
        """
        prompts = {
            "summary": f"""Analyze these Discord messages from {user_name} (also known as: {', '.join(user_nicknames)}).
Write a brief, fun summary of this person based on their messages. Include:
- What topics they discuss most
- Their communication style
- Any notable opinions or recurring themes
Keep it light and friendly, like you're describing a friend to someone.""",
            "topics": f"""Based on these messages from {user_name}, list the main topics they discuss.
Format as a bullet list with brief descriptions.""",
            "personality": f"""Based on these Discord messages, describe {user_name}'s personality and communication style.
Be specific with examples from their messages.""",
        }

        system_prompt = """You are analyzing Discord messages to provide insights about users.
Be observant but respectful. Focus on public behavior in group chats.
Use a casual, friendly tone. Include specific examples when possible."""

        prompt = prompts.get(insight_type, prompts["summary"])

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""{prompt}

<messages>
{context}
</messages>""",
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=1000,
                extra_body={
                    "HTTP-Referer": "https://github.com/strofkabot",
                    "X-Title": "StrofkaBot RAG",
                },
            )

            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            if text:
                return LLMResponse(
                    text=text,
                    success=True,
                    model_used=self.model,
                    tokens_used=tokens,
                )
            else:
                return LLMResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=self.model,
                )

        except Exception as e:
            logger.exception("LLM insight generation error")
            return LLMResponse(
                text="",
                success=False,
                error_message=str(e),
            )

    async def generate_recap(
        self,
        timeframe: str,
        channel_name: str | None,
        context: str,
    ) -> LLMResponse:
        """Generate a channel activity recap/summary.

        Args:
            timeframe: The time period (e.g., "24h", "7d", "month").
            channel_name: The channel name, or None for all channels.
            context: Retrieved context containing recent messages.

        Returns:
            LLMResponse with the generated recap.
        """
        channel_ref = f"#{channel_name}" if channel_name else "the server"

        system_prompt = """You are summarizing Discord channel activity for a recap.
The server is Albanian, so messages are often in Albanian or a mix of Albanian and English.
Be observant and highlight interesting moments. Use a casual, engaging tone."""

        prompt = f"""Summarize the activity in {channel_ref} over the last {timeframe}.

<messages>
{context}
</messages>

Generate a recap (3-4 paragraphs) covering:
1. Main topics and discussions that came up
2. Most active participants and their notable contributions
3. Any interesting debates, jokes, or memorable moments
4. Overall vibe/sentiment of the conversations

Use a friendly, conversational tone like you're briefing a friend who missed the chat.
If there's Albanian content, you can mix Albanian and English naturally."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=1500,
                extra_body={
                    "HTTP-Referer": "https://github.com/strofkabot",
                    "X-Title": "StrofkaBot RAG",
                },
            )

            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            if text:
                return LLMResponse(
                    text=text,
                    success=True,
                    model_used=self.model,
                    tokens_used=tokens,
                )
            else:
                return LLMResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=self.model,
                )

        except Exception as e:
            logger.exception("LLM recap generation error")
            return LLMResponse(
                text="",
                success=False,
                error_message=str(e),
            )

    def _default_system_prompt(self) -> str:
        """Default system prompt for RAG answer generation."""
        return """You are a helpful assistant answering questions about conversations from a Discord server called "Strofka".
The server is Albanian, so messages are often in Albanian or a mix of Albanian and English.
Users have nicknames that are referenced in the context.

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't have the answer, say "I couldn't find information about that in the conversation history"
- Be concise and direct
- Reference specific messages or users when relevant
- Maintain the casual, friendly tone of Discord chat
- For Albanian content, you can respond in Albanian or English based on the question language"""
