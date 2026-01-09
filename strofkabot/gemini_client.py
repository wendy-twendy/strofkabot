"""Gemini AI client for the !ask command with rate limiting and model rotation."""

import datetime
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from strofkabot.config import (
    GEMINI_MODELS,
    GEMINI_RPD_LIMIT,
    GEMINI_USAGE_FILE,
)

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from Gemini API."""

    text: str
    success: bool
    error_message: str | None = None
    model_used: str | None = None


class GeminiUsageTracker:
    """Track daily usage per model, persist to JSON file."""

    def __init__(self, usage_file: Path = GEMINI_USAGE_FILE):
        self.usage_file = usage_file
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_usage(self) -> dict:
        """Load usage data from JSON file."""
        if not self.usage_file.exists():
            return {"date": "", "models": {}}
        try:
            with open(self.usage_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load usage file, resetting")
            return {"date": "", "models": {}}

    def _save_usage(self, data: dict) -> None:
        """Save usage data to JSON file."""
        try:
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            logger.exception("Failed to save usage file")

    def _get_today_str(self) -> str:
        """Get today's date string in UTC."""
        return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

    def _reset_if_new_day(self, data: dict) -> dict:
        """Reset all counts if date changed."""
        today = self._get_today_str()
        if data.get("date") != today:
            logger.info("New day detected, resetting usage counts")
            return {"date": today, "models": {}}
        return data

    def get_available_model(self) -> str | None:
        """Return first model under RPD limit, or None if all exhausted."""
        data = self._load_usage()
        data = self._reset_if_new_day(data)
        self._save_usage(data)

        for model in GEMINI_MODELS:
            usage = data["models"].get(model, 0)
            if usage < GEMINI_RPD_LIMIT:
                return model

        logger.warning("All models exhausted for today")
        return None

    def increment_usage(self, model: str) -> None:
        """Increment count for model, save to file."""
        data = self._load_usage()
        data = self._reset_if_new_day(data)

        current = data["models"].get(model, 0)
        data["models"][model] = current + 1

        self._save_usage(data)
        logger.debug("Model %s usage: %d/%d", model, current + 1, GEMINI_RPD_LIMIT)

    def get_remaining_requests(self) -> dict[str, int]:
        """Get remaining requests per model for today."""
        data = self._load_usage()
        data = self._reset_if_new_day(data)

        remaining = {}
        for model in GEMINI_MODELS:
            used = data["models"].get(model, 0)
            remaining[model] = max(0, GEMINI_RPD_LIMIT - used)
        return remaining


class GeminiClient:
    """Client for interacting with the Gemini API using generate_content."""

    def __init__(self):
        """Initialize the Gemini client.

        Raises:
            ValueError: If GEMINI_API_KEY environment variable is not set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        os.environ["GOOGLE_API_KEY"] = api_key
        self._client = genai.Client()
        self._usage_tracker = GeminiUsageTracker()

    async def ask_with_context(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
        images: list[dict] | None = None,
    ) -> GeminiResponse:
        """Send a question to Gemini with conversation context and optional images.

        Args:
            question: The user's question.
            system_prompt: Discord-aware system instructions.
            context_messages: List of context message dicts with author, content, etc.
            images: Optional list of image dicts with data (base64) and mime_type.

        Returns:
            GeminiResponse with the model's answer or error details.
        """
        model = self._usage_tracker.get_available_model()
        if model is None:
            return GeminiResponse(
                text="",
                success=False,
                error_message="Daily limit reached (60 requests). Try again tomorrow.",
            )

        try:
            contents = self._build_contents(question, system_prompt, context_messages, images)

            # Configure with Google Search grounding
            config = types.GenerateContentConfig(
                temperature=0.7,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )

            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            self._usage_tracker.increment_usage(model)

            if response.text:
                return GeminiResponse(
                    text=response.text,
                    success=True,
                    model_used=model,
                )
            else:
                return GeminiResponse(
                    text="",
                    success=False,
                    error_message="No text response from model",
                    model_used=model,
                )

        except Exception as e:
            logger.exception("Gemini API error with model %s", model)
            return GeminiResponse(
                text="",
                success=False,
                error_message=str(e),
                model_used=model,
            )

    def _build_contents(
        self,
        question: str,
        system_prompt: str,
        context_messages: list[dict],
        images: list[dict] | None,
    ) -> list:
        """Build the contents list for the Gemini API."""
        parts = []

        # Add system prompt and context as text
        context_text = self._format_context(system_prompt, context_messages)
        parts.append(types.Part.from_text(text=context_text))

        # Add images if present
        if images:
            import base64

            for img in images:
                image_bytes = base64.b64decode(img["data"])
                parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=img["mime_type"],
                    )
                )

        # Add the user's question
        parts.append(types.Part.from_text(text=f"\n\nUser's Question: {question}"))

        return parts

    def _format_context(
        self,
        system_prompt: str,
        context_messages: list[dict],
    ) -> str:
        """Format the system prompt and context messages into a single text block."""
        parts = [system_prompt]

        if context_messages:
            parts.append("\n\n--- Recent Conversation History ---\n")

            for msg in context_messages:
                author = msg.get("author", "Unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")

                reply_info = ""
                if msg.get("reply_to_author"):
                    reply_info = f" (replying to {msg['reply_to_author']})"

                image_count = msg.get("image_count", 0)
                image_info = f" [+{image_count} image(s)]" if image_count > 0 else ""

                me_indicator = " (you)" if msg.get("is_bot") else ""
                parts.append(
                    f"[{timestamp}] {author}{me_indicator}{reply_info}: {content}{image_info}"
                )

            parts.append("\n--- End of History ---")

        return "\n".join(parts)

    def get_remaining_requests(self) -> dict[str, int]:
        """Get remaining requests per model for today."""
        return self._usage_tracker.get_remaining_requests()
