"""Helper functions for the !ask command."""

from __future__ import annotations

import base64
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import discord
from PIL import Image

from strofkabot.image_processor import IMAGE_EXTENSIONS
from strofkabot.utils.discord_helpers import get_reply_info
from strofkabot.utils.nickname_loader import get_display_name

if TYPE_CHECKING:
    from strofkabot.openrouter_client import QueryMetadata

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 4 * 1024 * 1024  # 4MB
MAX_IMAGE_DIMENSION = 2048


async def fetch_context_messages(
    channel: discord.TextChannel,
    exclude_message_id: int,
    limit: int = 10,
) -> list[discord.Message]:
    """Fetch recent messages from a channel, excluding the command message.

    Args:
        channel: The Discord channel to fetch from.
        exclude_message_id: Message ID to exclude (the command itself).
        limit: Maximum number of messages to fetch.

    Returns:
        List of Discord messages, oldest first.
    """
    messages = []
    async for msg in channel.history(limit=limit + 1):
        if msg.id != exclude_message_id:
            messages.append(msg)
        if len(messages) >= limit:
            break

    return list(reversed(messages))


async def prepare_context(
    messages: list[discord.Message],
    nicknames: dict[int, list[str]] | None = None,
) -> list[dict]:
    """Prepare context messages for the AI (text only).

    Args:
        messages: List of Discord messages.
        nicknames: Optional dict mapping user IDs to nickname lists.

    Returns:
        List of formatted message context dicts.
    """
    context = []

    for msg in messages:
        reply_to_id, reply_to_author, reply_to_content, reply_to_author_id = get_reply_info(msg)

        # Use nickname if available for reply-to author
        if reply_to_author and reply_to_author_id:
            reply_to_author = get_display_name(reply_to_author_id, reply_to_author, nicknames)

        image_attachments = [att for att in msg.attachments if is_image_attachment(att.filename)]

        context_entry = {
            "author": get_display_name(msg.author.id, msg.author.display_name, nicknames),
            "content": msg.content or "[no text]",
            "timestamp": msg.created_at.strftime("%H:%M"),
            "reply_to_author": reply_to_author,
            "reply_to_content": reply_to_content[:100] if reply_to_content else None,
            "image_count": len(image_attachments),
        }
        context.append(context_entry)

    return context


async def extract_images_from_messages(
    messages: list[discord.Message],
) -> list[dict]:
    """Extract and encode images from Discord messages.

    Args:
        messages: List of Discord messages to extract images from.

    Returns:
        List of {data: base64, mime_type: str} for images.
    """
    images = []
    for msg in messages:
        image_attachments = [att for att in msg.attachments if is_image_attachment(att.filename)]
        for att in image_attachments[:3]:  # Max 3 per message
            image_data = await download_and_encode_image(att.url, att.filename)
            if image_data:
                images.append(image_data)
    return images


def is_image_attachment(filename: str) -> bool:
    """Check if a filename is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


async def download_and_encode_image(url: str, filename: str) -> dict | None:
    """Download an image and encode it for the Gemini API.

    Args:
        url: URL to download the image from.
        filename: Original filename (for MIME type detection).

    Returns:
        Dict with {data: base64_string, mime_type: str} or None if failed.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning("Failed to download image %s: HTTP %d", url, response.status)
                    return None

                image_bytes = await response.read()

                if len(image_bytes) > MAX_IMAGE_SIZE_BYTES * 2:
                    logger.warning("Image too large, skipping: %d bytes", len(image_bytes))
                    return None

        image = Image.open(io.BytesIO(image_bytes))

        if max(image.size) > MAX_IMAGE_DIMENSION:
            ratio = MAX_IMAGE_DIMENSION / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "data": encoded,
            "mime_type": "image/jpeg",
        }

    except Exception:
        logger.exception("Failed to process image: %s", url)
        return None


def build_system_prompt(
    guild_name: str,
    channel_name: str,
    user_name: str,
    query_metadata: QueryMetadata | None = None,
) -> str:
    """Build a dynamic, context-aware system prompt.

    Args:
        guild_name: Name of the Discord server.
        channel_name: Name of the current channel.
        user_name: Display name of the user asking.
        query_metadata: Optional metadata from query classification.

    Returns:
        Formatted system prompt string.
    """
    # Get current date/time
    now = datetime.now(UTC)

    # Build base prompt with context
    base = f"""You are Llumi, a helpful and knowledgeable assistant in the "{guild_name}" Discord server.

CURRENT CONTEXT:
- Server: {guild_name}
- Channel: #{channel_name}
- User asking: {user_name}
- Current date: {now.strftime('%A, %B %d, %Y')}
- Current time: {now.strftime('%H:%M UTC')}"""

    prompt_parts = [base]

    # Add dynamic instructions based on query metadata
    if query_metadata:
        # Query-type specific instructions
        type_instructions = {
            "technical": """TECHNICAL QUERY DETECTED:
- Provide code examples with proper syntax highlighting (use Discord code blocks)
- Explain technical concepts clearly with real-world analogies
- Include potential pitfalls and best practices
- Reference documentation when relevant""",
            "creative": """CREATIVE QUERY DETECTED:
- Be imaginative and engaging
- Offer multiple ideas or approaches when appropriate
- Encourage experimentation and exploration
- Be supportive of creative endeavors""",
            "factual": """FACTUAL QUERY DETECTED:
- Be precise and accurate
- Cite sources when available from web search
- Clearly distinguish between facts and opinions
- Acknowledge uncertainty when appropriate""",
            "opinion": """OPINION/ADVICE QUERY DETECTED:
- Provide balanced perspectives
- Note when giving subjective advice
- Consider the user's specific context
- Respect that they may have different preferences""",
            "comparison": """COMPARISON QUERY DETECTED:
- Use structured comparisons (pros/cons, tables if helpful)
- Highlight key differences and similarities
- Consider different use cases and tradeoffs
- Be fair to all options being compared""",
        }

        # Response style instructions
        style_instructions = {
            "brief": "RESPONSE STYLE: User wants a quick answer - be direct and concise.",
            "detailed": "RESPONSE STYLE: User wants depth - provide thorough explanations with examples.",
            "step-by-step": "RESPONSE STYLE: User wants a guide - use numbered steps with clear instructions.",
            "conversational": "RESPONSE STYLE: User is casual - be friendly and match the chat's relaxed tone.",
            "sarcastic": "RESPONSE STYLE: User is being playful or sarcastic - match their energy with witty, humorous responses while still being helpful.",
        }

        # Add query type instructions
        query_type = query_metadata.query_type
        if query_type in type_instructions:
            prompt_parts.append(type_instructions[query_type])

        # Add style instructions
        style = query_metadata.suggested_response_style
        if style in style_instructions:
            prompt_parts.append(style_instructions[style])

        # Add key topics
        if query_metadata.key_topics:
            prompt_parts.append(f"KEY TOPICS IDENTIFIED: {', '.join(query_metadata.key_topics)}")

        # Add follow-up context hint
        if query_metadata.is_followup:
            prompt_parts.append(
                "FOLLOW-UP DETECTED: This question references previous messages. Pay close attention to the conversation history to understand what the user is referring to."
            )

        # Add language instruction for non-English
        if query_metadata.language != "en":
            prompt_parts.append(
                f"LANGUAGE: Respond in {query_metadata.language} (the user's language)"
            )

    # Standard guidelines (always included)
    prompt_parts.append("""GENERAL GUIDELINES:
- Reference the conversation history ONLY when relevant
- If someone asks about a previous message or image, ONLY THEN use the provided context
- When using web search results, synthesize information naturally and ALWAYS cite sources
- If unsure about something from the conversation, say so
- Keep responses under 1000 characters unless a detailed explanation is needed
- You are responding in a Discord chat - be helpful but concise
- Be cheeky, fun and playful when responding to casual or non-serious topics - match the vibe of the conversation""")

    return "\n\n".join(prompt_parts)


def split_response(text: str, max_length: int = 2000) -> list[str]:
    """Split a long response into multiple Discord-safe chunks.

    Attempts to split at natural boundaries (paragraphs, sentences).

    Args:
        text: The full response text.
        max_length: Maximum length per chunk (Discord limit is 2000).

    Returns:
        List of text chunks, each under max_length.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_length:
        split_point = max_length

        para_break = remaining.rfind("\n\n", 0, max_length)
        if para_break > max_length // 2:
            split_point = para_break + 2
        else:
            for punct in [". ", "! ", "? "]:
                sent_break = remaining.rfind(punct, 0, max_length)
                if sent_break > max_length // 2:
                    split_point = sent_break + len(punct)
                    break
            else:
                space = remaining.rfind(" ", 0, max_length)
                if space > max_length // 2:
                    split_point = space + 1

        chunks.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def format_error_response(error_type: str, details: str | None = None) -> str:
    """Format a user-friendly error message.

    Args:
        error_type: Category of error (api, rate_limit, config, exhausted, etc.)
        details: Optional additional details.

    Returns:
        Formatted error message for Discord.
    """
    messages = {
        "api": "I couldn't reach the AI service right now. Please try again in a moment.",
        "rate_limit": "I'm getting too many requests. Please wait a minute before trying again.",
        "config": "The AI feature isn't configured properly. Please contact a server admin.",
        "no_question": "Please provide a question after `!ask`. Example: `!ask What time is it in Tokyo?`",
        "too_long": "Your question is too long. Please keep it under 20,000 characters.",
        "exhausted": "Daily limit reached (60 requests). Try again tomorrow.",
    }

    base_message = messages.get(error_type, "Something went wrong. Please try again.")

    if details and error_type == "api":
        return f"{base_message}\n||Debug: {details[:100]}||"

    return base_message
