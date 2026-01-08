"""Helper functions for the !ask command."""

import base64
import io
import logging
from pathlib import Path

import aiohttp
import discord
from PIL import Image

from strofkabot.image_processor import IMAGE_EXTENSIONS
from strofkabot.utils.discord_helpers import get_reply_info

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
) -> tuple[list[dict], list[dict]]:
    """Prepare context messages and extract images for the Gemini API.

    Args:
        messages: List of Discord messages.

    Returns:
        Tuple of (context_dicts, image_dicts)
        - context_dicts: List of formatted message context
        - image_dicts: List of {data: base64, mime_type: str} for images
    """
    context = []
    images = []

    for msg in messages:
        reply_to_id, reply_to_author, reply_to_content = get_reply_info(msg)

        image_attachments = [att for att in msg.attachments if is_image_attachment(att.filename)]

        context_entry = {
            "author": msg.author.display_name,
            "content": msg.content or "[no text]",
            "timestamp": msg.created_at.strftime("%H:%M"),
            "reply_to_author": reply_to_author,
            "reply_to_content": reply_to_content[:100] if reply_to_content else None,
            "image_count": len(image_attachments),
        }
        context.append(context_entry)

        for att in image_attachments[:3]:
            image_data = await download_and_encode_image(att.url, att.filename)
            if image_data:
                images.append(image_data)

    return context, images


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
    user_roles: list[str],
) -> str:
    """Build a Discord-aware system prompt for Gemini.

    Args:
        guild_name: Name of the Discord server.
        channel_name: Name of the current channel.
        user_name: Display name of the user asking.
        user_roles: List of role names the user has.

    Returns:
        Formatted system prompt string.
    """
    roles_str = ", ".join(user_roles[:5]) if user_roles else "Member"

    return f"""You are a helpful assistant in the "{guild_name}" Discord server.

CONTEXT:
- Server: {guild_name}
- Channel: #{channel_name}
- User asking: {user_name} (roles: {roles_str})

YOUR ROLE:
- Answer questions helpfully and concisely
- You can see recent conversation history for context
- You may receive images from the conversation - describe them if relevant
- Use Google Search when asked about current events, recent news, or factual information
- Keep responses under 1800 characters to fit Discord's message limit

GUIDELINES:
- Be conversational and friendly
- Reference the conversation context when relevant
- If someone asks about a previous message or image, use the provided history
- When using search results, synthesize the information naturally
- If you're unsure about something from the conversation, say so

IMPORTANT: You are responding in a Discord chat. Be concise but helpful."""


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
