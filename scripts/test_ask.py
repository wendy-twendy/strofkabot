#!/usr/bin/env python3
"""Test script for !ask command - test AI models via OpenRouter.

This script allows testing AI integration without running the Discord bot.
It uses OpenRouter API which provides access to Gemini, DeepSeek, and other models.

Features:
- Auto-detects if web search is needed using a fast classifier model (gemma)
- Uses mimo-v2 as default inference model (free)
- Supports thinking/reasoning mode
- Shows Discord chunking simulation

Usage:
    ./scripts/test_ask.py "What is the capital of France?"
    ./scripts/test_ask.py "What happened in the news today?"  # Auto-detects search needed
    ./scripts/test_ask.py -s "Force search on this question"
    ./scripts/test_ask.py --no-auto "Skip auto-detection"
    ./scripts/test_ask.py -m google/gemini-3-flash-preview "Use specific model"
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from strofkabot.utils.ask_helpers import build_system_prompt, split_response

# Load environment variables from .env file
load_dotenv()

# Model configuration
DEFAULT_MODEL = "xiaomi/mimo-v2-flash:free"  # Fast, free inference model
ROUTER_MODEL = "xiaomi/mimo-v2-flash:free"  # Fast, free classifier for search detection

# Available models for testing
AVAILABLE_MODELS = [
    "xiaomi/mimo-v2-flash:free",
    "google/gemma-3-27b-it:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-2.0-flash-exp:free",
]


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_context(context_str: str | None) -> list[dict]:
    """Parse context string into list of message dicts.

    Supports two formats:
    1. Pipe-separated: "User1: Hello|User2: Hi there"
    2. JSON array: '[{"author": "User1", "content": "Hello"}]'
    """
    if not context_str:
        return []

    # Try JSON first
    if context_str.strip().startswith("["):
        try:
            return json.loads(context_str)
        except json.JSONDecodeError:
            pass

    # Parse pipe-separated format
    messages = []
    for i, part in enumerate(context_str.split("|")):
        part = part.strip()
        if ":" in part:
            author, content = part.split(":", 1)
            messages.append(
                {
                    "author": author.strip(),
                    "content": content.strip(),
                    "timestamp": f"10:{i:02d}",
                }
            )
        else:
            messages.append(
                {
                    "author": "User",
                    "content": part,
                    "timestamp": f"10:{i:02d}",
                }
            )

    return messages


def format_context_for_messages(context_messages: list[dict]) -> str:
    """Format context messages into a single text block for the prompt."""
    if not context_messages:
        return ""

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


async def detect_query_requirements(
    question: str,
    client: AsyncOpenAI,
    logger: logging.Logger,
) -> tuple[bool, bool, float]:
    """Use a fast model to detect if search and/or thinking are needed.

    Returns:
        Tuple of (needs_search: bool, needs_thinking: bool, elapsed_seconds: float)
    """
    classification_prompt = """You are a query classifier. Analyze the question and output JSON with two boolean fields.

SEARCH (true/false): Does this need CURRENT or RECENT information?
- true: news, current events, prices, weather, "latest", "today", "now", "recent", "this week"
- false: general knowledge, math, logic, history, programming, static facts

THINKING (true/false): Does this need COMPLEX REASONING or analysis?
- true: logic puzzles, multi-step problems, "why", "implications", "analyze", comparisons
- false: simple facts, definitions, "what is X", straightforward answers

Output ONLY valid JSON: {"search": true/false, "thinking": true/false}"""

    start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        needs_search = result.get("search", False)
        needs_thinking = result.get("thinking", False)

        logger.info(
            f"Router ({ROUTER_MODEL}): search={needs_search}, thinking={needs_thinking} in {elapsed:.2f}s"
        )

        return needs_search, needs_thinking, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"Router failed ({e}), defaulting to no search/thinking")
        return False, False, elapsed


async def ask_openrouter(
    question: str,
    model: str,
    thinking: bool,
    search: bool,
    temperature: float,
    context_messages: list[dict],
    logger: logging.Logger,
    router_time: float = 0.0,
) -> dict:
    """Send a question to OpenRouter and return the response with metadata."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Build system prompt (simulating Discord context)
    system_prompt = build_system_prompt(
        guild_name="Test Server",
        channel_name="test-channel",
        user_name="TestUser",
        user_roles=["Member"],
    )

    # Build messages in OpenAI format
    messages = [{"role": "system", "content": system_prompt}]

    # Add context if provided
    if context_messages:
        context_text = format_context_for_messages(context_messages)
        messages.append({"role": "user", "content": context_text})
        messages.append(
            {"role": "assistant", "content": "I can see the conversation history. How can I help?"}
        )

    # Add the actual question
    messages.append({"role": "user", "content": question})

    # Handle model suffix for search
    model_to_use = f"{model}:online" if search else model

    # Build extra_body for reasoning
    extra_body = {
        "HTTP-Referer": "https://github.com/strofkabot",
        "X-Title": "StrofkaBot Test Script",
    }
    if thinking:
        extra_body["reasoning"] = {"effort": "high"}

    logger.info(f"Model: {model_to_use}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Search: {'enabled' if search else 'disabled'}")
    logger.info(f"Thinking: {'enabled' if thinking else 'disabled'}")
    logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")

    if context_messages:
        logger.info(f"Context messages: {len(context_messages)}")

    # Make the API call
    start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
            extra_body=extra_body,
        )
        elapsed = time.time() - start_time

        message = response.choices[0].message
        result = {
            "success": True,
            "text": message.content or "",
            "model": model_to_use,
            "elapsed_seconds": elapsed,
            "total_time": elapsed + router_time,
            "router_time": router_time,
            "thinking_enabled": thinking,
            "search_enabled": search,
        }

        # Extract usage metadata
        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "response_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # Extract reasoning content if available
        if hasattr(message, "reasoning") and message.reasoning:
            result["reasoning"] = message.reasoning

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "model": model_to_use,
            "elapsed_seconds": elapsed,
            "total_time": elapsed + router_time,
            "router_time": router_time,
            "thinking_enabled": thinking,
            "search_enabled": search,
        }


async def get_openrouter_credits(logger: logging.Logger) -> dict | None:
    """Get remaining OpenRouter credits."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {api_key}"},
            ) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch credits: {e}")
    return None


def print_result(result: dict, verbose: bool, logger: logging.Logger) -> None:
    """Print the result in a formatted way."""
    print()
    print("=" * 60)

    if result["success"]:
        print(f"Model: {result['model']}")
        if result.get("router_time", 0) > 0:
            print(
                f"Time: {result['elapsed_seconds']:.2f}s inference + {result['router_time']:.2f}s router = {result['total_time']:.2f}s total"
            )
        else:
            print(f"Time: {result['elapsed_seconds']:.2f}s")
        print(f"Search: {'enabled' if result.get('search_enabled') else 'disabled'}")
        print(f"Thinking: {'enabled' if result.get('thinking_enabled') else 'disabled'}")

        if "usage" in result:
            usage = result["usage"]
            if usage.get("total_tokens"):
                print(
                    f"Tokens: {usage['total_tokens']} total "
                    f"({usage.get('prompt_tokens', '?')} prompt, "
                    f"{usage.get('response_tokens', '?')} response)"
                )

        print("=" * 60)

        # Print reasoning if available
        if verbose and "reasoning" in result:
            print("\n--- Reasoning Process ---")
            print(result["reasoning"])
            print("--- End Reasoning ---\n")

        # Print response
        print("\nResponse:")
        print("-" * 40)

        # Split for Discord simulation
        chunks = split_response(result["text"])
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"\n[Chunk {i + 1}/{len(chunks)}]")
            print(chunk)

        print("-" * 40)
        print(f"\nTotal response length: {len(result['text'])} chars")
        print(f"Discord chunks: {len(chunks)}")

    else:
        print("ERROR")
        print("=" * 60)
        print(f"Model: {result['model']}")
        print(f"Time: {result['elapsed_seconds']:.2f}s")
        print(f"Error: {result['error']}")


async def print_credits(logger: logging.Logger) -> None:
    """Print OpenRouter credits info."""
    credits = await get_openrouter_credits(logger)
    if credits:
        total = float(credits.get("total_credits", 0))
        used = float(credits.get("total_usage", 0))
        remaining = total - used
        print(
            f"\nOpenRouter Credits: ${remaining:.4f} remaining (${used:.4f} used of ${total:.4f})"
        )
    else:
        print("\nOpenRouter Credits: Unable to fetch")


def main():
    parser = argparse.ArgumentParser(
        description="Test AI models via OpenRouter with auto search detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is Python?"                    # No search (auto-detected)
  %(prog)s "What's the latest AI news?"         # Search enabled (auto-detected)
  %(prog)s -s "Force search on this"            # Force search on
  %(prog)s --no-auto "Skip auto-detection"      # No search, no auto-detect
  %(prog)s -m google/gemini-3-flash-preview "Use specific model"
  %(prog)s -t "Solve: If A>B and B>C, is A>C?"  # Enable thinking mode

Auto-detection (default):
  Uses gemma-3-27b-it with structured output to classify:
  - Search: needed for current/recent information
  - Thinking: needed for complex reasoning/analysis
  Use -s/-t to force on, or --no-auto to disable auto-detection.

Available models:
  xiaomi/mimo-v2-flash:free      - Fast, free (default)
  google/gemma-3-27b-it:free     - Fast, free (also used for routing)
  tngtech/deepseek-r1t2-chimera:free - Reasoning model, slower
  google/gemini-2.5-flash-lite   - Paid, fast
  google/gemini-3-flash-preview  - Paid, latest
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask (can also use -q)",
    )

    parser.add_argument(
        "-q",
        "--question-arg",
        dest="question_arg",
        help="The question to ask (alternative to positional)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for inference (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-t",
        "--thinking",
        action="store_true",
        help="Enable thinking/reasoning mode for complex problems",
    )

    parser.add_argument(
        "-s",
        "--search",
        action="store_true",
        help="Force web search on (skip auto-detection)",
    )

    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable auto-detection, no search unless -s is used",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)",
    )

    parser.add_argument(
        "-c",
        "--context",
        help="Context messages: 'User1: msg1|User2: msg2' or JSON array",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging and show reasoning process",
    )

    args = parser.parse_args()

    # Get question from either source
    question = args.question or args.question_arg
    if not question:
        parser.error("Please provide a question")

    # Setup logging
    logger = setup_logging(args.verbose)

    # Parse context
    context_messages = parse_context(args.context)

    # Run the query
    async def run():
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Determine if search/thinking are needed
        router_time = 0.0
        needs_search = args.search
        needs_thinking = args.thinking

        if args.no_auto:
            # Disable auto-detection, use only explicit flags
            logger.info("Auto-detection disabled, using explicit flags only")
        elif not args.search or not args.thinking:
            # Auto-detect using router model (unless both are already forced on)
            logger.info(f"Auto-detecting requirements using {ROUTER_MODEL}...")
            auto_search, auto_thinking, router_time = await detect_query_requirements(
                question, client, logger
            )

            # Apply auto-detection only if not explicitly set
            if not args.search:
                needs_search = auto_search
            if not args.thinking:
                needs_thinking = auto_thinking

        if args.search:
            logger.info("Search: forced ON via -s flag")
        if args.thinking:
            logger.info("Thinking: forced ON via -t flag")

        result = await ask_openrouter(
            question=question,
            model=args.model,
            thinking=needs_thinking,
            search=needs_search,
            temperature=args.temperature,
            context_messages=context_messages,
            logger=logger,
            router_time=router_time,
        )
        print_result(result, args.verbose, logger)
        await print_credits(logger)
        return result

    try:
        result = asyncio.run(run())
        sys.exit(0 if result["success"] else 1)

    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
