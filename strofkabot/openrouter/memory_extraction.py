"""Memory filtering and extraction via LLM function calling."""

import json
import logging

from openai import AsyncOpenAI

from strofkabot.config import OPENROUTER_ROUTER_MODEL
from strofkabot.openrouter.memory_tools import build_memory_tools

logger = logging.getLogger(__name__)

RELEVANCE_PROMPT = """Given a question and list of known facts about the user/server, identify which facts (if any) are DIRECTLY relevant and would help answer the question better.

Be STRICT: only include facts that genuinely help answer THIS specific question.
- If asking about food → dietary preferences are relevant
- If asking about coding → programming interests/job are relevant
- If casual chat with no clear connection → return empty array (don't force irrelevant info)

Output JSON with a "relevant" field containing indices (1-indexed) of relevant facts.
Example: {"relevant": [1, 3]} or {"relevant": []}"""

EXTRACTION_PROMPT = """You are a memory extractor for a Discord bot. Analyze the conversation and extract ONLY genuinely useful, long-term facts.

WHAT TO SAVE (call save_*_memory):
- Explicit personal facts: "I'm a software engineer", "I live in Berlin"
- Clear preferences: "I love jazz", "I hate mornings"
- Life events: "I just got married", "I'm starting a new job"
- Recurring server jokes/traditions referenced multiple times

WHAT NOT TO SAVE (do NOT call any tools):
- Temporary states: "I'm tired today", "I'm eating lunch"
- Opinions about current events: "That movie was bad"
- Speculation or inference without explicit statement
- Anything about the bot itself (Llumi, StrofkaBot)
- Facts already known (check the "Already known" sections)

WHEN TO UPDATE (use update_*_memory):
- User explicitly contradicts a previous fact with new info
- User's situation has changed (moved, new job, changed preference)

WHEN TO INVALIDATE (use invalidate_*_memory):
- User explicitly says "I don't X anymore" or "I stopped doing Y"
- Information is clearly outdated by new statement

DEFAULT BEHAVIOR: Call NO tools. Most conversations have nothing worth saving.
When in doubt, don't save. False positives are worse than false negatives."""


async def filter_relevant_memories(
    client: AsyncOpenAI,
    question: str,
    memories: list,
) -> list[int]:
    """Filter memories to only those relevant to the current question.

    Args:
        client: AsyncOpenAI client instance.
        question: The user's question.
        memories: List of Memory objects to filter.

    Returns:
        List of indices of relevant memories (0-indexed).
    """
    if not memories:
        return []

    # Build numbered list of memories for the classifier
    memory_list = "\n".join(f"{i + 1}. {mem.text}" for i, mem in enumerate(memories))

    user_content = f"Question: {question}\n\nKnown facts:\n{memory_list}"

    try:
        response = await client.chat.completions.create(
            model=OPENROUTER_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": RELEVANCE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        relevant_indices = result.get("relevant", [])

        # Convert 1-indexed to 0-indexed and validate
        valid_indices = []
        for idx in relevant_indices:
            if isinstance(idx, int) and 1 <= idx <= len(memories):
                valid_indices.append(idx - 1)

        logger.debug(
            "Memory relevance filter: %d/%d memories relevant",
            len(valid_indices),
            len(memories),
        )

        return valid_indices

    except Exception as e:
        logger.warning("Memory relevance filter failed (%s), returning all", e)
        # On error, return all memories rather than none
        return list(range(len(memories)))


async def extract_memories(
    client: AsyncOpenAI,
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
        client: AsyncOpenAI client instance.
        context_messages: List of context message dicts.
        question: The user's question.
        response_text: The AI's response (not used in extraction anymore).
        user_id: Discord ID of the user who asked.
        user_name: Display name of the user who asked.
        known_users: Mapping of display_name -> user_id for users in context.
        existing_user_memories: List of existing Memory objects for the user.
        existing_server_memories: List of existing server Memory objects.

    Returns:
        Dict with memory operations:
        - "user_memories": list of new memories to save
        - "server_memories": list of new server memories to save
        - "user_updates": list of {user_id, old_match, new_memory} for updates
        - "server_updates": list of {old_match, new_memory} for updates
        - "user_invalidations": list of {user_id, text_match} for invalidations
        - "server_invalidations": list of {text_match} for invalidations
    """
    # Build known_users if not provided (fallback to just the asker)
    if known_users is None:
        known_users = {user_name.lower(): user_id}

    # Filter bot messages from context for extraction
    context_lines = []
    for msg in context_messages[-10:]:
        if msg.get("is_bot"):
            continue
        author = msg.get("author", "Unknown")
        content = msg.get("content", "")[:200]
        context_lines.append(f"{author}: {content}")

    context_text = "\n".join(context_lines)

    # Build existing memory context
    existing_context = ""
    if existing_user_memories:
        user_facts = "\n".join(f"  - {m.text}" for m in existing_user_memories[:10])
        existing_context += f"\n\nAlready known about {user_name}:\n{user_facts}"
    if existing_server_memories:
        server_facts = "\n".join(f"  - {m.text}" for m in existing_server_memories[:10])
        existing_context += f"\n\nAlready known about this server:\n{server_facts}"

    # Build tools with current known users
    tools = build_memory_tools(known_users)

    user_content = f"""Conversation context:
{context_text}

Question from {user_name}: {question}{existing_context}

(Only save NEW facts. Use update/invalidate for changes to existing memories.)"""

    try:
        response = await client.chat.completions.create(
            model=OPENROUTER_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500,
        )

        # Process tool calls
        user_memories = []
        server_memories = []
        user_updates = []
        server_updates = []
        user_invalidations = []
        server_invalidations = []

        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    continue

                func_name = tool_call.function.name

                if func_name == "save_user_memory":
                    # Validate user_id is in known_users (defense in depth)
                    # Note: LLM may return user_id as string, so convert to int
                    try:
                        extracted_user_id = int(args.get("user_id", 0))
                    except (TypeError, ValueError):
                        extracted_user_id = 0
                    if extracted_user_id in known_users.values():
                        user_memories.append(
                            {
                                "user_id": extracted_user_id,
                                "memory_text": args.get("memory_text", ""),
                                "category": args.get("category", "facts"),
                                "importance": args.get("importance", 5),
                                "confidence": args.get("confidence", 1.0),
                                "tags": args.get("tags", []),
                            }
                        )

                elif func_name == "update_user_memory":
                    try:
                        extracted_user_id = int(args.get("user_id", 0))
                    except (TypeError, ValueError):
                        extracted_user_id = 0
                    if extracted_user_id in known_users.values():
                        user_updates.append(
                            {
                                "user_id": extracted_user_id,
                                "old_match": args.get("old_memory_match", ""),
                                "new_memory_text": args.get("new_memory_text", ""),
                                "category": args.get("category", "facts"),
                                "importance": args.get("importance", 5),
                            }
                        )

                elif func_name == "invalidate_user_memory":
                    try:
                        extracted_user_id = int(args.get("user_id", 0))
                    except (TypeError, ValueError):
                        extracted_user_id = 0
                    if extracted_user_id in known_users.values():
                        user_invalidations.append(
                            {
                                "user_id": extracted_user_id,
                                "text_match": args.get("memory_text_match", ""),
                            }
                        )

                elif func_name == "save_server_memory":
                    # Additional filter: reject bot-related memories
                    text_lower = args.get("memory_text", "").lower()
                    bot_indicators = ["strofkabot", "llumi", "the bot", "bot's"]
                    if not any(x in text_lower for x in bot_indicators):
                        server_memories.append(
                            {
                                "memory_text": args.get("memory_text", ""),
                                "category": args.get("category", "knowledge"),
                                "importance": args.get("importance", 7),
                                "confidence": args.get("confidence", 1.0),
                                "tags": args.get("tags", []),
                            }
                        )

                elif func_name == "update_server_memory":
                    text_lower = args.get("new_memory_text", "").lower()
                    bot_indicators = ["strofkabot", "llumi", "the bot", "bot's"]
                    if not any(x in text_lower for x in bot_indicators):
                        server_updates.append(
                            {
                                "old_match": args.get("old_memory_match", ""),
                                "new_memory_text": args.get("new_memory_text", ""),
                                "category": args.get("category", "knowledge"),
                                "importance": args.get("importance", 7),
                            }
                        )

                elif func_name == "invalidate_server_memory":
                    server_invalidations.append(
                        {
                            "text_match": args.get("memory_text_match", ""),
                        }
                    )

        logger.debug(
            "Memory extraction: %d saves, %d updates, %d invalidations (user); "
            "%d saves, %d updates, %d invalidations (server)",
            len(user_memories),
            len(user_updates),
            len(user_invalidations),
            len(server_memories),
            len(server_updates),
            len(server_invalidations),
        )

        return {
            "user_memories": user_memories,
            "server_memories": server_memories,
            "user_updates": user_updates,
            "server_updates": server_updates,
            "user_invalidations": user_invalidations,
            "server_invalidations": server_invalidations,
        }

    except Exception as e:
        logger.warning("Memory extraction failed (%s)", e)
        return {
            "user_memories": [],
            "server_memories": [],
            "user_updates": [],
            "server_updates": [],
            "user_invalidations": [],
            "server_invalidations": [],
        }
