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
        if not memories:
            return []

        # Build numbered list of memories for the classifier
        memory_list = "\n".join(f"{i + 1}. {mem.text}" for i, mem in enumerate(memories))

        relevance_prompt = """Given a question and list of known facts about the user/server, identify which facts (if any) are DIRECTLY relevant and would help answer the question better.

Be STRICT: only include facts that genuinely help answer THIS specific question.
- If asking about food → dietary preferences are relevant
- If asking about coding → programming interests/job are relevant
- If casual chat with no clear connection → return empty array (don't force irrelevant info)

Output JSON with a "relevant" field containing indices (1-indexed) of relevant facts.
Example: {"relevant": [1, 3]} or {"relevant": []}"""

        user_content = f"Question: {question}\n\nKnown facts:\n{memory_list}"

        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": relevance_prompt},
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

    def _build_memory_tools(self, known_users: dict[str, int]) -> list[dict]:
        """Build tool definitions for memory extraction.

        Args:
            known_users: Mapping of display_name -> user_id for users in context.

        Returns:
            List of tool definitions for save_user_memory and save_server_memory.
        """
        # Build enum of valid user IDs from known users
        user_enum = list(known_users.values())
        user_descriptions = [f"{name} ({uid})" for name, uid in known_users.items()]

        return [
            {
                "type": "function",
                "function": {
                    "name": "save_user_memory",
                    "description": (
                        "Save a long-term fact about a specific user. Only call for "
                        "genuinely important, persistent facts. Valid users: "
                        f"{', '.join(user_descriptions)}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "integer",
                                "enum": user_enum,
                                "description": "Discord user ID - MUST be one of the known users",
                            },
                            "memory_text": {
                                "type": "string",
                                "description": (
                                    "The fact to remember, in 3rd person present tense, "
                                    "under 150 chars"
                                ),
                                "maxLength": 150,
                            },
                            "category": {
                                "type": "string",
                                "enum": ["preferences", "facts", "interests", "events"],
                                "description": "Category of the memory",
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": (
                                    "1-10, where 10 is very important. "
                                    "Use 7+ only for truly significant facts."
                                ),
                            },
                        },
                        "required": ["user_id", "memory_text", "category", "importance"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_server_memory",
                    "description": (
                        "Save shared knowledge about this Discord server/community. "
                        "Only for recurring jokes, traditions, or unique cultural knowledge. "
                        "VERY SELECTIVE - most conversations have nothing worth saving."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_text": {
                                "type": "string",
                                "description": "The shared knowledge, under 150 chars",
                                "maxLength": 150,
                            },
                            "category": {
                                "type": "string",
                                "enum": ["jokes", "memes", "knowledge", "events"],
                                "description": "Category of the memory",
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 7,
                                "maximum": 10,
                                "description": "7-10 only. Server memories must be important (7+).",
                            },
                        },
                        "required": ["memory_text", "category", "importance"],
                    },
                },
            },
        ]

    async def extract_memories(
        self,
        context_messages: list[dict],
        question: str,
        response_text: str,
        user_id: int,
        user_name: str,
        known_users: dict[str, int] | None = None,
    ) -> dict:
        """Extract new memories worth saving from a conversation using tool calling.

        Args:
            context_messages: List of context message dicts.
            question: The user's question.
            response_text: The AI's response (not used in extraction anymore).
            user_id: Discord ID of the user who asked.
            user_name: Display name of the user who asked.
            known_users: Mapping of display_name -> user_id for users in context.

        Returns:
            Dict with "user_memories" and "server_memories" lists.
            Each memory has: user_id (for user memories), memory_text, category, importance.
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

        # Build tools with current known users
        tools = self._build_memory_tools(known_users)

        system_prompt = """You are a memory extractor for a Discord bot. Analyze the conversation and extract ONLY genuinely useful, long-term facts worth remembering.

RULES:
- NEVER extract anything about the bot itself (Llumi, StrofkaBot)
- Only extract facts explicitly stated, not speculation
- User memories: personal preferences, facts about them, interests, life events
- Server memories: ONLY recurring jokes or unique community knowledge (very rare)
- Most conversations have NOTHING worth saving. When in doubt, don't save.
- If importance would be below 5 for user memory or below 7 for server memory, don't save it.

Call save_user_memory or save_server_memory tools ONLY if there's something genuinely worth remembering. It's perfectly fine to call no tools at all - that should be the default."""

        user_content = f"""Conversation context:
{context_text}

Question from {user_name}: {question}

(Analyze the conversation above. Only save truly important, long-term facts.)"""

        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
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

            message = response.choices[0].message
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        continue

                    if tool_call.function.name == "save_user_memory":
                        # Validate user_id is in known_users (defense in depth)
                        if args.get("user_id") in known_users.values():
                            user_memories.append(
                                {
                                    "user_id": args["user_id"],
                                    "memory_text": args.get("memory_text", ""),
                                    "category": args.get("category", "facts"),
                                    "importance": args.get("importance", 5),
                                }
                            )

                    elif tool_call.function.name == "save_server_memory":
                        # Additional filter: reject bot-related memories
                        text_lower = args.get("memory_text", "").lower()
                        bot_indicators = ["strofkabot", "llumi", "the bot", "bot's"]
                        if not any(x in text_lower for x in bot_indicators):
                            server_memories.append(
                                {
                                    "memory_text": args.get("memory_text", ""),
                                    "category": args.get("category", "knowledge"),
                                    "importance": args.get("importance", 7),
                                }
                            )

            logger.debug(
                "Memory extraction (tools): %d user memories, %d server memories",
                len(user_memories),
                len(server_memories),
            )

            return {
                "user_memories": user_memories,
                "server_memories": server_memories,
            }

        except Exception as e:
            logger.warning("Memory extraction failed (%s)", e)
            return {"user_memories": [], "server_memories": []}
