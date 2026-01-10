"""Tool definitions for memory extraction via function calling."""


def build_memory_tools(known_users: dict[str, int]) -> list[dict]:
    """Build tool definitions for memory extraction.

    Args:
        known_users: Mapping of display_name -> user_id for users in context.

    Returns:
        List of tool definitions for memory CRUD operations.
    """
    # Build enum of valid user IDs from known users
    user_enum = list(known_users.values())
    user_descriptions = [f"{name} ({uid})" for name, uid in known_users.items()]

    return [
        # Save user memory
        {
            "type": "function",
            "function": {
                "name": "save_user_memory",
                "description": (
                    "Save a NEW long-term fact about a specific user. Only call for "
                    "genuinely important, persistent facts NOT already known. Valid users: "
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
                        "confidence": {
                            "type": "number",
                            "minimum": 0.5,
                            "maximum": 1.0,
                            "description": (
                                "How certain is this fact? 1.0 = explicitly stated, "
                                "0.8 = strongly implied, 0.6 = inferred. Only save if >= 0.7"
                            ),
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                            "description": "Keywords for retrieval (e.g., 'programming', 'food')",
                        },
                    },
                    "required": ["user_id", "memory_text", "category", "importance"],
                },
            },
        },
        # Update user memory
        {
            "type": "function",
            "function": {
                "name": "update_user_memory",
                "description": (
                    "Update an existing memory when new information supersedes it. "
                    "Use when a user's situation has changed (moved, new job, changed preference). "
                    f"Valid users: {', '.join(user_descriptions)}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "enum": user_enum,
                            "description": "Discord user ID",
                        },
                        "old_memory_match": {
                            "type": "string",
                            "description": "Text from the existing memory to update (partial match OK)",
                        },
                        "new_memory_text": {
                            "type": "string",
                            "description": "The updated memory text",
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
                        },
                    },
                    "required": [
                        "user_id",
                        "old_memory_match",
                        "new_memory_text",
                        "category",
                        "importance",
                    ],
                },
            },
        },
        # Invalidate user memory
        {
            "type": "function",
            "function": {
                "name": "invalidate_user_memory",
                "description": (
                    "Mark a memory as no longer valid. Use when someone explicitly "
                    "contradicts a stored fact (e.g., 'I don't like X anymore'). "
                    f"Valid users: {', '.join(user_descriptions)}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "enum": user_enum,
                            "description": "Discord user ID",
                        },
                        "memory_text_match": {
                            "type": "string",
                            "description": "Text of the memory to invalidate (partial match OK)",
                        },
                    },
                    "required": ["user_id", "memory_text_match"],
                },
            },
        },
        # Save server memory
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
                        "confidence": {
                            "type": "number",
                            "minimum": 0.7,
                            "maximum": 1.0,
                            "description": "Confidence in this fact (0.7-1.0)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                            "description": "Keywords for retrieval",
                        },
                    },
                    "required": ["memory_text", "category", "importance"],
                },
            },
        },
        # Update server memory
        {
            "type": "function",
            "function": {
                "name": "update_server_memory",
                "description": ("Update an existing server memory with new information."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "old_memory_match": {
                            "type": "string",
                            "description": "Text from the existing memory to update (partial match OK)",
                        },
                        "new_memory_text": {
                            "type": "string",
                            "description": "The updated memory text",
                            "maxLength": 150,
                        },
                        "category": {
                            "type": "string",
                            "enum": ["jokes", "memes", "knowledge", "events"],
                        },
                        "importance": {
                            "type": "integer",
                            "minimum": 7,
                            "maximum": 10,
                        },
                    },
                    "required": ["old_memory_match", "new_memory_text", "category", "importance"],
                },
            },
        },
        # Invalidate server memory
        {
            "type": "function",
            "function": {
                "name": "invalidate_server_memory",
                "description": ("Mark a server memory as no longer valid."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_text_match": {
                            "type": "string",
                            "description": "Text of the memory to invalidate (partial match OK)",
                        },
                    },
                    "required": ["memory_text_match"],
                },
            },
        },
    ]
