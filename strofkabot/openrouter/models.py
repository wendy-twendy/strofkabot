"""Data models for the OpenRouter client."""

from dataclasses import dataclass


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
    needs_rag: bool = False  # Whether server history lookup is needed
    rag_query: str | None = None  # DEPRECATED - kept for backwards compat
    # Enhanced query rewriting fields
    rag_queries: list[str] | None = None  # Multiple query variants for better retrieval
    resolved_query: str | None = None  # Query with pronouns/references resolved
    detected_entities: list[str] | None = None  # People/topics mentioned
    temporal_filter: dict | None = None  # {"after": "YYYY-MM-DD"} or {"before": ...}


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
