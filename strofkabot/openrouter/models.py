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
