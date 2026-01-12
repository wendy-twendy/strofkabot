# RAG (Retrieval-Augmented Generation) module for semantic search over message history

"""RAG implementation for strofkabot message history.

This module provides:
- Conversation-aware chunking (reply chains + time windows + semantic splitting)
- OpenRouter embeddings via google/gemini-embedding-001
- LLM metadata extraction via google/gemini-2.0-flash-lite-001
- LanceDB vector store for persistence (ChromaDB not available for Python 3.14)
- RAG pipeline for semantic search and Q&A
"""

from .chunking import ConversationGroup, ConversationGrouper

__all__ = [
    "ConversationGroup",
    "ConversationGrouper",
]

# Optional imports - add as they're implemented
try:
    from .chunking import ChunkFormatter, FormattedChunk

    __all__.extend(["ChunkFormatter", "FormattedChunk"])
except ImportError:
    pass

try:
    from .embeddings import EmbeddingResponse, OpenRouterEmbeddingClient

    __all__.extend(["EmbeddingResponse", "OpenRouterEmbeddingClient"])
except ImportError:
    pass

try:
    from .metadata import (
        ChunkMetadata,
        ExtractedMetadata,
        ExtractionResponse,
        MetadataExtractor,
    )

    __all__.extend(
        [
            "ChunkMetadata",
            "ExtractedMetadata",
            "ExtractionResponse",
            "MetadataExtractor",
        ]
    )
except ImportError:
    pass

try:
    from .vector_store import SearchResult, VectorStore

    __all__.extend(["SearchResult", "VectorStore"])
except ImportError:
    pass

try:
    from .pipeline import RAGPipeline, RAGResponse, SearchRequest

    __all__.extend(["RAGPipeline", "RAGResponse", "SearchRequest"])
except ImportError:
    pass

try:
    from .llm_client import LLMResponse, RAGLLMClient

    __all__.extend(["RAGLLMClient", "LLMResponse"])
except ImportError:
    pass

try:
    from .ingestion import IngestionPipeline, IngestionProgress

    __all__.extend(["IngestionPipeline", "IngestionProgress"])
except ImportError:
    pass

try:
    from .extraction import ExtractionResult, extract_relevant_context

    __all__.extend(["ExtractionResult", "extract_relevant_context"])
except ImportError:
    pass

try:
    from .query_rewriter import (
        ConversationMessage,
        MemberInfo,
        RewrittenQuery,
        rewrite_query,
    )

    __all__.extend(["ConversationMessage", "MemberInfo", "RewrittenQuery", "rewrite_query"])
except ImportError:
    pass
