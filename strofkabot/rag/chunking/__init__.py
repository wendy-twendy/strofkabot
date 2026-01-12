# Conversation-aware chunking module

"""Chunking module for conversation-aware message grouping.

Provides a 3-layer hybrid chunking strategy:
- Layer 1: Conversation grouping (reply chains + time windows)
- Layer 2: Semantic boundary detection using embeddings
- Layer 3: Large message handling
"""

from .conversation_grouper import ConversationGroup, ConversationGrouper

__all__ = [
    "ConversationGroup",
    "ConversationGrouper",
]

# Optional imports - add as they're implemented
try:
    from .chunk_formatter import ChunkFormatter, FormattedChunk

    __all__.extend(["ChunkFormatter", "FormattedChunk"])
except ImportError:
    pass

try:
    from .semantic_splitter import SemanticSplitter

    __all__.append("SemanticSplitter")
except ImportError:
    pass
