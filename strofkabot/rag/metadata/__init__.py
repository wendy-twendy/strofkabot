# Metadata extraction module

"""Metadata extraction for conversation chunks.

Provides:
- ChunkMetadata schema for ChromaDB storage
- ExtractedMetadata for LLM output parsing
- MetadataExtractor for LLM-based extraction (topics, sentiment, summary)
"""

from .extractor import ExtractionResponse, MetadataExtractor
from .schema import ChunkMetadata, ExtractedMetadata

__all__ = [
    "ChunkMetadata",
    "ExtractedMetadata",
    "ExtractionResponse",
    "MetadataExtractor",
]
