"""Configuration constants for the Strofkabot Discord bot."""

from pathlib import Path

# Discord configuration
GUILD_ID = 413619835096924160
PREDICTIONS_CHANNEL_ID = 1458523811824537621

# Timing configuration
UPDATE_INTERVAL_SECONDS = 3600 * 24  # 24 hours

# File paths
DATABASE_FILE_LOCATION = Path(__file__).parent.parent / "data" / "db.sqlite3"
ARTAN_QUOTES_PATH = Path(__file__).parent.parent / "data" / "artan_quotes.yaml"
NICKNAMES_FILE = Path(__file__).parent.parent / "data" / "nicknames.yaml"
ATTACHMENTS_DIR = Path(__file__).parent.parent / "data" / "attachments"

# Message filtering
REACT_COUNT_THRESHOLD = 4

# Image processing settings
IMAGE_MAX_SIZE_BYTES = 1024 * 1024  # 1 MB
IMAGE_QUALITY_START = 80
IMAGE_QUALITY_MIN = 30

# Gemini AI configuration (free tier)
GEMINI_MODELS = [
    "gemini-2.5-flash",  # Primary - stable, good balance
    "gemini-2.5-flash-lite",  # Fallback - faster, lighter
]
GEMINI_RPD_LIMIT = 20  # Requests per day per model
GEMINI_MAX_CONTEXT_MESSAGES = 10
DISCORD_MAX_MESSAGE_LENGTH = 2000
GEMINI_USAGE_FILE = Path(__file__).parent.parent / "data" / "gemini_usage.json"

# OpenRouter AI configuration (primary, uses free models)
OPENROUTER_INFERENCE_MODEL = "google/gemini-3-flash-preview"  # Gemini Flash 3
OPENROUTER_ROUTER_MODEL = "google/gemini-3-flash-preview"  # Gemini Flash 3 for classification
OPENROUTER_VISION_MODEL = "google/gemini-3-flash-preview"  # Vision model for images

# Memory system configuration
MEMORIES_DIR = Path(__file__).parent.parent / "data" / "memories"
MEMORY_USER_LIMIT = 20  # Max memories per user
MEMORY_SERVER_LIMIT = 50  # Max server-wide memories
MEMORY_TEXT_MAX_LENGTH = 200  # Max characters per memory
MEMORY_INJECTION_ENABLED = False  # Set to True to inject memories into prompts

# RAG (Retrieval-Augmented Generation) configuration
RAG_VECTOR_STORE_DIR = Path(__file__).parent.parent / "data" / "vector_store"
RAG_EMBEDDING_MODEL = "google/gemini-embedding-001"
RAG_LLM_MODEL = "google/gemini-2.0-flash-lite-001"
RAG_RERANK_MODEL = "google/gemini-2.0-flash-lite-001"

# RAG extraction configuration (for !ask integration)
RAG_EXTRACTION_MODEL = "google/gemini-2.0-flash-lite-001"
RAG_SEARCH_K = 5  # Number of chunks to retrieve
RAG_EXTRACTION_MAX_TOKENS = 500
RAG_MAX_QUERY_VARIANTS = 3  # Max query variants to search in parallel
RAG_ENABLED = True  # Feature flag to enable/disable RAG in !ask
