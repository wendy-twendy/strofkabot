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
MESSAGE_HISTORY_DATABASE_FILE = Path(__file__).parent.parent / "data" / "message_history.db"
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
OPENROUTER_ROUTER_MODEL = (
    "xiaomi/mimo-v2-flash:free"  # Fast, free classifier for search/thinking detection
)
OPENROUTER_VISION_MODEL = "google/gemini-3-flash-preview"  # Vision model for images

# Memory system configuration
MEMORIES_DIR = Path(__file__).parent.parent / "data" / "memories"
MEMORY_USER_LIMIT = 20  # Max memories per user
MEMORY_SERVER_LIMIT = 50  # Max server-wide memories
MEMORY_TEXT_MAX_LENGTH = 200  # Max characters per memory
MEMORY_INJECTION_ENABLED = False  # Set to True to inject memories into prompts
