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
MESSAGE_HISTORY_DATABASE_FILE = Path(__file__).parent.parent / "data" / "message_history.db"
ATTACHMENTS_DIR = Path(__file__).parent.parent / "data" / "attachments"

# Message filtering
REACT_COUNT_THRESHOLD = 4

# Image processing settings
IMAGE_MAX_SIZE_BYTES = 1024 * 1024  # 1 MB
IMAGE_QUALITY_START = 80
IMAGE_QUALITY_MIN = 30
