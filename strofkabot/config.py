"""Configuration constants for the Strofkabot Discord bot."""

from pathlib import Path

# Discord configuration
GUILD_ID = 413619835096924160

# Timing configuration
UPDATE_INTERVAL_SECONDS = 3600 * 24  # 24 hours

# File paths
DATABASE_FILE_LOCATION = Path(__file__).parent.parent / "data" / "db.sqlite3"
ARTAN_QUOTES_PATH = Path(__file__).parent.parent / "data" / "artan_quotes.yaml"

# Message filtering
REACT_COUNT_THRESHOLD = 4
