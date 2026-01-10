# db/base.py

"""Base database class with connection management and table creation."""

import asyncio
import datetime
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


def parse_datetime_safe(value: str | None) -> datetime.datetime | None:
    """Safely parse an ISO format datetime string.

    Args:
        value: ISO format datetime string or None.

    Returns:
        Parsed datetime or None if parsing fails.
    """
    if value is None:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse datetime '{value}': {e}")
        return None


def parse_date_safe(value: str | None) -> datetime.date | None:
    """Safely parse an ISO format date string.

    Args:
        value: ISO format date string or None.

    Returns:
        Parsed date or None if parsing fails.
    """
    if value is None:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse date '{value}': {e}")
        return None


def validate_month(month: int) -> int:
    """Validate and clamp month to valid range (1-12)."""
    return max(1, min(12, month))


def validate_day(day: int) -> int:
    """Validate and clamp day to valid range (1-31)."""
    return max(1, min(31, day))


def validate_timezone_offset(offset: int) -> int:
    """Validate and clamp timezone offset to valid range (-12 to +14)."""
    return max(-12, min(14, offset))


# Database connection timeout in seconds
CONNECTION_TIMEOUT = 30.0


class DatabaseError(Exception):
    """Base exception for database errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class DatabaseInitializationError(DatabaseError):
    """Raised when database initialization (schema creation) fails."""

    pass


class BaseDatabase:
    """Base class handling database connection and schema creation."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize database connection and create tables.

        Thread-safe: uses async lock to prevent concurrent initialization.
        """
        async with self._init_lock:
            if self.conn is not None:
                return

            try:
                self.conn = await aiosqlite.connect(
                    str(self.db_path),
                    timeout=CONNECTION_TIMEOUT,
                )
                logger.debug(f"Connected to database: {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e

            try:
                await self._create_tables()
                logger.debug("Database tables created/verified")
            except Exception as e:
                logger.error(f"Failed to create database tables: {e}")
                # Clean up the connection on schema creation failure
                if self.conn:
                    try:
                        await self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                raise DatabaseInitializationError(f"Failed to create tables: {e}") from e

    async def ensure_connection(self):
        """Ensure database is connected, initializing if needed."""
        if self.conn is None:
            await self.initialize()

    async def _create_tables(self):
        await self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                content TEXT,
                timestamp TEXT,
                reaction_count INTEGER,
                author_id INTEGER,
                reply_to_id INTEGER,
                reply_to_author TEXT,
                reply_to_content TEXT
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS user_stats_monthly (
                author_id INTEGER,
                year INTEGER,
                month INTEGER,
                total_messages INTEGER,
                total_reactions INTEGER,
                PRIMARY KEY (author_id, year, month)
            );

            CREATE TABLE IF NOT EXISTS user_mapping (
                author_id INTEGER PRIMARY KEY,
                username TEXT,
                last_updated TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_reactions_monthly (
                giver_id INTEGER,
                receiver_id INTEGER,
                year INTEGER,
                month INTEGER,
                reaction_count INTEGER,
                PRIMARY KEY (giver_id, receiver_id, year, month)
            );

            CREATE TABLE IF NOT EXISTS user_replies_monthly (
                replier_id INTEGER,
                replied_to_id INTEGER,
                year INTEGER,
                month INTEGER,
                reply_count INTEGER,
                PRIMARY KEY (replier_id, replied_to_id, year, month)
            );

            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY,
                message_id INTEGER,
                message_content TEXT,
                author_id INTEGER,
                timestamp TEXT,
                reaction_count INTEGER,
                original_filename TEXT,
                local_path TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author_id INTEGER NOT NULL,
                author_name TEXT NOT NULL,
                channel_id INTEGER NOT NULL,
                target_date TEXT NOT NULL,
                prediction_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                posted BOOLEAN DEFAULT FALSE,
                posted_at TEXT,
                retry_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_target_date
                ON predictions(target_date, posted);

            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY,
                channel_id INTEGER,
                channel_name TEXT,
                author_id INTEGER,
                author_name TEXT,
                content TEXT,
                timestamp TEXT,
                reply_to_id INTEGER,
                reply_to_author TEXT,
                reply_to_content TEXT,
                reactions TEXT
            );

            CREATE TABLE IF NOT EXISTS scrape_progress (
                channel_id INTEGER PRIMARY KEY,
                last_message_id INTEGER,
                last_updated TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_message_history_channel_id
                ON message_history(channel_id);
            CREATE INDEX IF NOT EXISTS idx_message_history_timestamp
                ON message_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_message_history_author_timestamp
                ON message_history(author_id, timestamp);
        """)
        await self.conn.commit()

    async def close(self):
        """Close database connection safely."""
        if self.conn:
            try:
                await self.conn.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self.conn = None

    async def is_connected(self) -> bool:
        """Check if database connection is alive."""
        if self.conn is None:
            return False
        try:
            # Simple query to verify connection is working
            async with self.conn.execute("SELECT 1") as cursor:
                await cursor.fetchone()
            return True
        except Exception:
            return False
