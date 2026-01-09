# db/base.py

"""Base database class with connection management and table creation."""

from pathlib import Path

import aiosqlite


class BaseDatabase:
    """Base class handling database connection and schema creation."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: aiosqlite.Connection | None = None

    async def initialize(self):
        if self.conn is None:
            self.conn = await aiosqlite.connect(str(self.db_path))
            await self._create_tables()

    async def ensure_connection(self):
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
        """)
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
