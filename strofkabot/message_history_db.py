# message_history_db.py

"""Database access layer for AI message history (unfiltered messages)."""

import datetime
from dataclasses import dataclass
from pathlib import Path

import aiosqlite


@dataclass
class HistoryMessage:
    id: int
    channel_id: int
    channel_name: str
    author_id: int
    author_name: str
    content: str
    timestamp: datetime.datetime
    reply_to_id: int | None
    reply_to_author: str | None
    reply_to_content: str | None
    reactions: str  # JSON array of {emoji, count}


class MessageHistoryDatabase:
    """Database for storing all messages (unfiltered) for AI purposes."""

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

            CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        """)
        await self.conn.commit()

    async def add_messages(self, messages: list[HistoryMessage]):
        """Batch insert messages."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.executemany(
            """
            INSERT OR REPLACE INTO messages
            (id, channel_id, channel_name, author_id, author_name, content, timestamp,
             reply_to_id, reply_to_author, reply_to_content, reactions)
            VALUES (:id, :channel_id, :channel_name, :author_id, :author_name, :content, :timestamp,
                    :reply_to_id, :reply_to_author, :reply_to_content, :reactions)
        """,
            [
                {
                    "id": msg.id,
                    "channel_id": msg.channel_id,
                    "channel_name": msg.channel_name,
                    "author_id": msg.author_id,
                    "author_name": msg.author_name,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "reply_to_id": msg.reply_to_id,
                    "reply_to_author": msg.reply_to_author,
                    "reply_to_content": msg.reply_to_content,
                    "reactions": msg.reactions,
                }
                for msg in messages
            ],
        ):
            pass
        await self.conn.commit()

    async def get_last_message_id(self, channel_id: int) -> int | None:
        """Get the last scraped message ID for a channel (for incremental scraping)."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute(
            "SELECT last_message_id FROM scrape_progress WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def update_scrape_progress(self, channel_id: int, last_message_id: int):
        """Update the scrape progress for a channel."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO scrape_progress (channel_id, last_message_id, last_updated)
            VALUES (?, ?, ?)
        """,
            (channel_id, last_message_id, datetime.datetime.now(datetime.UTC).isoformat()),
        )
        await self.conn.commit()

    async def get_message_count(self) -> int:
        """Get total number of messages in the database."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute("SELECT COUNT(*) FROM messages") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
