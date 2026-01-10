# db/messages.py

"""Message operations mixin for the database."""

import datetime
from dataclasses import dataclass

from strofkabot.db.base import parse_datetime_safe


@dataclass
class Message:
    id: int
    content: str
    timestamp: datetime.datetime
    reaction_count: int
    author_id: int
    reply_to_id: int | None = None
    reply_to_author: str | None = None
    reply_to_content: str | None = None


class MessagesMixin:
    """Mixin providing message-related database operations."""

    async def add_messages(self, messages: list[Message]):
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        await self.conn.executemany(
            """
            INSERT OR REPLACE INTO messages
            (id, content, timestamp, reaction_count, author_id, reply_to_id, reply_to_author, reply_to_content)
            VALUES (:id, :content, :timestamp, :reaction_count, :author_id, :reply_to_id, :reply_to_author, :reply_to_content)
        """,
            [msg.__dict__ for msg in messages],
        )
        await self.conn.commit()

    async def update_last_scanned_timestamp(self, channel_id: int, timestamp: datetime.datetime):
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        key = f"last_scanned_timestamp_{channel_id}"
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        """,
            (key, timestamp.isoformat()),
        )
        await self.conn.commit()

    async def get_last_scanned_timestamp(self, channel_id: int) -> datetime.datetime | None:
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        key = f"last_scanned_timestamp_{channel_id}"
        async with self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return parse_datetime_safe(row[0])
            return None

    async def get_random_message(self) -> Message | None:
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute("SELECT * FROM messages ORDER BY RANDOM() LIMIT 1") as cursor:
            row = await cursor.fetchone()
            if row:
                timestamp = parse_datetime_safe(row[2])
                if timestamp is None:
                    return None
                return Message(
                    id=row[0],
                    content=row[1],
                    timestamp=timestamp,
                    reaction_count=row[3],
                    author_id=row[4],
                    reply_to_id=row[5],
                    reply_to_author=row[6],
                    reply_to_content=row[7],
                )
            return None

    async def get_message_count(self) -> int:
        """Get total number of messages in the database."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute("SELECT COUNT(*) FROM messages") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
