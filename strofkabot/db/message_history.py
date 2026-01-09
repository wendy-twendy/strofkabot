# db/message_history.py

"""Mixin for message history operations (unfiltered messages for AI context)."""

import datetime
from dataclasses import dataclass


@dataclass
class HistoryMessage:
    """Represents an unfiltered message for AI context."""

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


class MessageHistoryMixin:
    """Mixin providing message history operations for AI context."""

    async def add_history_messages(self, messages: list[HistoryMessage]):
        """Batch insert messages into message_history table."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.executemany(
            """
            INSERT OR REPLACE INTO message_history
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

    async def get_history_message_count(self) -> int:
        """Get total number of messages in the message_history table."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute("SELECT COUNT(*) FROM message_history") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_hourly_activity_by_user(
        self, author_id: int, timezone_offset: int = 0
    ) -> list[tuple[int, int, int]]:
        """Get message counts by day-of-week and hour for a user (last 3 months).

        Args:
            author_id: Discord user ID to query.
            timezone_offset: Hours offset from UTC (e.g., +1 for CET, -5 for EST).

        Returns:
            List of (day_of_week, hour, count) tuples.
            day_of_week: 0=Sunday through 6=Saturday (SQLite convention).
            hour: 0-23.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        # Build timezone offset string for SQLite datetime modifier
        offset_str = f"{timezone_offset:+d} hours"

        query = """
            SELECT
                CAST(strftime('%w', datetime(timestamp, ?)) AS INTEGER) as day_of_week,
                CAST(strftime('%H', datetime(timestamp, ?)) AS INTEGER) as hour,
                COUNT(*) as message_count
            FROM message_history
            WHERE author_id = ?
              AND timestamp >= date('now', '-3 months')
            GROUP BY day_of_week, hour
            ORDER BY day_of_week, hour
        """

        async with self.conn.execute(query, (offset_str, offset_str, author_id)) as cursor:
            rows = await cursor.fetchall()
            return [(row[0], row[1], row[2]) for row in rows]

    async def get_hourly_activity_by_user_for_month(
        self, author_id: int, year: int, month: int, timezone_offset: int = 0
    ) -> list[tuple[int, int, int]]:
        """Get message counts by day-of-week and hour for a user for a specific month.

        Args:
            author_id: Discord user ID to query.
            year: Year to filter by.
            month: Month to filter by (1-12).
            timezone_offset: Hours offset from UTC (e.g., +1 for CET, -5 for EST).

        Returns:
            List of (day_of_week, hour, count) tuples.
            day_of_week: 0=Sunday through 6=Saturday (SQLite convention).
            hour: 0-23.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        # Build timezone offset string for SQLite datetime modifier
        offset_str = f"{timezone_offset:+d} hours"

        query = """
            SELECT
                CAST(strftime('%w', datetime(timestamp, ?)) AS INTEGER) as day_of_week,
                CAST(strftime('%H', datetime(timestamp, ?)) AS INTEGER) as hour,
                COUNT(*) as message_count
            FROM message_history
            WHERE author_id = ?
              AND strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
            GROUP BY day_of_week, hour
            ORDER BY day_of_week, hour
        """

        year_str = str(year)
        month_str = f"{month:02d}"

        async with self.conn.execute(
            query, (offset_str, offset_str, author_id, year_str, month_str)
        ) as cursor:
            rows = await cursor.fetchall()
            return [(row[0], row[1], row[2]) for row in rows]
