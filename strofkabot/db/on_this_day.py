# db/on_this_day.py

"""On This Day operations mixin for the database."""

import datetime

from strofkabot.db.attachments import Attachment
from strofkabot.db.messages import Message


class OnThisDayMixin:
    """Mixin providing On This Day database operations."""

    async def get_on_this_day_years(self, month: int, day: int) -> list[int]:
        """Get distinct years that have messages or attachments on the given month-day.

        Args:
            month: The month (1-12)
            day: The day of month (1-31)

        Returns:
            List of years with content on this day, sorted ascending.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        month_day = f"{month:02d}-{day:02d}"

        query = """
            SELECT DISTINCT strftime('%Y', timestamp) as year
            FROM (
                SELECT timestamp FROM messages
                WHERE strftime('%m-%d', timestamp) = ?
                UNION ALL
                SELECT timestamp FROM attachments
                WHERE strftime('%m-%d', timestamp) = ?
            )
            ORDER BY year ASC
        """
        async with self.conn.execute(query, (month_day, month_day)) as cursor:
            rows = await cursor.fetchall()
            return [int(row[0]) for row in rows]

    async def get_top_message_on_this_day(
        self, year: int, month: int, day: int
    ) -> tuple[Message | None, Attachment | None]:
        """Get the highest-reacted message or attachment from a specific date.

        Args:
            year: The year
            month: The month (1-12)
            day: The day of month (1-31)

        Returns:
            Tuple of (Message or None, Attachment or None). The one with higher
            reaction count will be set, the other will be None.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        date_prefix = f"{year:04d}-{month:02d}-{day:02d}"

        # Get top message
        msg_query = """
            SELECT * FROM messages
            WHERE timestamp LIKE ? || '%'
            ORDER BY reaction_count DESC
            LIMIT 1
        """
        async with self.conn.execute(msg_query, (date_prefix,)) as cursor:
            msg_row = await cursor.fetchone()

        # Get top attachment
        att_query = """
            SELECT * FROM attachments
            WHERE timestamp LIKE ? || '%'
            ORDER BY reaction_count DESC
            LIMIT 1
        """
        async with self.conn.execute(att_query, (date_prefix,)) as cursor:
            att_row = await cursor.fetchone()

        # Parse results
        top_message = None
        top_attachment = None

        if msg_row:
            top_message = Message(
                id=msg_row[0],
                content=msg_row[1],
                timestamp=datetime.datetime.fromisoformat(msg_row[2]),
                reaction_count=msg_row[3],
                author_id=msg_row[4],
                reply_to_id=msg_row[5],
                reply_to_author=msg_row[6],
                reply_to_content=msg_row[7],
            )

        if att_row:
            top_attachment = Attachment(
                id=att_row[0],
                message_id=att_row[1],
                message_content=att_row[2],
                author_id=att_row[3],
                timestamp=datetime.datetime.fromisoformat(att_row[4]),
                reaction_count=att_row[5],
                original_filename=att_row[6],
                local_path=att_row[7],
            )

        # Return the one with higher reaction count
        if top_message and top_attachment:
            if top_attachment.reaction_count > top_message.reaction_count:
                return (None, top_attachment)
            else:
                return (top_message, None)
        elif top_message:
            return (top_message, None)
        elif top_attachment:
            return (None, top_attachment)
        else:
            return (None, None)
