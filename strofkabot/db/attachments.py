# db/attachments.py

"""Attachment operations mixin for the database."""

import datetime
from dataclasses import dataclass


@dataclass
class Attachment:
    id: int
    message_id: int
    message_content: str | None
    author_id: int
    timestamp: datetime.datetime
    reaction_count: int
    original_filename: str
    local_path: str


class AttachmentsMixin:
    """Mixin providing attachment-related database operations."""

    async def add_attachments(self, attachments: list[Attachment]):
        """Batch insert attachments."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.executemany(
            """
            INSERT OR REPLACE INTO attachments
            (id, message_id, message_content, author_id, timestamp, reaction_count, original_filename, local_path)
            VALUES (:id, :message_id, :message_content, :author_id, :timestamp, :reaction_count, :original_filename, :local_path)
        """,
            [
                {
                    "id": att.id,
                    "message_id": att.message_id,
                    "message_content": att.message_content,
                    "author_id": att.author_id,
                    "timestamp": att.timestamp.isoformat(),
                    "reaction_count": att.reaction_count,
                    "original_filename": att.original_filename,
                    "local_path": att.local_path,
                }
                for att in attachments
            ],
        ):
            pass
        await self.conn.commit()

    async def get_random_attachment(self) -> Attachment | None:
        """Get a random attachment from the database."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute(
            "SELECT * FROM attachments ORDER BY RANDOM() LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Attachment(
                    id=row[0],
                    message_id=row[1],
                    message_content=row[2],
                    author_id=row[3],
                    timestamp=datetime.datetime.fromisoformat(row[4]),
                    reaction_count=row[5],
                    original_filename=row[6],
                    local_path=row[7],
                )
            return None

    async def get_attachment_count(self) -> int:
        """Get total number of attachments in the database."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute("SELECT COUNT(*) FROM attachments") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def attachment_exists(self, attachment_id: int) -> bool:
        """Check if an attachment already exists in the database."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute(
            "SELECT 1 FROM attachments WHERE id = ?", (attachment_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None
