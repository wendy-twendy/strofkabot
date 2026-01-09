# db/predictions.py

"""Prediction operations mixin for the database."""

import datetime
from dataclasses import dataclass


@dataclass
class Prediction:
    id: int
    author_id: int
    author_name: str
    channel_id: int
    target_date: datetime.date
    prediction_text: str
    created_at: datetime.datetime
    posted: bool
    posted_at: datetime.datetime | None = None
    retry_count: int = 0


class PredictionsMixin:
    """Mixin providing prediction-related database operations."""

    async def add_prediction(
        self,
        author_id: int,
        author_name: str,
        channel_id: int,
        target_date: datetime.date,
        prediction_text: str,
    ) -> int:
        """Insert a new prediction and return its ID."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        created_at = datetime.datetime.now(datetime.UTC)
        async with self.conn.execute(
            """
            INSERT INTO predictions
            (author_id, author_name, channel_id, target_date, prediction_text, created_at, posted)
            VALUES (?, ?, ?, ?, ?, ?, FALSE)
            """,
            (
                author_id,
                author_name,
                channel_id,
                target_date.isoformat(),
                prediction_text,
                created_at.isoformat(),
            ),
        ) as cursor:
            prediction_id = cursor.lastrowid
        await self.conn.commit()
        return prediction_id

    async def get_due_predictions(
        self, target_date: datetime.date, max_retries: int = 5
    ) -> list[Prediction]:
        """Get all unposted predictions for target_date or earlier.

        Args:
            target_date: The date to check for due predictions.
            max_retries: Maximum retry attempts before a prediction is skipped.

        Returns:
            List of Prediction objects that are due and haven't exceeded retry limit.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        async with self.conn.execute(
            """
            SELECT id, author_id, author_name, channel_id, target_date,
                   prediction_text, created_at, posted, posted_at,
                   COALESCE(retry_count, 0) as retry_count
            FROM predictions
            WHERE target_date <= ? AND posted = FALSE
              AND COALESCE(retry_count, 0) < ?
            ORDER BY target_date ASC
            """,
            (target_date.isoformat(), max_retries),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            Prediction(
                id=row[0],
                author_id=row[1],
                author_name=row[2],
                channel_id=row[3],
                target_date=datetime.date.fromisoformat(row[4]),
                prediction_text=row[5],
                created_at=datetime.datetime.fromisoformat(row[6]),
                posted=bool(row[7]),
                posted_at=(datetime.datetime.fromisoformat(row[8]) if row[8] else None),
                retry_count=row[9],
            )
            for row in rows
        ]

    async def mark_prediction_posted(self, prediction_id: int) -> None:
        """Mark a prediction as posted."""
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        posted_at = datetime.datetime.now(datetime.UTC)
        await self.conn.execute(
            """
            UPDATE predictions
            SET posted = TRUE, posted_at = ?
            WHERE id = ?
            """,
            (posted_at.isoformat(), prediction_id),
        )
        await self.conn.commit()

    async def increment_prediction_retry(self, prediction_id: int) -> int:
        """Increment the retry count for a prediction.

        Returns:
            The new retry count.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        await self.conn.execute(
            """
            UPDATE predictions
            SET retry_count = COALESCE(retry_count, 0) + 1
            WHERE id = ?
            """,
            (prediction_id,),
        )
        await self.conn.commit()

        # Return the new retry count
        async with self.conn.execute(
            "SELECT retry_count FROM predictions WHERE id = ?",
            (prediction_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
