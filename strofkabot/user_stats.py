"""User statistics business logic layer."""

import logging
from datetime import datetime

from strofkabot.discord_db import Database


class UserStats:
    """Business logic for user statistics and reaction data."""

    def __init__(self, db: Database, logger: logging.Logger | None = None):
        self.db = db
        self.logger = logger or logging.getLogger("UserStats")

    async def update_stats(self, author_id: int, reaction_count: int, timestamp: datetime):
        """Update message and reaction stats for a user."""
        year, month = timestamp.year, timestamp.month
        await self.db.upsert_user_stats(author_id, year, month, reaction_count)

    async def batch_update_stats(self, stats: list[tuple[int, int, datetime]]):
        """Batch update message and reaction stats for multiple users."""
        db_stats = [
            (author_id, timestamp.year, timestamp.month, reaction_count)
            for author_id, reaction_count, timestamp in stats
        ]
        await self.db.batch_upsert_user_stats(db_stats)

    async def update_user_mapping(self, author_id: int, username: str):
        """Update the username mapping for a user."""
        await self.db.upsert_user_mapping(author_id, username)

    async def get_monthly_stats(self, year: int, month: int) -> list[dict]:
        """Get all user statistics for a specific month."""
        self.logger.info(f"Fetching monthly stats for {year}-{month}")
        rows = await self.db.fetch_monthly_stats(year, month)
        self.logger.debug(f"Fetched {len(rows)} rows for monthly stats")
        result = [
            {
                "author_id": row[0],
                "username": row[1],
                "total_msgs": row[2],
                "total_reacts": row[3],
                "avg_reacts": row[4],
            }
            for row in rows
        ]
        self.logger.debug(f"Processed monthly stats: {result[:3]}...")
        return result

    async def get_user_monthly_stats(self, author_id: int, year: int, month: int) -> dict | None:
        """Get statistics for a specific user in a specific month."""
        self.logger.info(f"Fetching monthly stats for user {author_id} in {year}-{month}")
        row = await self.db.fetch_user_monthly_stats(author_id, year, month)
        if row:
            result = {"total_msgs": row[0], "total_reacts": row[1], "avg_reacts": row[2]}
            self.logger.debug(f"User monthly stats: {result}")
            return result
        self.logger.warning(f"No monthly stats found for user {author_id} in {year}-{month}")
        return None

    async def update_reaction_stats(self, giver_id: int, receiver_id: int, timestamp: datetime):
        """Update reaction stats between two users."""
        year, month = timestamp.year, timestamp.month
        await self.db.upsert_reaction_stats(giver_id, receiver_id, year, month)

    async def batch_update_reaction_stats(self, stats: list[tuple[int, int, datetime]]):
        """Batch update reaction stats for multiple user pairs."""
        db_stats = [
            (giver_id, receiver_id, timestamp.year, timestamp.month)
            for giver_id, receiver_id, timestamp in stats
        ]
        await self.db.batch_upsert_reaction_stats(db_stats)

    async def reset_stats(self):
        """Reset all user statistics and reaction data."""
        self.logger.info("Resetting user_stats_monthly and user_reactions_monthly tables.")
        try:
            await self.db.delete_all_user_stats()
            self.logger.info("Successfully reset user statistics and reaction data.")
        except Exception as e:
            self.logger.error(f"Error resetting statistics: {e}")
            raise

    async def get_reaction_inflation_raw(
        self, monthly: bool = False, limit: int = None
    ) -> list[dict]:
        """Retrieve raw data for reaction inflation calculations.

        Args:
            monthly: If True, return monthly data. If False, return yearly data.
            limit: Maximum number of records to return (for monthly data).

        Returns:
            List of dictionaries with year, month (if monthly), total_reactions,
            total_messages, and average_rpm.
        """
        if monthly:
            self.logger.info("Fetching raw monthly data for reaction inflation.")
        else:
            self.logger.info("Fetching raw yearly data for reaction inflation.")

        rows = await self.db.fetch_inflation_data(monthly, limit)

        if monthly:
            records = [
                {
                    "year": row[0],
                    "month": row[1],
                    "total_reactions": row[2],
                    "total_messages": row[3],
                    "average_rpm": row[2] / row[3] if row[3] else 0,
                }
                for row in rows
            ]
        else:
            records = [
                {
                    "year": row[0],
                    "total_reactions": row[1],
                    "total_messages": row[2],
                    "average_rpm": row[1] / row[2] if row[2] else 0,
                }
                for row in rows
            ]

        self.logger.debug(f"Fetched raw inflation records: {records[:3]}...")
        return records

    async def get_gdp_data(self, limit: int | None = 24) -> list[dict]:
        """Get total messages per month for GDP calculation.

        Args:
            limit: Maximum number of months to return. If None, returns all data.

        Returns:
            List of dictionaries with year, month, and total_messages.
        """
        self.logger.info(f"Fetching GDP data (limit={limit})")
        rows = await self.db.fetch_gdp_data(limit)
        return [{"year": row[0], "month": row[1], "total_messages": row[2]} for row in rows]

    async def get_hdi_data(self, limit: int = 24) -> list[dict]:
        """Get HDI data (quality messages / total messages) per month.

        Args:
            limit: Maximum number of months to return.

        Returns:
            List of dictionaries with year, month, quality_count, total_count, and hdi_ratio.
        """
        self.logger.info(f"Fetching HDI data (limit={limit})")
        rows = await self.db.fetch_hdi_data(limit)
        return [
            {
                "year": int(row[0]),
                "month": int(row[1]),
                "quality_count": row[2],
                "total_count": row[3],
                "hdi_ratio": row[4],
            }
            for row in rows
        ]

    async def get_reaction_trade_data(
        self, user_id: int, year: int, month: int, limit: int = 5
    ) -> dict:
        """Get reaction trade data for a specific user from a given date.

        Args:
            user_id: The Discord user ID.
            year: Start year for the query.
            month: Start month for the query.
            limit: Maximum number of top users to return.

        Returns:
            Dictionary with exports, imports, total_given, total_received, and trade_balance.
        """
        self.logger.info(f"Fetching trade data for user {user_id} from {year}-{month:02d}")
        data = await self.db.fetch_trade_data(user_id, year, month, limit)
        # Convert tuples to the expected format
        return {
            "exports": [(row[0], row[1]) for row in data["exports"]],
            "imports": [(row[0], row[1]) for row in data["imports"]],
            "total_given": data["total_given"],
            "total_received": data["total_received"],
            "trade_balance": data["trade_balance"],
        }

    async def get_reaction_trade_data_for_month(
        self, user_id: int, year: int, month: int, limit: int = 5
    ) -> dict:
        """Get reaction trade data for a specific user for a single month.

        Args:
            user_id: The Discord user ID.
            year: The year to query.
            month: The month to query.
            limit: Maximum number of top users to return.

        Returns:
            Dictionary with exports, imports, total_given, total_received, and trade_balance.
        """
        self.logger.info(f"Fetching trade data for user {user_id} for {year}-{month:02d}")
        data = await self.db.fetch_trade_data_for_month(user_id, year, month, limit)
        # Convert tuples to the expected format
        return {
            "exports": [(row[0], row[1]) for row in data["exports"]],
            "imports": [(row[0], row[1]) for row in data["imports"]],
            "total_given": data["total_given"],
            "total_received": data["total_received"],
            "trade_balance": data["trade_balance"],
        }

    async def get_reaction_network_for_month(self, year: int, month: int) -> list[dict]:
        """Get reaction network data for most-liked calculation.

        Args:
            year: The year to query.
            month: The month to query.

        Returns:
            List of dictionaries with giver/receiver usernames, reaction counts, and message counts.
        """
        self.logger.info(f"Fetching reaction network for {year}-{month:02d}")
        rows = await self.db.fetch_reaction_network(year, month)
        return [
            {
                "giver_username": row[0],
                "receiver_username": row[1],
                "reaction_count": row[2],
                "giver_messages": row[3],
                "receiver_messages": row[4],
            }
            for row in rows
        ]

    async def get_reaction_network_rolling(
        self, start_year: int, start_month: int
    ) -> list[tuple[str, str, int]]:
        """Get aggregated reaction network for a rolling window.

        Args:
            start_year: Start year for the rolling window.
            start_month: Start month for the rolling window.

        Returns:
            List of (giver_username, receiver_username, reaction_count) tuples.
        """
        self.logger.info(f"Fetching reaction network from {start_year}-{start_month:02d}")
        rows = await self.db.fetch_reaction_network_rolling(start_year, start_month)
        return [(row[0], row[1], row[2]) for row in rows]
