# db/stats.py

"""User statistics and analytics operations mixin for the database."""

import datetime


class StatsMixin:
    """Mixin providing user stats, reactions, and analytics database operations."""

    async def upsert_user_stats(self, author_id: int, year: int, month: int, reaction_count: int):
        """Insert or update user stats for a month."""
        await self.ensure_connection()
        async with self.conn.execute(
            """
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        """,
            (author_id, year, month, reaction_count),
        ):
            await self.conn.commit()

    async def batch_upsert_user_stats(self, stats: list[tuple[int, int, int, int]]):
        """Batch insert/update user stats. Each tuple: (author_id, year, month, reaction_count)."""
        await self.ensure_connection()
        async with self.conn.executemany(
            """
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        """,
            stats,
        ):
            await self.conn.commit()

    async def upsert_user_mapping(self, author_id: int, username: str):
        """Insert or replace user mapping."""
        await self.ensure_connection()
        current_time = datetime.datetime.now()
        async with self.conn.execute(
            """
            INSERT OR REPLACE INTO user_mapping (author_id, username, last_updated)
            VALUES (?, ?, ?)
        """,
            (author_id, username, current_time),
        ):
            await self.conn.commit()

    async def get_username_by_id(self, author_id: int) -> str | None:
        """Get username from user_mapping table by author_id.

        Args:
            author_id: The Discord user ID to look up.

        Returns:
            The username if found, None otherwise.
        """
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")

        async with self.conn.execute(
            "SELECT username FROM user_mapping WHERE author_id = ?",
            (author_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def upsert_reaction_stats(self, giver_id: int, receiver_id: int, year: int, month: int):
        """Insert or update reaction stats between two users."""
        await self.ensure_connection()
        async with self.conn.execute(
            """
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        """,
            (giver_id, receiver_id, year, month),
        ):
            await self.conn.commit()

    async def batch_upsert_reaction_stats(self, stats: list[tuple[int, int, int, int]]):
        """Batch insert/update reaction stats. Each tuple: (giver_id, receiver_id, year, month)."""
        await self.ensure_connection()
        async with self.conn.executemany(
            """
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        """,
            stats,
        ):
            await self.conn.commit()

    async def batch_upsert_reply_stats(self, stats: list[tuple[int, int, int, int]]):
        """Batch insert/update reply stats. Each tuple: (replier_id, replied_to_id, year, month)."""
        await self.ensure_connection()
        async with self.conn.executemany(
            """
            INSERT INTO user_replies_monthly (replier_id, replied_to_id, year, month, reply_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(replier_id, replied_to_id, year, month) DO UPDATE SET
                reply_count = reply_count + 1
        """,
            stats,
        ):
            await self.conn.commit()

    async def delete_all_user_stats(self):
        """Delete all user statistics and reaction data."""
        await self.ensure_connection()
        await self.conn.execute("DELETE FROM user_stats_monthly;")
        await self.conn.execute("DELETE FROM user_reactions_monthly;")
        await self.conn.commit()

    async def fetch_monthly_stats(self, year: int, month: int) -> list[tuple]:
        """Fetch all user statistics for a specific month."""
        await self.ensure_connection()
        query = """
            SELECT s.author_id, m.username, s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / NULLIF(s.total_messages, 0) AS avg_reactions
            FROM user_stats_monthly s
            LEFT JOIN user_mapping m ON s.author_id = m.author_id
            WHERE s.year = ? AND s.month = ?
            ORDER BY avg_reactions DESC
        """
        async with self.conn.execute(query, (year, month)) as cursor:
            return await cursor.fetchall()

    async def fetch_user_monthly_stats(self, author_id: int, year: int, month: int) -> tuple | None:
        """Fetch statistics for a specific user in a specific month."""
        await self.ensure_connection()
        async with self.conn.execute(
            """
            SELECT s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / NULLIF(s.total_messages, 0) AS avg_reactions
            FROM user_stats_monthly s
            WHERE s.author_id = ? AND s.year = ? AND s.month = ?
        """,
            (author_id, year, month),
        ) as cursor:
            return await cursor.fetchone()

    async def fetch_inflation_data(
        self, monthly: bool = False, limit: int | None = None
    ) -> list[tuple]:
        """Fetch raw data for reaction inflation calculations."""
        await self.ensure_connection()

        if monthly:
            query = """
                SELECT
                    year,
                    month,
                    COALESCE(SUM(total_reactions), 0) as total_reactions,
                    COALESCE(SUM(total_messages), 0) as total_messages
                FROM user_stats_monthly
                GROUP BY year, month
                ORDER BY year DESC, month DESC
                LIMIT ?
            """
            async with self.conn.execute(query, (limit or -1,)) as cursor:
                return await cursor.fetchall()
        else:
            query = """
                SELECT
                    year,
                    COALESCE(SUM(total_reactions), 0) as total_reactions,
                    COALESCE(SUM(total_messages), 0) AS total_messages
                FROM user_stats_monthly
                GROUP BY year
                ORDER BY year
            """
            async with self.conn.execute(query) as cursor:
                return await cursor.fetchall()

    async def fetch_gdp_data(self, limit: int | None = 24) -> list[tuple]:
        """Fetch total messages per month for GDP calculation.

        Args:
            limit: Maximum number of months to return. If None, returns all data.
        """
        await self.ensure_connection()
        if limit is None:
            query = """
                SELECT year, month, SUM(total_messages) as total_messages
                FROM user_stats_monthly
                GROUP BY year, month
                ORDER BY year DESC, month DESC
            """
            async with self.conn.execute(query) as cursor:
                return await cursor.fetchall()
        else:
            query = """
                SELECT year, month, SUM(total_messages) as total_messages
                FROM user_stats_monthly
                GROUP BY year, month
                ORDER BY year DESC, month DESC
                LIMIT ?
            """
            async with self.conn.execute(query, (limit,)) as cursor:
                return await cursor.fetchall()

    async def fetch_hdi_data(self, limit: int = 24) -> list[tuple]:
        """Fetch HDI data (quality messages / total messages) per month."""
        await self.ensure_connection()
        query = """
            WITH quality_messages AS (
                SELECT
                    strftime('%Y', timestamp) as year,
                    strftime('%m', timestamp) as month,
                    COUNT(*) as quality_count
                FROM messages
                GROUP BY year, month
            ),
            total_messages AS (
                SELECT
                    year,
                    month,
                    SUM(total_messages) as total_count
                FROM user_stats_monthly
                GROUP BY year, month
            )
            SELECT
                q.year,
                q.month,
                q.quality_count,
                t.total_count,
                CAST(q.quality_count AS FLOAT) / NULLIF(t.total_count, 0) as hdi_ratio
            FROM quality_messages q
            JOIN total_messages t ON q.year = t.year AND q.month = t.month
            ORDER BY q.year DESC, q.month DESC
            LIMIT ?
        """
        async with self.conn.execute(query, (limit,)) as cursor:
            return await cursor.fetchall()

    async def fetch_trade_data(self, user_id: int, year: int, month: int, limit: int = 5) -> dict:
        """Fetch reaction trade data for a specific user from a given date."""
        await self.ensure_connection()

        export_query = """
            SELECT receiver_id, SUM(reaction_count) as total_given
            FROM user_reactions_monthly
            WHERE giver_id = ? AND (year > ? OR (year = ? AND month >= ?))
            GROUP BY receiver_id
            ORDER BY total_given DESC
            LIMIT ?
        """

        import_query = """
            SELECT giver_id, SUM(reaction_count) as total_received
            FROM user_reactions_monthly
            WHERE receiver_id = ? AND (year > ? OR (year = ? AND month >= ?))
            GROUP BY giver_id
            ORDER BY total_received DESC
            LIMIT ?
        """

        total_query = """
            SELECT
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE giver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_given,
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE receiver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_received
        """

        async with self.conn.execute(export_query, (user_id, year, year, month, limit)) as cursor:
            export_data = await cursor.fetchall()

        async with self.conn.execute(import_query, (user_id, year, year, month, limit)) as cursor:
            import_data = await cursor.fetchall()

        async with self.conn.execute(
            total_query, (user_id, year, year, month, user_id, year, year, month)
        ) as cursor:
            total_data = await cursor.fetchone()

        if total_data is None:
            total_given, total_received = 0, 0
        else:
            total_given, total_received = total_data
        return {
            "exports": export_data,
            "imports": import_data,
            "total_given": total_given,
            "total_received": total_received,
            "trade_balance": total_received - total_given,
        }

    async def fetch_trade_data_for_month(
        self, user_id: int, year: int, month: int, limit: int = 5
    ) -> dict:
        """Fetch reaction trade data for a specific user for a single month.

        Args:
            user_id: The Discord user ID.
            year: The year to query.
            month: The month to query.
            limit: Maximum number of top partners to return.

        Returns:
            Dictionary with exports, imports, total_given, total_received, and trade_balance.
        """
        await self.ensure_connection()

        export_query = """
            SELECT receiver_id, SUM(reaction_count) as total_given
            FROM user_reactions_monthly
            WHERE giver_id = ? AND year = ? AND month = ?
            GROUP BY receiver_id
            ORDER BY total_given DESC
            LIMIT ?
        """

        import_query = """
            SELECT giver_id, SUM(reaction_count) as total_received
            FROM user_reactions_monthly
            WHERE receiver_id = ? AND year = ? AND month = ?
            GROUP BY giver_id
            ORDER BY total_received DESC
            LIMIT ?
        """

        total_query = """
            SELECT
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE giver_id = ? AND year = ? AND month = ?) as total_given,
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE receiver_id = ? AND year = ? AND month = ?) as total_received
        """

        async with self.conn.execute(export_query, (user_id, year, month, limit)) as cursor:
            export_data = await cursor.fetchall()

        async with self.conn.execute(import_query, (user_id, year, month, limit)) as cursor:
            import_data = await cursor.fetchall()

        async with self.conn.execute(
            total_query, (user_id, year, month, user_id, year, month)
        ) as cursor:
            total_data = await cursor.fetchone()

        if total_data is None:
            total_given, total_received = 0, 0
        else:
            total_given, total_received = total_data
        return {
            "exports": export_data,
            "imports": import_data,
            "total_given": total_given,
            "total_received": total_received,
            "trade_balance": total_received - total_given,
        }

    async def fetch_reaction_network(self, year: int, month: int) -> list[tuple]:
        """Fetch reaction network data for most-liked calculation."""
        await self.ensure_connection()
        query = """
            SELECT
                um_giver.username AS giver_username,
                um_receiver.username AS receiver_username,
                urm.reaction_count,
                usm_giver.total_messages AS giver_messages,
                usm_receiver.total_messages AS receiver_messages
            FROM user_reactions_monthly urm
            JOIN user_mapping um_giver ON urm.giver_id = um_giver.author_id
            JOIN user_mapping um_receiver ON urm.receiver_id = um_receiver.author_id
            JOIN user_stats_monthly usm_giver
                ON urm.giver_id = usm_giver.author_id
                AND urm.year = usm_giver.year
                AND urm.month = usm_giver.month
            JOIN user_stats_monthly usm_receiver
                ON urm.receiver_id = usm_receiver.author_id
                AND urm.year = usm_receiver.year
                AND urm.month = usm_receiver.month
            WHERE urm.year = ? AND urm.month = ?
        """
        async with self.conn.execute(query, (year, month)) as cursor:
            return await cursor.fetchall()

    async def fetch_reaction_network_rolling(
        self, start_year: int, start_month: int
    ) -> list[tuple]:
        """Fetch aggregated reaction data from start date to previous month.

        Args:
            start_year: Start year for the rolling window.
            start_month: Start month for the rolling window.

        Returns:
            List of (giver_username, receiver_username, total_reaction_count) tuples.
        """
        await self.ensure_connection()
        query = """
            SELECT
                um_giver.username AS giver_username,
                um_receiver.username AS receiver_username,
                SUM(urm.reaction_count) AS total_reactions
            FROM user_reactions_monthly urm
            JOIN user_mapping um_giver ON urm.giver_id = um_giver.author_id
            JOIN user_mapping um_receiver ON urm.receiver_id = um_receiver.author_id
            WHERE ((urm.year > ?) OR (urm.year = ? AND urm.month >= ?))
              AND urm.giver_id != urm.receiver_id
            GROUP BY urm.giver_id, urm.receiver_id
        """
        async with self.conn.execute(query, (start_year, start_year, start_month)) as cursor:
            return await cursor.fetchall()

    async def fetch_reply_network_for_months(
        self, year_months: list[tuple[int, int]]
    ) -> list[tuple]:
        """Fetch aggregated reply data for specific months.

        Args:
            year_months: List of (year, month) tuples to include.

        Returns:
            List of (replier_username, replied_to_username, total_reply_count) tuples.
        """
        if not year_months:
            return []

        await self.ensure_connection()

        # Build WHERE clause for year/month pairs
        conditions = " OR ".join(["(urm.year = ? AND urm.month = ?)"] * len(year_months))
        params = [v for ym in year_months for v in ym]

        query = f"""
            SELECT
                um_replier.username AS replier_username,
                um_replied.username AS replied_to_username,
                SUM(urm.reply_count) AS total_replies
            FROM user_replies_monthly urm
            JOIN user_mapping um_replier ON urm.replier_id = um_replier.author_id
            JOIN user_mapping um_replied ON urm.replied_to_id = um_replied.author_id
            WHERE ({conditions})
              AND urm.replier_id != urm.replied_to_id
            GROUP BY urm.replier_id, urm.replied_to_id
        """
        async with self.conn.execute(query, params) as cursor:
            return await cursor.fetchall()

    async def fetch_user_reaction_distribution_rolling(
        self, user_id: int, start_year: int, start_month: int
    ) -> dict:
        """Fetch complete reaction distribution for a user over a rolling window.

        Args:
            user_id: The Discord user ID.
            start_year: Start year for the rolling window.
            start_month: Start month for the rolling window.

        Returns:
            Dictionary with:
            - outgoing: list of (receiver_id, total_count) tuples
            - incoming: list of (giver_id, total_count) tuples
        """
        await self.ensure_connection()

        # Get all outgoing reactions (where user is the giver)
        outgoing_query = """
            SELECT receiver_id, SUM(reaction_count) as total_count
            FROM user_reactions_monthly
            WHERE giver_id = ?
              AND ((year > ?) OR (year = ? AND month >= ?))
              AND giver_id != receiver_id
            GROUP BY receiver_id
            ORDER BY total_count DESC
        """

        # Get all incoming reactions (where user is the receiver)
        incoming_query = """
            SELECT giver_id, SUM(reaction_count) as total_count
            FROM user_reactions_monthly
            WHERE receiver_id = ?
              AND ((year > ?) OR (year = ? AND month >= ?))
              AND giver_id != receiver_id
            GROUP BY giver_id
            ORDER BY total_count DESC
        """

        async with self.conn.execute(
            outgoing_query, (user_id, start_year, start_year, start_month)
        ) as cursor:
            outgoing = await cursor.fetchall()

        async with self.conn.execute(
            incoming_query, (user_id, start_year, start_year, start_month)
        ) as cursor:
            incoming = await cursor.fetchall()

        return {
            "outgoing": list(outgoing),
            "incoming": list(incoming),
        }
