"""User statistics database access layer."""

import logging
from datetime import datetime

import aiosqlite


class UserStats:
    """Manages user statistics and reaction data in SQLite."""

    def __init__(self, db_path: str, logger: logging.Logger | None = None):
        self.db_path = db_path
        self.conn: aiosqlite.Connection | None = None
        self.logger = logger or logging.getLogger('UserStats')

    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            self.logger.info("UserStats database initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing UserStats database: {e}")
            raise

    async def _create_tables(self):
        """Create required database tables if they don't exist."""
        await self.conn.executescript('''
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
        ''')
        await self.conn.commit()

    async def ensure_connection(self):
        """Ensure database connection is active."""
        if self.conn is None:
            await self.initialize()

    async def update_stats(self, author_id: int, reaction_count: int, timestamp: datetime):
        """Update message and reaction stats for a user."""
        await self.ensure_connection()
        year, month = timestamp.year, timestamp.month
        async with self.conn.execute('''
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        ''', (author_id, year, month, reaction_count)):
            await self.conn.commit()

    async def batch_update_stats(self, stats: list[tuple[int, int, datetime]]):
        """Batch update message and reaction stats for multiple users."""
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        ''', [(author_id, timestamp.year, timestamp.month, reaction_count)
              for author_id, reaction_count, timestamp in stats]):
            await self.conn.commit()

    async def update_user_mapping(self, author_id: int, username: str):
        """Update the username mapping for a user."""
        await self.ensure_connection()
        current_time = datetime.now()
        async with self.conn.execute('''
            INSERT OR REPLACE INTO user_mapping (author_id, username, last_updated)
            VALUES (?, ?, ?)
        ''', (author_id, username, current_time)):
            await self.conn.commit()

    async def get_monthly_stats(self, year: int, month: int) -> list[dict]:
        """Get all user statistics for a specific month."""
        await self.ensure_connection()
        self.logger.info(f"Fetching monthly stats for {year}-{month}")
        query = '''
            SELECT s.author_id, m.username, s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / s.total_messages AS avg_reactions
            FROM user_stats_monthly s
            LEFT JOIN user_mapping m ON s.author_id = m.author_id
            WHERE s.year = ? AND s.month = ?
            ORDER BY avg_reactions DESC
        '''
        async with self.conn.execute(query, (year, month)) as cursor:
            rows = await cursor.fetchall()
            self.logger.debug(f"Fetched {len(rows)} rows for monthly stats")
            result = [
                {
                    'author_id': row[0],
                    'username': row[1],
                    'total_msgs': row[2],
                    'total_reacts': row[3],
                    'avg_reacts': row[4]
                }
                for row in rows
            ]
            self.logger.debug(f"Processed monthly stats: {result[:3]}...")
            return result

    async def get_user_monthly_stats(self, author_id: int, year: int, month: int) -> dict | None:
        """Get statistics for a specific user in a specific month."""
        await self.ensure_connection()
        self.logger.info(f"Fetching monthly stats for user {author_id} in {year}-{month}")
        async with self.conn.execute('''
            SELECT s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / s.total_messages AS avg_reactions
            FROM user_stats_monthly s
            WHERE s.author_id = ? AND s.year = ? AND s.month = ?
        ''', (author_id, year, month)) as cursor:
            row = await cursor.fetchone()
            if row:
                result = {
                    'total_msgs': row[0],
                    'total_reacts': row[1],
                    'avg_reacts': row[2]
                }
                self.logger.debug(f"User monthly stats: {result}")
                return result
            self.logger.warning(f"No monthly stats found for user {author_id} in {year}-{month}")
            return None

    async def update_reaction_stats(self, giver_id: int, receiver_id: int, timestamp: datetime):
        """Update reaction stats between two users."""
        await self.ensure_connection()
        year, month = timestamp.year, timestamp.month
        async with self.conn.execute('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', (giver_id, receiver_id, year, month)):
            await self.conn.commit()

    async def batch_update_reaction_stats(self, stats: list[tuple[int, int, datetime]]):
        """Batch update reaction stats for multiple user pairs."""
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', [(giver_id, receiver_id, timestamp.year, timestamp.month)
              for giver_id, receiver_id, timestamp in stats]):
            await self.conn.commit()

    async def reset_stats(self):
        """Reset all user statistics and reaction data."""
        await self.ensure_connection()
        self.logger.info("Resetting user_stats_monthly and user_reactions_monthly tables.")
        try:
            await self.conn.execute('DELETE FROM user_stats_monthly;')
            await self.conn.execute('DELETE FROM user_reactions_monthly;')
            await self.conn.commit()
            self.logger.info("Successfully reset user statistics and reaction data.")
        except Exception as e:
            self.logger.error(f"Error resetting statistics: {e}")
            raise

    async def get_reaction_inflation_raw(self, monthly: bool = False, limit: int = None) -> list[dict]:
        """Retrieve raw data for reaction inflation calculations.

        Args:
            monthly: If True, return monthly data. If False, return yearly data.
            limit: Maximum number of records to return (for monthly data).

        Returns:
            List of dictionaries with year, month (if monthly), total_reactions,
            total_messages, and average_rpm.
        """
        await self.ensure_connection()
        records = []

        if monthly:
            self.logger.info("Fetching raw monthly data for reaction inflation.")
            query = '''
                SELECT
                    usm.year,
                    usm.month,
                    COALESCE((SELECT SUM(reaction_count)
                     FROM user_reactions_monthly urm
                     WHERE urm.year = usm.year AND urm.month = usm.month), 0) as total_reactions,
                    COALESCE(SUM(usm.total_messages), 0) as total_messages
                FROM user_stats_monthly usm
                GROUP BY usm.year, usm.month
                ORDER BY usm.year DESC, usm.month DESC
                LIMIT ?
            '''
            async with self.conn.execute(query, (limit or -1,)) as cursor:
                rows = await cursor.fetchall()
                records = [
                    {
                        'year': row[0],
                        'month': row[1],
                        'total_reactions': row[2],
                        'total_messages': row[3],
                        'average_rpm': row[2] / row[3] if row[3] else 0
                    }
                    for row in rows
                ]
        else:
            self.logger.info("Fetching raw yearly data for reaction inflation.")
            query = '''
                SELECT
                    usm.year,
                    COALESCE((SELECT SUM(reaction_count)
                     FROM user_reactions_monthly urm
                     WHERE urm.year = usm.year), 0) as total_reactions,
                    COALESCE(SUM(usm.total_messages), 0) AS total_messages
                FROM user_stats_monthly usm
                GROUP BY usm.year
                ORDER BY usm.year
            '''
            async with self.conn.execute(query) as cursor:
                rows = await cursor.fetchall()
                records = [
                    {
                        'year': row[0],
                        'total_reactions': row[1],
                        'total_messages': row[2],
                        'average_rpm': row[1] / row[2] if row[2] else 0
                    }
                    for row in rows
                ]

        self.logger.debug(f"Fetched raw inflation records: {records[:3]}...")
        return records

    async def get_gdp_data(self, limit: int = 24) -> list[dict]:
        """Get total messages per month for GDP calculation.

        Args:
            limit: Maximum number of months to return.

        Returns:
            List of dictionaries with year, month, and total_messages.
        """
        await self.ensure_connection()
        self.logger.info(f"Fetching GDP data (limit={limit})")
        query = '''
            SELECT year, month, SUM(total_messages) as total_messages
            FROM user_stats_monthly
            GROUP BY year, month
            ORDER BY year DESC, month DESC
            LIMIT ?
        '''
        async with self.conn.execute(query, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [{'year': row[0], 'month': row[1], 'total_messages': row[2]} for row in rows]

    async def get_hdi_data(self, limit: int = 24) -> list[dict]:
        """Get HDI data (quality messages / total messages) per month.

        Args:
            limit: Maximum number of months to return.

        Returns:
            List of dictionaries with year, month, quality_count, total_count, and hdi_ratio.
        """
        await self.ensure_connection()
        self.logger.info(f"Fetching HDI data (limit={limit})")
        query = '''
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
        '''
        async with self.conn.execute(query, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [{
                'year': int(row[0]),
                'month': int(row[1]),
                'quality_count': row[2],
                'total_count': row[3],
                'hdi_ratio': row[4]
            } for row in rows]

    async def get_reaction_trade_data(
        self,
        user_id: int,
        year: int,
        month: int,
        limit: int = 5
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
        await self.ensure_connection()
        self.logger.info(f"Fetching trade data for user {user_id} from {year}-{month:02d}")

        export_query = '''
            SELECT receiver_id, SUM(reaction_count) as total_given
            FROM user_reactions_monthly
            WHERE giver_id = ? AND (year > ? OR (year = ? AND month >= ?))
            GROUP BY receiver_id
            ORDER BY total_given DESC
            LIMIT ?
        '''

        import_query = '''
            SELECT giver_id, SUM(reaction_count) as total_received
            FROM user_reactions_monthly
            WHERE receiver_id = ? AND (year > ? OR (year = ? AND month >= ?))
            GROUP BY giver_id
            ORDER BY total_received DESC
            LIMIT ?
        '''

        total_query = '''
            SELECT
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE giver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_given,
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE receiver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_received
        '''

        async with self.conn.execute(export_query, (user_id, year, year, month, limit)) as cursor:
            export_data = await cursor.fetchall()

        async with self.conn.execute(import_query, (user_id, year, year, month, limit)) as cursor:
            import_data = await cursor.fetchall()

        async with self.conn.execute(
            total_query, (user_id, year, year, month, user_id, year, year, month)
        ) as cursor:
            total_data = await cursor.fetchone()

        total_given, total_received = total_data
        trade_balance = total_received - total_given

        return {
            'exports': [(row[0], row[1]) for row in export_data],  # (receiver_id, count)
            'imports': [(row[0], row[1]) for row in import_data],  # (giver_id, count)
            'total_given': total_given,
            'total_received': total_received,
            'trade_balance': trade_balance
        }

    async def get_reaction_network_for_month(self, year: int, month: int) -> list[dict]:
        """Get reaction network data for most-liked calculation.

        Args:
            year: The year to query.
            month: The month to query.

        Returns:
            List of dictionaries with giver/receiver usernames, reaction counts, and message counts.
        """
        await self.ensure_connection()
        self.logger.info(f"Fetching reaction network for {year}-{month:02d}")

        query = '''
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
        '''

        async with self.conn.execute(query, (year, month)) as cursor:
            rows = await cursor.fetchall()
            return [{
                'giver_username': row[0],
                'receiver_username': row[1],
                'reaction_count': row[2],
                'giver_messages': row[3],
                'receiver_messages': row[4]
            } for row in rows]

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
        self.logger.info("UserStats database connection closed.")
