# discord_db.py

"""Database access layer for all SQLite operations."""

import datetime
from dataclasses import dataclass
from pathlib import Path

import aiosqlite


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


class Database:
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
        await self.conn.executescript('''
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
        ''')
        await self.conn.commit()

    async def add_messages(self, messages: list[Message]):
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.executemany('''
            INSERT OR REPLACE INTO messages
            (id, content, timestamp, reaction_count, author_id, reply_to_id, reply_to_author, reply_to_content)
            VALUES (:id, :content, :timestamp, :reaction_count, :author_id, :reply_to_id, :reply_to_author, :reply_to_content)
        ''', [msg.__dict__ for msg in messages]):
            pass
        await self.conn.commit()

    async def update_last_scanned_timestamp(self, channel_id: int, timestamp: datetime.datetime):
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        key = f"last_scanned_timestamp_{channel_id}"
        await self.conn.execute('''
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        ''', (key, timestamp.isoformat()))
        await self.conn.commit()

    async def get_last_scanned_timestamp(self, channel_id: int) -> datetime.datetime | None:
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        key = f"last_scanned_timestamp_{channel_id}"
        async with self.conn.execute('SELECT value FROM metadata WHERE key = ?', (key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return datetime.datetime.fromisoformat(row[0])
            return None

    async def get_random_message(self) -> Message | None:
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        async with self.conn.execute('SELECT * FROM messages ORDER BY RANDOM() LIMIT 1') as cursor:
            row = await cursor.fetchone()
            if row:
                return Message(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.datetime.fromisoformat(row[2]),
                    reaction_count=row[3],
                    author_id=row[4],
                    reply_to_id=row[5],
                    reply_to_author=row[6],
                    reply_to_content=row[7]
                )
            return None

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    # =========================================================================
    # User Stats Operations
    # =========================================================================

    async def upsert_user_stats(
        self, author_id: int, year: int, month: int, reaction_count: int
    ):
        """Insert or update user stats for a month."""
        await self.ensure_connection()
        async with self.conn.execute('''
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        ''', (author_id, year, month, reaction_count)):
            await self.conn.commit()

    async def batch_upsert_user_stats(
        self, stats: list[tuple[int, int, int, int]]
    ):
        """Batch insert/update user stats. Each tuple: (author_id, year, month, reaction_count)."""
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        ''', stats):
            await self.conn.commit()

    async def upsert_user_mapping(self, author_id: int, username: str):
        """Insert or replace user mapping."""
        await self.ensure_connection()
        current_time = datetime.datetime.now()
        async with self.conn.execute('''
            INSERT OR REPLACE INTO user_mapping (author_id, username, last_updated)
            VALUES (?, ?, ?)
        ''', (author_id, username, current_time)):
            await self.conn.commit()

    async def upsert_reaction_stats(
        self, giver_id: int, receiver_id: int, year: int, month: int
    ):
        """Insert or update reaction stats between two users."""
        await self.ensure_connection()
        async with self.conn.execute('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', (giver_id, receiver_id, year, month)):
            await self.conn.commit()

    async def batch_upsert_reaction_stats(
        self, stats: list[tuple[int, int, int, int]]
    ):
        """Batch insert/update reaction stats. Each tuple: (giver_id, receiver_id, year, month)."""
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', stats):
            await self.conn.commit()

    async def delete_all_user_stats(self):
        """Delete all user statistics and reaction data."""
        await self.ensure_connection()
        await self.conn.execute('DELETE FROM user_stats_monthly;')
        await self.conn.execute('DELETE FROM user_reactions_monthly;')
        await self.conn.commit()

    async def fetch_monthly_stats(self, year: int, month: int) -> list[tuple]:
        """Fetch all user statistics for a specific month."""
        await self.ensure_connection()
        query = '''
            SELECT s.author_id, m.username, s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / s.total_messages AS avg_reactions
            FROM user_stats_monthly s
            LEFT JOIN user_mapping m ON s.author_id = m.author_id
            WHERE s.year = ? AND s.month = ?
            ORDER BY avg_reactions DESC
        '''
        async with self.conn.execute(query, (year, month)) as cursor:
            return await cursor.fetchall()

    async def fetch_user_monthly_stats(
        self, author_id: int, year: int, month: int
    ) -> tuple | None:
        """Fetch statistics for a specific user in a specific month."""
        await self.ensure_connection()
        async with self.conn.execute('''
            SELECT s.total_messages, s.total_reactions,
                CAST(s.total_reactions AS FLOAT) / s.total_messages AS avg_reactions
            FROM user_stats_monthly s
            WHERE s.author_id = ? AND s.year = ? AND s.month = ?
        ''', (author_id, year, month)) as cursor:
            return await cursor.fetchone()

    async def fetch_inflation_data(
        self, monthly: bool = False, limit: int | None = None
    ) -> list[tuple]:
        """Fetch raw data for reaction inflation calculations."""
        await self.ensure_connection()

        if monthly:
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
                return await cursor.fetchall()
        else:
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
                return await cursor.fetchall()

    async def fetch_gdp_data(self, limit: int = 24) -> list[tuple]:
        """Fetch total messages per month for GDP calculation."""
        await self.ensure_connection()
        query = '''
            SELECT year, month, SUM(total_messages) as total_messages
            FROM user_stats_monthly
            GROUP BY year, month
            ORDER BY year DESC, month DESC
            LIMIT ?
        '''
        async with self.conn.execute(query, (limit,)) as cursor:
            return await cursor.fetchall()

    async def fetch_hdi_data(self, limit: int = 24) -> list[tuple]:
        """Fetch HDI data (quality messages / total messages) per month."""
        await self.ensure_connection()
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
            return await cursor.fetchall()

    async def fetch_trade_data(
        self, user_id: int, year: int, month: int, limit: int = 5
    ) -> dict:
        """Fetch reaction trade data for a specific user from a given date."""
        await self.ensure_connection()

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

        async with self.conn.execute(
            export_query, (user_id, year, year, month, limit)
        ) as cursor:
            export_data = await cursor.fetchall()

        async with self.conn.execute(
            import_query, (user_id, year, year, month, limit)
        ) as cursor:
            import_data = await cursor.fetchall()

        async with self.conn.execute(
            total_query, (user_id, year, year, month, user_id, year, year, month)
        ) as cursor:
            total_data = await cursor.fetchone()

        total_given, total_received = total_data
        return {
            'exports': export_data,
            'imports': import_data,
            'total_given': total_given,
            'total_received': total_received,
            'trade_balance': total_received - total_given
        }

    async def fetch_reaction_network(self, year: int, month: int) -> list[tuple]:
        """Fetch reaction network data for most-liked calculation."""
        await self.ensure_connection()
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
            return await cursor.fetchall()


# Backward compatibility alias
MessageDatabase = Database
