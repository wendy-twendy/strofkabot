import aiosqlite
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import numpy as np
import logging

class UserStats:
    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None
        self.logger = logger or logging.getLogger('UserStats')

    async def initialize(self):
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            self.logger.info("UserStats database initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing UserStats database: {e}")
            raise

    async def _create_tables(self):
        # Use executescript to run multiple SQL statements
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

            CREATE TABLE IF NOT EXISTS user_reactions (
                giver_id INTEGER,
                receiver_id INTEGER,
                reaction_count INTEGER,
                PRIMARY KEY (giver_id, receiver_id)
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
        if self.conn is None:
            await self.initialize()

    async def update_stats(self, author_id: int, reaction_count: int, timestamp: datetime):
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

    async def batch_update_stats(self, stats: List[Tuple[int, int, datetime]]):
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_stats_monthly (author_id, year, month, total_messages, total_reactions)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(author_id, year, month) DO UPDATE SET
                total_messages = total_messages + 1,
                total_reactions = total_reactions + excluded.total_reactions
        ''', [(author_id, timestamp.year, timestamp.month, reaction_count) for author_id, reaction_count, timestamp in stats]):
            await self.conn.commit()

    async def update_user_mapping(self, author_id: int, username: str):
        await self.ensure_connection()
        current_time = datetime.now()
        async with self.conn.execute('''
            INSERT OR REPLACE INTO user_mapping (author_id, username, last_updated)
            VALUES (?, ?, ?)
        ''', (author_id, username, current_time)):
            await self.conn.commit()

    async def get_stats(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[dict]:
        await self.ensure_connection()
        query = '''
            SELECT s.author_id, m.username, SUM(s.total_messages) as total_messages,
                   SUM(s.total_reactions) as total_reactions,
                   CAST(SUM(s.total_reactions) AS FLOAT) / SUM(s.total_messages) AS avg_reactions
            FROM user_stats_monthly s
            LEFT JOIN user_mapping m ON s.author_id = m.author_id
        '''
        params = []
        if start_date and end_date:
            query += '''
                WHERE (s.year > ? OR (s.year = ? AND s.month >= ?))
                  AND (s.year < ? OR (s.year = ? AND s.month <= ?))
            '''
            params = [start_date.year, start_date.year, start_date.month,
                      end_date.year, end_date.year, end_date.month]
        
        query += ' GROUP BY s.author_id ORDER BY avg_reactions DESC'
        
        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    'author_id': row[0],
                    'username': row[1],
                    'total_msgs': row[2],
                    'total_reacts': row[3],
                    'avg_reacts': row[4]
                }
                for row in rows
            ]
        
    async def get_monthly_stats(self, year: int, month: int) -> List[dict]:
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
            self.logger.debug(f"Processed monthly stats: {result[:3]}...")  # Log first 3 entries
            return result

    async def get_user_monthly_stats(self, author_id: int, year: int, month: int) -> Optional[dict]:
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
        await self.ensure_connection()
        year, month = timestamp.year, timestamp.month
        async with self.conn.execute('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', (giver_id, receiver_id, year, month)):
            await self.conn.commit()

    async def batch_update_reaction_stats(self, stats: List[Tuple[int, int, datetime]]):
        await self.ensure_connection()
        async with self.conn.executemany('''
            INSERT INTO user_reactions_monthly (giver_id, receiver_id, year, month, reaction_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(giver_id, receiver_id, year, month) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', [(giver_id, receiver_id, timestamp.year, timestamp.month) for giver_id, receiver_id, timestamp in stats]):
            await self.conn.commit()

    async def get_reaction_graph(self, server_members: List[int]) -> np.ndarray:
        await self.ensure_connection()
        member_count = len(server_members)
        reaction_graph = np.zeros((member_count, member_count), dtype=int)
        
        placeholders = ','.join('?' * member_count)
        query = f'''
            SELECT giver_id, receiver_id, reaction_count
            FROM user_reactions
            WHERE giver_id IN ({placeholders}) AND receiver_id IN ({placeholders})
        '''
        
        async with self.conn.execute(query, server_members + server_members) as cursor:
            async for row in cursor:
                try:
                    giver_index = server_members.index(row[0])
                    receiver_index = server_members.index(row[1])
                    reaction_graph[giver_index][receiver_index] = row[2]
                except ValueError:
                    # Skip if the member is not in the server_members list
                    continue
        
        return reaction_graph

    async def reset_stats(self):
        """Resets the user_stats_monthly and user_reactions tables."""
        await self.ensure_connection()
        self.logger.info("Resetting user_stats_monthly and user_reactions tables.")
        try:
            await self.conn.execute('DELETE FROM user_stats_monthly;')
            await self.conn.execute('DELETE FROM user_reactions;')
            await self.conn.commit()
            self.logger.info("Successfully reset user statistics and reaction data.")
        except Exception as e:
            self.logger.error(f"Error resetting statistics: {e}")
            raise

    async def get_active_users(self, year: int, month: int, min_messages: int = 30) -> List[int]:
        """Retrieve users with at least min_messages in the specified month."""
        await self.ensure_connection()
        self.logger.info(f"Fetching active users for {year}-{month} with at least {min_messages} messages.")
        query = '''
            SELECT author_id
            FROM user_stats_monthly
            WHERE year = ? AND month = ? AND total_messages >= ?
        '''
        async with self.conn.execute(query, (year, month, min_messages)) as cursor:
            rows = await cursor.fetchall()
            active_users = [row[0] for row in rows]
            self.logger.debug(f"Found {len(active_users)} active users.")
            return active_users

    async def get_reaction_inflation_raw(self, monthly=False, limit=None):
        """Retrieve raw data for reaction inflation calculations."""
        await self.ensure_connection()
        records = []
        if monthly:
            self.logger.info("Fetching raw monthly data for reaction inflation.")
            query = '''
                SELECT 
                    year, 
                    month,
                    AVG(CAST(total_reactions AS FLOAT) / NULLIF(total_messages, 0)) as average_rpm
                FROM user_stats_monthly
                GROUP BY year, month
                ORDER BY year DESC, month DESC
                LIMIT ?
            '''
            async with self.conn.execute(query, (limit or -1,)) as cursor:
                rows = await cursor.fetchall()
                records = [
                    {
                        'year': row[0],
                        'month': row[1],
                        'average_rpm': row[2]
                    }
                    for row in rows
                ]
        else:
            self.logger.info("Fetching raw yearly data for reaction inflation.")
            query = '''
                SELECT 
                    year, 
                    SUM(total_reactions) AS total_reactions,
                    SUM(total_messages) AS total_messages,
                    CAST(SUM(total_reactions) AS FLOAT) / NULLIF(SUM(total_messages), 0) AS average_rpm
                FROM user_stats_monthly
                GROUP BY year
                ORDER BY year
            '''
            async with self.conn.execute(query) as cursor:
                rows = await cursor.fetchall()
                records = [
                    {
                        'year': row[0],
                        'total_reactions': row[1],
                        'total_messages': row[2],
                        'average_rpm': row[3]
                    }
                    for row in rows
                ]
        self.logger.debug(f"Fetched raw inflation records: {records[:3]}...")
        return records


    async def get_reaction_inflation_raw_new(self, monthly=False, limit=None):
        """Retrieve raw data for reaction inflation calculations."""
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
                        'average_rpm': row[2] / row[3] if row[3] else 0  # Avoid division by zero
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
                        'average_rpm': row[1] / row[2] if row[2] else 0  # Avoid division by zero
                    }
                    for row in rows
                ]
        self.logger.debug(f"Fetched raw inflation records: {records[:3]}...")
        return records

    async def get_reactions_given_received(self, member_ids: List[int]) -> Dict[int, Dict[str, int]]:
        """Retrieve the number of reactions given and received for each member."""
        await self.ensure_connection()
        self.logger.info("Fetching reactions given and received for clustering.")
        reactions_data = {user_id: {'given': 0, 'received': 0} for user_id in member_ids}

        # Fetch reactions given
        placeholders = ','.join('?' * len(member_ids))
        query_given = f'''
            SELECT giver_id, SUM(reaction_count) as total_given
            FROM user_reactions
            WHERE giver_id IN ({placeholders})
            GROUP BY giver_id
        '''
        async with self.conn.execute(query_given, member_ids) as cursor:
            async for row in cursor:
                giver_id, total_given = row
                if giver_id in reactions_data:
                    reactions_data[giver_id]['given'] = total_given

        # Fetch reactions received
        query_received = f'''
            SELECT receiver_id, SUM(reaction_count) as total_received
            FROM user_reactions
            WHERE receiver_id IN ({placeholders})
            GROUP BY receiver_id
        '''
        async with self.conn.execute(query_received, member_ids) as cursor:
            async for row in cursor:
                receiver_id, total_received = row
                if receiver_id in reactions_data:
                    reactions_data[receiver_id]['received'] = total_received

        self.logger.debug(f"Reactions data sample: {list(reactions_data.items())[:3]}")
        return reactions_data

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
        self.logger.info("UserStats database connection closed.")