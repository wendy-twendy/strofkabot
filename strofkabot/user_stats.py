import aiosqlite
from typing import List, Tuple, Optional
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
                total_reactions = total_reactions + ?
        ''', (author_id, year, month, reaction_count, reaction_count)):
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
        async with self.conn.execute('''
            SELECT s.author_id, m.username, s.total_messages, s.total_reactions,
                   CAST(s.total_reactions AS FLOAT) / s.total_messages AS avg_reactions
            FROM user_stats_monthly s
            LEFT JOIN user_mapping m ON s.author_id = m.author_id
            WHERE s.year = ? AND s.month = ?
            ORDER BY avg_reactions DESC
        ''', (year, month)) as cursor:
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

    async def update_reaction_stats(self, giver_id: int, receiver_id: int):
        await self.ensure_connection()
        async with self.conn.execute('''
            INSERT INTO user_reactions (giver_id, receiver_id, reaction_count)
            VALUES (?, ?, 1)
            ON CONFLICT(giver_id, receiver_id) DO UPDATE SET
                reaction_count = reaction_count + 1
        ''', (giver_id, receiver_id)):
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
                giver_index = server_members.index(row[0])
                receiver_index = server_members.index(row[1])
                reaction_graph[giver_index][receiver_index] = row[2]
        
        return reaction_graph

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None
        self.logger.info("UserStats database connection closed.")