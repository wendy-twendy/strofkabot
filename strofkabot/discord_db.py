# discord_db.py

import aiosqlite
from pathlib import Path
import datetime
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Message:
    id: int
    content: str
    timestamp: datetime.datetime
    reaction_count: int
    author_id: int
    reply_to_id: Optional[int] = None
    reply_to_author: Optional[str] = None
    reply_to_content: Optional[str] = None

class MessageDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        if self.conn is None:
            self.conn = await aiosqlite.connect(str(self.db_path))
            await self._create_tables()

    async def ensure_connection(self):
            if self.conn is None:
                await self.initialize()

    async def _create_tables(self):
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                content TEXT,
                timestamp TEXT,
                reaction_count INTEGER,
                author_id INTEGER,
                reply_to_id INTEGER,
                reply_to_author TEXT,
                reply_to_content TEXT
            )
        ''')
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        await self.conn.commit()

    async def add_messages(self, messages: List[Message]):
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

    async def get_last_scanned_timestamp(self, channel_id: int) -> Optional[datetime.datetime]:
        await self.ensure_connection()
        if not self.conn:
            raise RuntimeError("Database not initialized.")
        key = f"last_scanned_timestamp_{channel_id}"
        async with self.conn.execute('SELECT value FROM metadata WHERE key = ?', (key,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return datetime.datetime.fromisoformat(row[0])
            return None

    async def get_random_message(self) -> Optional[Message]:
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