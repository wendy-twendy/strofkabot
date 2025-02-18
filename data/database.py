import aiosqlite
import asyncio
from typing import Optional

class Database:
    def __init__(self):
        self.conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def ensure_connection(self):
        """Ensures database connection is established."""
        if self.conn is None:
            async with self._lock:
                if self.conn is None:  # Double-check pattern
                    self.conn = await aiosqlite.connect('strofkabot.db')

    async def execute(self, query: str, params: tuple = None):
        """Execute a query with parameters."""
        await self.ensure_connection()
        async with self._lock:
            return await self.conn.execute(query, params or ())

    async def fetchall(self, query: str, params: tuple = None):
        """Execute a query and fetch all results."""
        await self.ensure_connection()
        async with self._lock:
            async with self.conn.execute(query, params or ()) as cursor:
                return await cursor.fetchall()

    async def fetchone(self, query: str, params: tuple = None):
        """Execute a query and fetch one result."""
        await self.ensure_connection()
        async with self._lock:
            async with self.conn.execute(query, params or ()) as cursor:
                return await cursor.fetchone()

    async def close(self):
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
