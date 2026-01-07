from typing import List, Dict, Any, Optional
from datetime import datetime
from .database import Database

class UserStatsDAL:
    def __init__(self, db: Database):
        self.db = db

    async def get_monthly_stats(self, year: int, month: int) -> List[Dict[str, Any]]:
        Get monthly statistics for all users.
        query = '''
            SELECT 
                um.username,
                usm.author_id,
                usm.total_msgs,
                usm.total_reacts,
                CAST(usm.total_reacts AS FLOAT) / NULLIF(usm.total_msgs, 0) as avg_reacts
            FROM user_stats_monthly usm
            LEFT JOIN user_mapping um ON usm.author_id = um.author_id
            WHERE year = ? AND month = ?
        '''
        rows = await self.db.fetchall(query, (year, month))
        return [
            {
                'username': row[0],
                'author_id': row[1],
                'total_msgs': row[2],
                'total_reacts': row[3],
                'avg_reacts': row[4] or 0
            }
            for row in rows
        ]

    async def get_user_monthly_stats(self, user_id: int, year: int, month: int) -> Optional[Dict[str, Any]]:
        Get monthly statistics for a specific user.
        query = '''
            SELECT 
                total_msgs,
                total_reacts,
                CAST(total_reacts AS FLOAT) / NULLIF(total_msgs, 0) as avg_reacts
            FROM user_stats_monthly
            WHERE author_id = ? AND year = ? AND month = ?
        '''
        row = await self.db.fetchone(query, (user_id, year, month))
        if row:
            return {
                'total_msgs': row[0],
                'total_reacts': row[1],
                'avg_reacts': row[2] or 0
            }
        return None

    async def get_reaction_inflation_raw(self, monthly: bool = True, limit: int = 12) -> List[Dict[str, Any]]:
        Get reaction inflation data.
        if monthly:
            query = '''
                SELECT year, month, 
                       AVG(CAST(total_reacts AS FLOAT) / NULLIF(total_msgs, 0)) as average_rpm,
                       SUM(total_reacts) as total_reactions,
                       SUM(total_msgs) as total_messages
                FROM user_stats_monthly
                GROUP BY year, month
                ORDER BY year DESC, month DESC
                LIMIT ?
            '''
        else:
            query = '''
                SELECT year,
                       AVG(CAST(total_reacts AS FLOAT) / NULLIF(total_msgs, 0)) as average_rpm,
                       SUM(total_reacts) as total_reactions,
                       SUM(total_msgs) as total_messages
                FROM user_stats_monthly
                GROUP BY year
                ORDER BY year DESC
            '''
        rows = await self.db.fetchall(query, (limit,) if monthly else ())
        return [
            {
                'year': row[0],
                'month': row[1] if monthly else None,
                'average_rpm': row[2] or 0,
                'total_reactions': row[3],
                'total_messages': row[4]
            }
            for row in rows
        ]
