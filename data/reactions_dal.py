from typing import List, Dict, Any, Tuple
from datetime import datetime
from .database import Database

class ReactionsDAL:
    def __init__(self, db: Database):
        self.db = db

    async def get_user_reaction_trade(self, user_id: int, year: int, month: int, limit: int = 5) -> Dict[str, Any]:
        Get reaction trade data for a specific user.
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
        
        exports = await self.db.fetchall(export_query, (user_id, year, year, month, limit))
        imports = await self.db.fetchall(import_query, (user_id, year, year, month, limit))
        
        total_query = '''
            SELECT
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE giver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_given,
                (SELECT COALESCE(SUM(reaction_count), 0)
                 FROM user_reactions_monthly
                 WHERE receiver_id = ? AND (year > ? OR (year = ? AND month >= ?))) as total_received
        '''
        
        total_data = await self.db.fetchone(total_query, 
            (user_id, year, year, month, user_id, year, year, month))
        
        return {
            'exports': [(row[0], row[1]) for row in exports],
            'imports': [(row[0], row[1]) for row in imports],
            'total_given': total_data[0],
            'total_received': total_data[1],
            'trade_balance': total_data[1] - total_data[0]
        }

    async def get_monthly_reaction_network(self, year: int, month: int) -> List[Dict[str, Any]]:
        Get reaction network data for a specific month.
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
        rows = await self.db.fetchall(query, (year, month))
        return [
            {
                'giver_username': row[0],
                'receiver_username': row[1],
                'reaction_count': row[2],
                'giver_messages': row[3],
                'receiver_messages': row[4]
            }
            for row in rows
        ]
