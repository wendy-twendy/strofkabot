# First, create a backup of the main branch
git checkout main
git pull origin main
git checkout -b backup/main-2025-02-18
git push origin backup/main-2025-02-18

# Now create and switch to a new branch for our changes
git checkout -b feature/data-access-restructure

# Create the required directories
mkdir -p  strofkabot/models strofkabot/utils

# Create all the new files
# data/database.py
cat > strofkabot/data/database.py << 'EOL'
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
EOL

# data/user_stats_dal.py
cat > strofkabot/data/user_stats_dal.py << 'EOL'
from typing import List, Dict, Any, Optional
from datetime import datetime
from .database import Database

class UserStatsDAL:
    def __init__(self, db: Database):
        self.db = db

    async def get_monthly_stats(self, year: int, month: int) -> List[Dict[str, Any]]:
        """Get monthly statistics for all users."""
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
        """Get monthly statistics for a specific user."""
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
        """Get reaction inflation data."""
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
EOL

# data/reactions_dal.py
cat > strofkabot/data/reactions_dal.py << 'EOL'
from typing import List, Dict, Any, Tuple
from datetime import datetime
from .database import Database

class ReactionsDAL:
    def __init__(self, db: Database):
        self.db = db

    async def get_user_reaction_trade(self, user_id: int, year: int, month: int, limit: int = 5) -> Dict[str, Any]:
        """Get reaction trade data for a specific user."""
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
        """Get reaction network data for a specific month."""
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
EOL

# utils/date_utils.py
cat > strofkabot/utils/date_utils.py << 'EOL'
from datetime import datetime, date

def adjust_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Adjust year and month by the given offset."""
    new_month = month + offset
    new_year = year
    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -= 1
    return new_year, new_month

def get_previous_month(date: date = None) -> tuple[int, int]:
    """Get the year and month of the previous month."""
    if date is None:
        date = datetime.now().date()
    return adjust_month(date.year, date.month, -1)
EOL

# utils/discord_utils.py
cat > strofkabot/utils/discord_utils.py << 'EOL'
import discord
from typing import List, Optional

async def get_member_names(guild: discord.Guild, member_ids: List[int]) -> List[str]:
    """Get display names for a list of member IDs."""
    member_names = []
    for user_id in member_ids:
        member = guild.get_member(user_id)
        if member:
            name = ''.join(char for char in member.display_name if ord(char) < 128) or f"User {user_id}"
            member_names.append(name)
        else:
            member_names.append(f"User {user_id}")
    return member_names

def get_reply_info(message: discord.Message) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Get information about a message reply."""
    reply_to_id = None
    reply_to_author = None
    reply_to_content = None

    if message.reference and message.reference.resolved:
        replied_msg = message.reference.resolved
        reply_to_id = replied_msg.id

        if isinstance(replied_msg, discord.DeletedReferencedMessage):
            reply_to_author = "Deleted User"
            reply_to_content = "Message was deleted"
        else:
            reply_to_author = replied_msg.author.display_name if replied_msg.author else "Unknown User"
            reply_to_content = replied_msg.content if hasattr(replied_msg, 'content') else "Content unavailable"

    return reply_to_id, reply_to_author, reply_to_content

async def get_non_bot_member_ids(guild: discord.Guild) -> List[int]:
    """Get IDs of all non-bot members in the guild."""
    return [member.id for member in guild.members if not member.bot]
EOL

# utils/viz_utils.py
cat > strofkabot/utils/viz_utils.py << 'EOL'
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Dict, Any
import networkx as nx
import pandas as pd

def create_reaction_matrix_plot(data: pd.DataFrame, labels: List[str], fig_size: float) -> io.BytesIO:
    """Create a heatmap of reaction interactions."""
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, 
                     cmap="YlGnBu", square=True, cbar_kws={'shrink': .8})
    
    plt.xticks(rotation=90, ha='center', fontsize=16)
    plt.yticks(rotation=0, va='center', fontsize=16)
    
    plt.title("Reaction Matrix (Percentage of Reactions Given)", fontsize=22, pad=20)
    plt.xlabel("Reactions Received From", fontsize=18, labelpad=15)
    plt.ylabel("Reactions Given To", fontsize=18, labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_gdp_plot(data: List[Dict[str, Any]]) -> io.BytesIO:
    """Create a GDP (message count) plot."""
    plt.figure(figsize=(12, 6))
    data = data[::-1]  # Reverse to show oldest to newest
    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    messages = [record['total_messages'] for record in data]
    
    plt.plot(months, messages, marker='o', linestyle='-', linewidth=2, markersize=8, color='#ff7f0e')
    plt.fill_between(months, messages, alpha=0.2, color='#ff7f0e')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Month")
    plt.ylabel("Total Messages")
    plt.title("Server GDP (Total Messages per Month)")
    
    for i, v in enumerate(messages):
        plt.text(i, v + (max(messages) * 0.02), str(v), 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf
EOL

# Add empty __init__.py files
touch strofkabot/data/__init__.py
touch strofkabot/models/__init__.py
touch strofkabot/utils/__init__.py

# Stage all new files
git add strofkabot/data/
git add strofkabot/models/
git add strofkabot/utils/

# Commit the changes
git commit -m "refactor: reorganize data access layer and utilities

- Split utils.py into modular components
- Create dedicated data access layer
