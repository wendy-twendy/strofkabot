#!/usr/bin/env python3
"""Backfill user_replies_monthly table from message_history database.

This script populates the reply cache table with historical data from
January 1, 2025 onwards. It uses SQLite ATTACH to join across databases.

Usage:
    .venv/bin/python scripts/backfill_reply_stats.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiosqlite

from strofkabot.config import DATABASE_FILE_LOCATION, MESSAGE_HISTORY_DATABASE_FILE


async def backfill_replies():
    """Populate user_replies_monthly from message_history database."""
    print(f"Main database: {DATABASE_FILE_LOCATION}")
    print(f"Message history database: {MESSAGE_HISTORY_DATABASE_FILE}")

    # Open main database
    async with aiosqlite.connect(str(DATABASE_FILE_LOCATION)) as db:
        # Create table if it doesn't exist
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_replies_monthly (
                replier_id INTEGER,
                replied_to_id INTEGER,
                year INTEGER,
                month INTEGER,
                reply_count INTEGER,
                PRIMARY KEY (replier_id, replied_to_id, year, month)
            )
        """)
        await db.commit()

        # Attach message history database
        await db.execute(f"ATTACH DATABASE '{MESSAGE_HISTORY_DATABASE_FILE}' AS history")

        # Query for reply counts from 2025-01-01 onwards
        # Join message_history.messages with main.user_mapping to get replied_to_id
        query = """
            SELECT
                m.author_id AS replier_id,
                um.author_id AS replied_to_id,
                CAST(strftime('%Y', m.timestamp) AS INTEGER) AS year,
                CAST(strftime('%m', m.timestamp) AS INTEGER) AS month,
                COUNT(*) AS reply_count
            FROM history.messages m
            JOIN main.user_mapping um ON m.reply_to_author = um.username
            WHERE m.reply_to_author IS NOT NULL
              AND m.timestamp >= '2025-01-01'
              AND m.author_id != um.author_id
            GROUP BY m.author_id, um.author_id, year, month
        """

        print("Querying reply data from 2025-01-01...")
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()

        print(f"Found {len(rows)} reply aggregates to insert")

        if not rows:
            print("No data to backfill.")
            return

        # Insert into user_replies_monthly (with conflict handling)
        insert_query = """
            INSERT INTO user_replies_monthly (replier_id, replied_to_id, year, month, reply_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(replier_id, replied_to_id, year, month) DO UPDATE SET
                reply_count = excluded.reply_count
        """

        print("Inserting into user_replies_monthly...")
        await db.executemany(insert_query, rows)
        await db.commit()

        print(f"Successfully backfilled {len(rows)} reply aggregates.")

        # Show summary by month
        summary_query = """
            SELECT year, month, SUM(reply_count) as total_replies, COUNT(*) as pairs
            FROM user_replies_monthly
            GROUP BY year, month
            ORDER BY year, month
        """
        async with db.execute(summary_query) as cursor:
            summary = await cursor.fetchall()

        print("\nSummary by month:")
        for year, month, total, pairs in summary:
            print(f"  {year}-{month:02d}: {total} replies across {pairs} user pairs")


if __name__ == "__main__":
    asyncio.run(backfill_replies())
