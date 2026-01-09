#!/usr/bin/env python3
"""One-time migration script to move data from message_history.db to main db.

This script transfers all data from the old separate message_history.db
into the new consolidated message_history table in the main db.sqlite3.

Usage:
    .venv/bin/python scripts/migrate_message_history.py

The script is idempotent and uses INSERT OR IGNORE to avoid duplicates.
"""

import sqlite3
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
OLD_DB_PATH = PROJECT_ROOT / "data" / "message_history.db"
MAIN_DB_PATH = PROJECT_ROOT / "data" / "db.sqlite3"


def migrate():
    """Migrate data from message_history.db to main database."""
    if not OLD_DB_PATH.exists():
        print(f"No {OLD_DB_PATH.name} found, nothing to migrate.")
        return

    if not MAIN_DB_PATH.exists():
        print(f"Main database {MAIN_DB_PATH.name} not found. Run the bot first to create it.")
        return

    print(f"Starting migration from {OLD_DB_PATH} to {MAIN_DB_PATH}...")

    old_conn = sqlite3.connect(OLD_DB_PATH)
    main_conn = sqlite3.connect(MAIN_DB_PATH)

    try:
        # Check if message_history table exists in main db, create if not
        main_cursor = main_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='message_history'"
        )
        if not main_cursor.fetchone():
            print("Creating message_history and scrape_progress tables...")
            main_conn.executescript("""
                CREATE TABLE IF NOT EXISTS message_history (
                    id INTEGER PRIMARY KEY,
                    channel_id INTEGER,
                    channel_name TEXT,
                    author_id INTEGER,
                    author_name TEXT,
                    content TEXT,
                    timestamp TEXT,
                    reply_to_id INTEGER,
                    reply_to_author TEXT,
                    reply_to_content TEXT,
                    reactions TEXT
                );

                CREATE TABLE IF NOT EXISTS scrape_progress (
                    channel_id INTEGER PRIMARY KEY,
                    last_message_id INTEGER,
                    last_updated TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_message_history_channel_id
                    ON message_history(channel_id);
                CREATE INDEX IF NOT EXISTS idx_message_history_timestamp
                    ON message_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_message_history_author_timestamp
                    ON message_history(author_id, timestamp);
            """)
            main_conn.commit()
            print("Tables created.")

        # Get count from old database
        old_cursor = old_conn.execute("SELECT COUNT(*) FROM messages")
        old_message_count = old_cursor.fetchone()[0]
        print(f"Found {old_message_count:,} messages in old database.")

        # Migrate messages table
        print("Migrating messages...")
        old_cursor = old_conn.execute("SELECT * FROM messages")
        rows = old_cursor.fetchall()

        main_conn.executemany(
            """INSERT OR IGNORE INTO message_history
               (id, channel_id, channel_name, author_id, author_name, content,
                timestamp, reply_to_id, reply_to_author, reply_to_content, reactions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        main_conn.commit()

        # Verify migration
        main_cursor = main_conn.execute("SELECT COUNT(*) FROM message_history")
        new_message_count = main_cursor.fetchone()[0]
        print(f"Messages in main database after migration: {new_message_count:,}")

        # Migrate scrape_progress table
        print("Migrating scrape progress...")
        old_cursor = old_conn.execute("SELECT * FROM scrape_progress")
        progress_rows = old_cursor.fetchall()

        if progress_rows:
            main_conn.executemany(
                "INSERT OR REPLACE INTO scrape_progress VALUES (?, ?, ?)",
                progress_rows,
            )
            main_conn.commit()
            print(f"Migrated {len(progress_rows)} channel progress records.")
        else:
            print("No scrape progress records to migrate.")

        print()
        print("Migration complete!")
        print(f"You can now safely delete {OLD_DB_PATH}")
        print()
        print("To verify, run the bot and check that !activity works correctly.")

    finally:
        old_conn.close()
        main_conn.close()


if __name__ == "__main__":
    migrate()
