#!/usr/bin/env python3
"""Ingest messages for a specific date range into RAG vector store."""

import argparse
import asyncio
import logging
from datetime import datetime

from strofkabot.config import DATABASE_FILE_LOCATION, NICKNAMES_FILE, RAG_VECTOR_STORE_DIR
from strofkabot.rag.ingestion import IngestionPipeline
from strofkabot.utils.nickname_loader import load_nicknames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    parser = argparse.ArgumentParser(description="Ingest messages by date range")
    parser.add_argument("--after", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--before", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")
    parser.add_argument("--dry-run", action="store_true", help="Count messages without ingesting")
    parser.add_argument(
        "--batch-days",
        type=int,
        default=30,
        help="Days per batch for memory-efficient processing (default: 30)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from progress",
    )
    args = parser.parse_args()

    after_date = datetime.fromisoformat(args.after) if args.after else None
    before_date = datetime.fromisoformat(args.before) if args.before else None

    print(f"Date range: {args.after or 'beginning'} to {args.before or 'now'}")
    print(f"Batch size: {args.batch_days} days")

    nicknames = load_nicknames(NICKNAMES_FILE)
    print(f"Loaded {len(nicknames)} nickname mappings")

    pipeline = IngestionPipeline(
        db_path=DATABASE_FILE_LOCATION,
        vector_store_dir=RAG_VECTOR_STORE_DIR,
        nicknames=nicknames,
    )

    if args.dry_run:
        messages = pipeline.load_messages(after_date=after_date, before_date=before_date)
        print(f"Would ingest {len(messages):,} messages")
        return

    print(f"Clear existing: {args.clear}")
    print(f"Resume from progress: {not args.no_resume}")
    print("Starting ingestion...")

    progress = await pipeline.run(
        after_date=after_date,
        before_date=before_date,
        clear_existing=args.clear,
        batch_days=args.batch_days,
        resume=not args.no_resume,
    )

    print("\nIngestion complete:")
    print(f"  Messages: {progress.processed_messages:,}")
    print(f"  Chunks: {progress.total_chunks:,}")
    print(f"  Embeddings: {progress.embedded_chunks:,}")
    print(f"  BM25 indexed: {progress.bm25_indexed:,}")
    print(f"  Periods completed: {len(progress.completed_periods)}")
    print(f"  Status: {progress.status}")


if __name__ == "__main__":
    asyncio.run(main())
