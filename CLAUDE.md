# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

**IMPORTANT: Always use the virtual environment for installing packages and running commands.**

```bash
# Create virtual environment (first time only)
python3 -m venv .venv

# Install dependencies (use venv pip)
.venv/bin/pip install -r requirements.txt

# Run the bot
.venv/bin/python strofkabot/llumi.py [--log-level {DEBUG|INFO|WARNING|ERROR|CRITICAL}]

# Run tests
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_message_filter.py

# Run linter
.venv/bin/ruff check .
```

## Development Workflow

Follow TDD: write tests first, verify they fail, then write code and confirm tests pass.

## Container (Production)

The bot runs as a Podman container via systemd Quadlet. See `PODMAN.md` for details.

```bash
systemctl --user status strofkabot      # Check status
systemctl --user restart strofkabot     # Restart
journalctl --user -u strofkabot -f      # Logs

# Rebuild after code changes
systemctl --user stop strofkabot && podman build -t strofkabot . && systemctl --user start strofkabot
```

## Environment Setup

The bot requires `LLUMI_BOT_TOKEN` in a `.env` file at the project root.

## Architecture

This is a Discord bot for community analytics on the "Strofka" server, tracking user engagement through reactions and messages.

### Project Structure

```
strofkabot/
├── llumi.py              # Entry point, LlumiBot cog with Discord commands
├── config.py             # Constants (GUILD_ID, paths, thresholds)
├── discord_db.py         # Database - single DAL for all SQLite operations
├── user_stats.py         # UserStats - business logic for user statistics
├── message_filter.py     # MessageFilter - validates message quality
├── artan_quotes.py       # ArtanQuotes - quote data for !artan command
├── tasks/
│   ├── __init__.py
│   └── update_tasks.py   # BackgroundTaskManager - DB/username update tasks
└── utils/
    ├── __init__.py       # Re-exports all utility functions
    ├── visualization.py  # Matplotlib/seaborn plotting functions
    ├── data_processing.py# Data fetching and calculations
    ├── discord_helpers.py# Discord-specific utilities (leaderboards, etc.)
    └── date_utils.py     # Date arithmetic helpers
```

### Core Components

**Entry Point**: `strofkabot/llumi.py` - Contains `LlumiBot` cog with Discord commands. Delegates background tasks to `BackgroundTaskManager`.

**Background Tasks**: `strofkabot/tasks/update_tasks.py` - `BackgroundTaskManager` handles:
- `update_db()` - Scans Discord history every 24 hours
- `update_usernames()` - Updates username cache every 24 hours

**Database Layer** (async SQLite via aiosqlite):
- `strofkabot/discord_db.py` - `Database` is the single DAL for all SQL operations:
  - Stores high-quality messages (≥4 reactions, passes filter)
  - Manages user stats tables (`user_stats_monthly`, `user_reactions_monthly`, `user_mapping`)
  - All raw SQL queries live here
- `strofkabot/user_stats.py` - `UserStats` provides business logic layer:
  - Takes a `Database` instance, no direct SQL
  - Formats query results into dicts, handles logging
  - Methods: `get_gdp_data()`, `get_hdi_data()`, `get_reaction_trade_data()`, `get_reaction_network_for_month()`

**Utilities**: `strofkabot/utils/` package with specialized modules:
- `visualization.py` - All plot generation (inflation, GDP, HDI, reaction matrix, clusters)
- `data_processing.py` - Data fetching via DAL, inflation calculations, clustering
- `discord_helpers.py` - Leaderboard display, personal stats, most-liked rankings, `get_reply_info()`
- `date_utils.py` - Month arithmetic with year boundary handling

### Data Flow

1. Background task `update_db_task` scans Discord history every 24 hours
2. Messages go through `MessageFilter` (min 15 chars, no links/emojis/mentions)
3. Stats aggregated in `user_stats_monthly` table, reactions tracked in `user_reactions_monthly`
4. Commands call utility functions → utilities call DAL methods → visualizations generated with matplotlib/seaborn

### Key Database Tables

- `user_stats_monthly` - (author_id, year, month) → message count, reaction count
- `user_reactions_monthly` - (giver_id, receiver_id, year, month) → reaction count (who reacts to whom)
- `messages` - High-quality messages for the `!llumi` command
- `user_mapping` - author_id → display name cache

### Discord Commands

Commands are prefixed with `!`: `llumi`, `rpm`, `trade`, `inflation`, `gdp`, `hdi`, `most-liked`, `artan`, `unsubscribe`

### Configuration

All constants are centralized in `strofkabot/config.py`:
- `GUILD_ID` - Discord server ID
- `UPDATE_INTERVAL_SECONDS` - Background task interval (24 hours)
- `DATABASE_FILE_LOCATION` - Path to SQLite database
- `ARTAN_QUOTES_PATH` - Path to quotes YAML file
- `REACT_COUNT_THRESHOLD` - Minimum reactions for quality messages (4)
