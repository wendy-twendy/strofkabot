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

**When making big AI-related changes** (to `openrouter_client.py`, `gemini_client.py`, `cogs/ai.py`, or `utils/ask_helpers.py`), run the E2E tests to verify the AI functionality still works correctly, but use only for big changes:

```bash
# Run AI E2E tests (requires OPENROUTER_API_KEY in .env)
set -a && source .env && set +a && .venv/bin/pytest tests/e2e/test_ask_e2e.py -v -m e2e
```

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

The bot requires the following in a `.env` file at the project root:
- `LLUMI_BOT_TOKEN` - Discord bot token (required)
- `GOOGLE_API_KEY` - Google Gemini API key (optional, for AI fallback)
- `OPENROUTER_API_KEY` - OpenRouter API key (optional, for AI features)

## Architecture

This is a Discord bot for community analytics on the "Strofka" server, tracking user engagement through reactions and messages.

### Project Structure

```
strofkabot/
├── llumi.py              # Entry point - orchestrates cogs and background tasks
├── config.py             # Constants (GUILD_ID, paths, thresholds, AI config)
├── discord_db.py         # Legacy database import (uses db/ module)
├── user_stats.py         # UserStats - business logic for user statistics
├── message_filter.py     # MessageFilter - validates message quality
├── artan_quotes.py       # ArtanQuotes - quote data for !artan command
├── gemini_client.py      # Google Gemini API client (AI fallback)
├── openrouter_client.py  # OpenRouter API client (primary AI)
├── image_processor.py    # Image compression and format handling
├── url_extractor.py      # URL detection in messages
├── cogs/                 # Discord command modules
│   ├── base.py           # BaseCog - shared dependencies for all cogs
│   ├── ai.py             # AICog - !ask, !predict commands
│   ├── economics.py      # EconomicsCog - !inflation, !trade, !gdp, !hdi
│   ├── entertainment.py  # EntertainmentCog - !llumi, !artan, !on-this-day, !ditaezeze
│   ├── network.py        # NetworkCog - !connections, !riekt-graph, !cluster, !echo-chamber
│   └── user_stats.py     # UserStatsCog - !rpm, !most-liked, !activity
├── db/                   # Database operation modules (mixin-based)
│   ├── base.py           # BaseDatabase - connection management, schema
│   ├── messages.py       # MessagesMixin - message table operations
│   ├── attachments.py    # AttachmentsMixin - attachment storage
│   ├── stats.py          # StatsMixin - user statistics queries
│   ├── predictions.py    # PredictionsMixin - prediction tracking
│   ├── on_this_day.py    # OnThisDayMixin - historical lookups
│   └── message_history.py # MessageHistoryMixin - unfiltered message history for AI
├── tasks/
│   ├── __init__.py
│   └── update_tasks.py   # BackgroundTaskManager - DB/username update tasks
└── utils/
    ├── __init__.py       # Re-exports all utility functions
    ├── visualization.py  # Matplotlib/seaborn plotting functions
    ├── data_processing.py# Data fetching and calculations
    ├── discord_helpers.py# Discord-specific utilities (leaderboards, etc.)
    ├── date_utils.py     # Date arithmetic helpers
    ├── ask_helpers.py    # AI context preparation, message formatting
    ├── nickname_loader.py# Custom nickname loading from YAML
    └── reaction_graph.py # Graph operations for network analysis
```

### Core Components

**Entry Point**: `strofkabot/llumi.py` - Orchestrates cog loading and manages background tasks. Commands are in `cogs/` modules.

**Cogs**: `strofkabot/cogs/` - Domain-based command modules:
- `base.py` - BaseCog with shared database, user_stats, and logger
- `ai.py` - AICog: AI-powered commands (!ask, !predict)
- `economics.py` - EconomicsCog: Server economy analytics
- `entertainment.py` - EntertainmentCog: Fun and historical content
- `network.py` - NetworkCog: Social network analysis
- `user_stats.py` - UserStatsCog: User engagement metrics

**Background Tasks**: `strofkabot/tasks/update_tasks.py` - `BackgroundTaskManager` handles:
- `update_db()` - Scans Discord history every 24 hours
- `update_usernames()` - Updates username cache every 24 hours

**Database Layer** (async SQLite via aiosqlite):

**IMPORTANT: Always use Python scripts to query the SQLite database, not the sqlite3 CLI.** This ensures consistency with the async database layer and avoids locking issues.

Single unified database (`db.sqlite3`) with mixin-based architecture in `strofkabot/db/`:
- `Database` class composes specialized mixins for separation of concerns
- `BaseDatabase` handles connection management and schema creation
- `MessagesMixin` - high-quality message operations (≥4 reactions)
- `AttachmentsMixin` - image/file attachment storage
- `StatsMixin` - user statistics queries
- `PredictionsMixin` - prediction storage and retrieval
- `OnThisDayMixin` - historical message lookup
- `MessageHistoryMixin` - unfiltered message history for AI context (!ask command)

**Business Logic**: `strofkabot/user_stats.py` - `UserStats` provides high-level methods:
- Takes a `Database` instance, no direct SQL
- Methods: `get_gdp_data()`, `get_hdi_data()`, `get_reaction_trade_data()`, `get_reaction_network_for_month()`

**AI Integration**:
- `openrouter_client.py` - Primary AI provider using free models (Gemini 3 Flash)
- `gemini_client.py` - Fallback Google Gemini API client
- `image_processor.py` - Image compression for AI vision
- Automatic search/thinking detection with classifier model

**Utilities**: `strofkabot/utils/` package with specialized modules:
- `visualization.py` - All plot generation (inflation, GDP, HDI, reaction matrix, clusters, activity heatmap)
- `data_processing.py` - Data fetching via DAL, inflation calculations, clustering, hourly activity
- `discord_helpers.py` - Leaderboard display, personal stats, most-liked rankings, `get_reply_info()`
- `date_utils.py` - Month arithmetic with year boundary handling
- `ask_helpers.py` - AI context preparation, message formatting
- `nickname_loader.py` - Custom nickname loading from YAML
- `reaction_graph.py` - Graph operations for network analysis

### Data Flow

1. Background task `update_db_task` scans Discord history every 24 hours
2. Messages go through `MessageFilter` (min 15 chars, no links/emojis/mentions)
3. Stats aggregated in `user_stats_monthly` table, reactions tracked in `user_reactions_monthly`
4. Commands call utility functions → utilities call DAL methods → visualizations generated with matplotlib/seaborn

### Key Database Tables

All tables are in a single database (`db.sqlite3`):
- `messages` - High-quality messages (≥4 reactions) for !llumi command
- `message_history` - All Discord messages (unfiltered) for AI context
- `user_stats_monthly` - (author_id, year, month) → message count, reaction count
- `user_reactions_monthly` - (giver_id, receiver_id, year, month) → reaction count
- `user_replies_monthly` - (replier_id, replied_to_id, year, month) → reply count
- `user_mapping` - author_id → display name cache
- `attachments` - Stored images/files from high-quality messages
- `predictions` - User predictions with target dates and retry tracking
- `scrape_progress` - Incremental scraping state per channel

### Discord Commands

Commands are prefixed with `!`:

**Entertainment**: `llumi`, `artan`, `unsubscribe`, `ditaezeze`, `on-this-day` (alias: `otd`)
**User Stats**: `rpm`, `most-liked`, `activity`
**Economics**: `inflation`, `trade`, `gdp`, `hdi`
**Network**: `connections`, `riekt-graph`, `cluster`, `echo-chamber` (aliases: `ec`, `bubble`)
**AI**: `ask`, `predict`

### Configuration

All constants are centralized in `strofkabot/config.py`:
- `GUILD_ID` - Discord server ID
- `PREDICTIONS_CHANNEL_ID` - Channel for prediction posts
- `UPDATE_INTERVAL_SECONDS` - Background task interval (24 hours)
- `DATABASE_FILE_LOCATION` - Path to SQLite database
- `ARTAN_QUOTES_PATH` - Path to quotes YAML file
- `NICKNAMES_FILE` - Path to custom nicknames YAML
- `ATTACHMENTS_DIR` - Directory for stored attachments
- `REACT_COUNT_THRESHOLD` - Minimum reactions for quality messages (4)
- `IMAGE_MAX_SIZE_BYTES` - Image size limit (1 MB)
- `GEMINI_MODELS` - List of Gemini model IDs for fallback
- `GEMINI_RPD_LIMIT` - Requests per day limit per model
- `OPENROUTER_INFERENCE_MODEL` - Primary AI model
- `OPENROUTER_VISION_MODEL` - Vision model for images
