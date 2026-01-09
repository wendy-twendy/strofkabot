# Strofka Bot

Discord bot for community analytics on the Strofka server, tracking user engagement through reactions and messages.

Built with [discord.py](https://github.com/Rapptz/discord.py)

![CI/CD](https://github.com/wendy-twendy/strofkabot/actions/workflows/ci.yml/badge.svg)

## Features

### Entertainment
| Command | Description |
|---------|-------------|
| `!llumi [-i\|--image]` | Returns a random high-quality message or image. Use `-i` to force an image |
| `!artan` | Returns a random Artan Kastro quote |
| `!otd` / `!on-this-day` | Shows a memorable message from this day in a previous year |
| `!ditaezeze` | Shows today's historical events |
| `!unsubscribe` | Easter egg response |

### User Stats
| Command | Description |
|---------|-------------|
| `!rpm [--leaderboard] [--least] [--all]` | Shows reaction stats. Default: personal stats. Use `--leaderboard` for rankings |
| `!most-liked [--all]` | Shows most liked users for the current month |
| `!activity` | Shows user activity heatmap |

### Economics
| Command | Description |
|---------|-------------|
| `!inflation` | Displays reaction inflation trends over time |
| `!trade [@user] [--yearly]` | Shows reaction trade balance. Default: current month with navigation. `--yearly` for 12-month data |
| `!gdp [--all]` | Plots server "GDP" (total messages per month). Use `--all` for full history |
| `!hdi` | Plots server "HDI" (quality messages with 4+ reactions) over time |

### Network Analysis
| Command | Description |
|---------|-------------|
| `!connections` | Shows top 10 mutual relationships with month navigation |
| `!riekt-graph [months]` | Shows reaction network graph. Months: 1-12, default 1 |
| `!cluster` | Shows community clusters based on interactions |
| `!echo-chamber` / `!ec` / `!bubble` | Analyzes echo chamber dynamics |

### AI
| Command | Description |
|---------|-------------|
| `!ask <question>` | Ask the AI a question with conversation context |
| `!predict <date> <text>` | Make a prediction for a future date (e.g., `tomorrow`, `25-12-2025`) |

## Setup

### Requirements
- Python 3.12+
- Discord bot token

### Installation

```bash
# Clone the repository
git clone https://github.com/wendy-twendy/strofkabot.git
cd strofkabot

# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -r requirements.txt

# Create .env file with your tokens
cat > .env << EOF
LLUMI_BOT_TOKEN=your_discord_token_here
OPENROUTER_API_KEY=your_openrouter_key_here  # Optional, for AI features
GOOGLE_API_KEY=your_google_key_here          # Optional, AI fallback
EOF

# Run the bot
.venv/bin/python strofkabot/llumi.py
```

### Running Tests

```bash
.venv/bin/pytest tests/
```

### Linting

```bash
.venv/bin/ruff check .
```

## Production Deployment

The bot runs as a Podman container via systemd. See [PODMAN.md](PODMAN.md) for container deployment instructions.

## CI/CD

The repository uses GitHub Actions for continuous integration and deployment:
- **On every push**: Runs linting (ruff) and tests (pytest)
- **On push to main**: Automatically deploys to the server via SSH

## Project Structure

```
strofkabot/
├── llumi.py              # Entry point - orchestrates cogs and background tasks
├── config.py             # Configuration constants
├── user_stats.py         # Business logic for user statistics
├── message_filter.py     # Message quality filtering
├── artan_quotes.py       # Quote data for !artan command
├── openrouter_client.py  # OpenRouter API client (primary AI)
├── gemini_client.py      # Google Gemini API client (AI fallback)
├── image_processor.py    # Image compression for AI vision
├── message_history_db.py # Message history database for AI context
├── url_extractor.py      # URL detection in messages
├── cogs/                 # Discord command modules
│   ├── base.py           # BaseCog - shared dependencies
│   ├── ai.py             # AI commands (!ask, !predict)
│   ├── economics.py      # Economy analytics (!inflation, !trade, !gdp, !hdi)
│   ├── entertainment.py  # Fun commands (!llumi, !artan, !otd, !ditaezeze)
│   ├── network.py        # Network analysis (!connections, !riekt-graph, !cluster)
│   └── user_stats.py     # User engagement (!rpm, !most-liked, !activity)
├── db/                   # Database layer (mixin-based architecture)
│   ├── base.py           # Connection management, schema
│   ├── messages.py       # Message operations
│   ├── attachments.py    # Attachment storage
│   ├── stats.py          # Statistics queries
│   ├── predictions.py    # Prediction tracking
│   └── on_this_day.py    # Historical lookups
├── tasks/
│   └── update_tasks.py   # Background tasks (DB/username updates)
└── utils/
    ├── visualization.py  # Matplotlib/seaborn plotting
    ├── data_processing.py# Data fetching and calculations
    ├── discord_helpers.py# Discord-specific utilities
    ├── date_utils.py     # Date arithmetic helpers
    ├── ask_helpers.py    # AI context preparation
    ├── nickname_loader.py# Custom nickname loading
    └── reaction_graph.py # Graph operations for network analysis
```

## License

Private project for the Strofka Discord server.
