# Strofka Bot

Discord bot for community analytics on the Strofka server, tracking user engagement through reactions and messages.

Built with [discord.py](https://github.com/Rapptz/discord.py)

![CI/CD](https://github.com/wendy-twendy/strofkabot/actions/workflows/ci.yml/badge.svg)

## Features

| Command | Description |
|---------|-------------|
| `!llumi [-i\|--image]` | Returns a random high-quality message or image. Use `-i` to force an image |
| `!rpm [--leaderboard] [--least] [--all]` | Shows reaction stats. Default: personal stats. Use `--leaderboard` for rankings |
| `!inflation` | Displays reaction inflation trends over time |
| `!trade [@user] [--yearly]` | Shows reaction trade balance. Default: current month with navigation. `--yearly` for 12-month data |
| `!gdp [--all]` | Plots server "GDP" (total messages per month). Use `--all` for full history |
| `!hdi` | Plots server "HDI" (quality messages with 4+ reactions) over time |
| `!most-liked [--all]` | Shows most liked users for the current month |
| `!riekt-graph [months]` | Shows reaction network graph. Months: 1-12, default 1 |
| `!connections` | Shows top 10 mutual relationships with month navigation |
| `!otd` / `!on-this-day` | Shows a memorable message from this day in a previous year |
| `!predict <date> <text>` | Make a prediction for a future date (e.g., `tomorrow`, `25-12-2025`) |
| `!artan` | Returns a random Artan Kastro quote |
| `!unsubscribe` | Easter egg response |

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

# Create .env file with your bot token
echo "LLUMI_BOT_TOKEN=your_token_here" > .env

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
├── llumi.py           # Entry point, Discord commands
├── config.py          # Configuration constants
├── discord_db.py      # Message database layer
├── user_stats.py      # User statistics and reactions
├── message_filter.py  # Message quality filtering
├── artan_quotes.py    # Quote management
├── tasks/             # Background tasks (DB updates)
└── utils/             # Visualization and helpers
```

## License

Private project for the Strofka Discord server.
