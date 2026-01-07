# Strofka Bot

Discord bot for community analytics on the Strofka server, tracking user engagement through reactions and messages.

Built with [discord.py](https://github.com/Rapptz/discord.py)

![CI/CD](https://github.com/wendy-twendy/strofkabot/actions/workflows/ci.yml/badge.svg)

## Features

| Command | Description |
|---------|-------------|
| `!llumi` | Returns a random high-quality message (4+ reactions) from history |
| `!rpm [month] [year]` | Shows reactions-per-message leaderboard |
| `!inflation` | Displays reaction inflation trends over time |
| `!trade @user` | Shows reaction trade balance with another user |
| `!gdp` | Plots server "GDP" (total reactions) over time |
| `!hdi` | Plots "Human Development Index" (engagement metrics) |
| `!most-liked` | Shows most liked users for a given month |
| `!artan` | Returns a random Artan Kastro quote |

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
