#!/usr/bin/env python3
"""List all guild members with their nicknames and usernames.

Usage:
    set -a && source .env && set +a && .venv/bin/python scripts/list_members.py
"""

import asyncio
import os
import sys

import discord
from dotenv import load_dotenv

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from strofkabot.config import GUILD_ID, NICKNAMES_FILE
from strofkabot.utils.nickname_loader import load_nicknames


async def main():
    load_dotenv()
    token = os.getenv("LLUMI_BOT_TOKEN")
    if not token:
        print("Error: LLUMI_BOT_TOKEN not set")
        return

    # Load custom nicknames
    custom_nicknames = load_nicknames(NICKNAMES_FILE)
    print(f"Loaded {len(custom_nicknames)} custom nickname mappings from YAML\n")

    intents = discord.Intents.default()
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        guild = client.get_guild(GUILD_ID)
        if not guild:
            print(f"Error: Guild {GUILD_ID} not found")
            await client.close()
            return

        print(f"Guild: {guild.name} ({guild.member_count} members)\n")
        print("=" * 80)
        print(f"{'ID':<20} {'Username':<25} {'Display Name':<25} {'Custom Nicks'}")
        print("=" * 80)

        # Sort by display name
        members = sorted(guild.members, key=lambda m: m.display_name.lower())

        for member in members:
            if member.bot:
                continue

            user_id = member.id
            username = member.name  # Discord username
            display_name = member.display_name  # Server nickname or username

            # Get custom nicknames from YAML
            custom_nicks = custom_nicknames.get(user_id, [])
            custom_str = ", ".join(custom_nicks) if custom_nicks else "-"

            print(f"{user_id:<20} {username:<25} {display_name:<25} {custom_str}")

        print("=" * 80)
        print(f"\nTotal non-bot members: {sum(1 for m in members if not m.bot)}")

        await client.close()

    await client.start(token)


if __name__ == "__main__":
    asyncio.run(main())
