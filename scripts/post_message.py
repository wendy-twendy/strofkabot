#!/usr/bin/env python3
"""Post or delete a message in Discord channel via the bot."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import discord
from dotenv import load_dotenv

from strofkabot.config import PREDICTIONS_CHANNEL_ID

CHANNEL_ID = PREDICTIONS_CHANNEL_ID  # 1458523811824537621


async def post_message(token: str, message: str):
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        channel = client.get_channel(CHANNEL_ID)
        if not channel:
            print(f"Error: Channel {CHANNEL_ID} not found")
            await client.close()
            return

        embed = discord.Embed(description=message, color=discord.Color.blue())
        sent = await channel.send(embed=embed)
        print(f"Message sent to #{channel.name} (ID: {sent.id})")
        await client.close()

    try:
        await client.start(token)
    finally:
        if not client.is_closed():
            await client.close()
        # Allow aiohttp to clean up connections
        await asyncio.sleep(0.25)


async def delete_message(token: str, message_id: int):
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        channel = client.get_channel(CHANNEL_ID)
        if not channel:
            print(f"Error: Channel {CHANNEL_ID} not found")
            await client.close()
            return

        try:
            msg = await channel.fetch_message(message_id)
            await msg.delete()
            print(f"Message {message_id} deleted from #{channel.name}")
        except discord.NotFound:
            print(f"Error: Message {message_id} not found")
        except discord.Forbidden:
            print(f"Error: No permission to delete message {message_id}")

        await client.close()

    try:
        await client.start(token)
    finally:
        if not client.is_closed():
            await client.close()
        await asyncio.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description="Post or delete Discord messages")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("message", nargs="?", help="Message to post")
    group.add_argument("--delete", "-d", type=int, metavar="ID", help="Message ID to delete")

    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("LLUMI_BOT_TOKEN")
    if not token:
        print("Error: LLUMI_BOT_TOKEN not found in environment")
        sys.exit(1)

    if args.delete:
        asyncio.run(delete_message(token, args.delete))
    else:
        asyncio.run(post_message(token, args.message))


if __name__ == "__main__":
    main()
