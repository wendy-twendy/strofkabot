"""Background task manager for database and username updates."""

import asyncio
import datetime
import logging
import re
from collections.abc import Callable

import discord

from strofkabot.config import REACT_COUNT_THRESHOLD
from strofkabot.discord_db import Message, MessageDatabase
from strofkabot.message_filter import MessageFilter
from strofkabot.user_stats import UserStats
from strofkabot.utils import get_reply_info


class BackgroundTaskManager:
    """Manages background tasks for database and username updates."""

    def __init__(
        self,
        bot: discord.ext.commands.Bot,
        db: MessageDatabase,
        user_stats: UserStats,
        message_filter: MessageFilter,
        logger: logging.Logger
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.message_filter = message_filter
        self.logger = logger
        self.guild: discord.Guild | None = None
        self.react_count_threshold = REACT_COUNT_THRESHOLD
        self.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        # Emoji pattern for username cleaning
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

    def set_guild(self, guild: discord.Guild) -> None:
        """Set the guild for background tasks."""
        self.guild = guild

    async def update_db(self) -> None:
        """Update the database with new messages from all channels."""
        if not self.guild:
            self.logger.warning("Guild not set. Skipping database update.")
            return

        current_time = datetime.datetime.now(datetime.UTC)
        scan_until = current_time - datetime.timedelta(days=1)

        message_counts = []
        reaction_counts = []

        channels = [
            channel for channel in self.guild.text_channels
            if channel.permissions_for(self.guild.me).read_messages
        ]

        await asyncio.gather(*(
            self._process_channel(
                channel,
                scan_until,
                lambda msg_count, react_count: (
                    message_counts.append(msg_count),
                    reaction_counts.append(react_count)
                )
            )
            for channel in channels
        ))

        total_message_count = sum(message_counts)
        total_reaction_count = sum(reaction_counts)

        self.logger.info(f"Database update completed. Total messages added: {total_message_count}")
        self.logger.info(f"Total reactions processed: {total_reaction_count}")

    async def _process_channel(
        self,
        channel: discord.TextChannel,
        scan_until: datetime.datetime,
        update_totals_callback: Callable[[int, int], None]
    ) -> None:
        """Process a single channel for new messages."""
        last_scanned = await self.db.get_last_scanned_timestamp(channel.id) or \
            datetime.datetime(2017, 1, 1, tzinfo=datetime.UTC)
        self.logger.info(f"Updating channel: {channel.name} (ID: {channel.id})")
        self.logger.info(f"Last scanned timestamp for this channel: {last_scanned}")

        messages_to_insert = []
        stats_to_update = []
        total_message_count = 0
        total_reaction_count = 0
        last_message_time = scan_until

        async for message in channel.history(after=last_scanned, before=scan_until, limit=None):
            if message.author.id == self.bot.user.id or not message.content.strip():
                continue

            last_message_time = message.created_at
            reaction_count = self._get_all_reacts(message)
            stats_to_update.append((message.author.id, reaction_count, message.created_at))

            await self._process_reactions(message)

            if len(stats_to_update) >= 100:
                await self.user_stats.batch_update_stats(stats_to_update)
                stats_to_update.clear()

            if reaction_count < self.react_count_threshold or \
               not self.message_filter.is_valid_message(message.content):
                continue

            reply_info = get_reply_info(message)
            messages_to_insert.append(Message(
                id=message.id,
                content=message.content,
                timestamp=message.created_at,
                reaction_count=reaction_count,
                author_id=message.author.id,
                reply_to_id=reply_info[0],
                reply_to_author=reply_info[1],
                reply_to_content=reply_info[2]
            ))

            total_message_count += 1
            total_reaction_count += reaction_count

            if len(messages_to_insert) >= 100:
                await self.db.add_messages(messages_to_insert)
                messages_to_insert.clear()

        if messages_to_insert:
            await self.db.add_messages(messages_to_insert)
        if stats_to_update:
            await self.user_stats.batch_update_stats(stats_to_update)
        await self.db.update_last_scanned_timestamp(channel.id, last_message_time)

        self.logger.info(f"Channel {channel.name} updated. Added {total_message_count} messages.")
        update_totals_callback(total_message_count, total_reaction_count)

    async def _process_reactions(self, message: discord.Message) -> None:
        """Process all reactions on a message."""
        for reaction in message.reactions:
            async for user in reaction.users():
                if user.bot:
                    continue
                await self.user_stats.update_reaction_stats(
                    user.id, message.author.id, message.created_at
                )

    def _get_all_reacts(self, message: discord.Message) -> int:
        """Get total reaction count for a message."""
        return sum(reaction.count for reaction in message.reactions)

    async def update_usernames(self) -> None:
        """Update username mappings for all guild members."""
        if not self.guild:
            self.logger.warning("Guild not set. Skipping username update.")
            return

        current_time = datetime.datetime.now(datetime.UTC)
        if (current_time - self.last_username_update).total_seconds() < 86400:
            self.logger.info("Username update skipped (less than 24 hours since last update).")
            return

        self.logger.info(
            f"Updating usernames for guild: {self.guild.name} (ID: {self.guild.id}). "
            f"Total members: {self.guild.member_count}"
        )

        for member in self.guild.members:
            clean_name = self._emoji_pattern.sub(r'', member.display_name)
            await self.user_stats.update_user_mapping(member.id, clean_name)

        self.last_username_update = current_time
        self.logger.info(f"Finished updating usernames at {current_time.isoformat()}.")
