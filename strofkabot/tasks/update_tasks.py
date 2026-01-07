"""Background task manager for database and username updates."""

import asyncio
import datetime
import json
import logging
import re
from collections.abc import Callable

import discord

from strofkabot.config import (
    ATTACHMENTS_DIR,
    MESSAGE_HISTORY_DATABASE_FILE,
    REACT_COUNT_THRESHOLD,
)
from strofkabot.discord_db import Attachment, Database, Message
from strofkabot.image_processor import ImageProcessor
from strofkabot.message_filter import MessageFilter
from strofkabot.message_history_db import HistoryMessage, MessageHistoryDatabase
from strofkabot.user_stats import UserStats
from strofkabot.utils import get_reply_info


class BackgroundTaskManager:
    """Manages background tasks for database and username updates."""

    def __init__(
        self,
        bot: discord.ext.commands.Bot,
        db: Database,
        user_stats: UserStats,
        message_filter: MessageFilter,
        logger: logging.Logger,
        message_history_db: MessageHistoryDatabase | None = None,
        image_processor: ImageProcessor | None = None,
    ):
        self.bot = bot
        self.db = db
        self.user_stats = user_stats
        self.message_filter = message_filter
        self.logger = logger
        self.guild: discord.Guild | None = None
        self.react_count_threshold = REACT_COUNT_THRESHOLD
        self.last_username_update = datetime.datetime.min.replace(tzinfo=datetime.UTC)

        # Message history database for AI purposes (unfiltered messages)
        self.message_history_db = message_history_db or MessageHistoryDatabase(
            MESSAGE_HISTORY_DATABASE_FILE
        )

        # Image processor for downloading and compressing attachments
        self.image_processor = image_processor or ImageProcessor()

        # Emoji pattern for username cleaning
        self._emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

    def set_guild(self, guild: discord.Guild) -> None:
        """Set the guild for background tasks."""
        self.guild = guild

    async def update_db(self) -> None:
        """Update the database with new messages from all channels."""
        if not self.guild:
            self.logger.warning("Guild not set. Skipping database update.")
            return

        # Initialize message history database
        await self.message_history_db.initialize()

        current_time = datetime.datetime.now(datetime.UTC)
        scan_until = current_time - datetime.timedelta(days=1)

        message_counts = []
        reaction_counts = []

        channels = [
            channel
            for channel in self.guild.text_channels
            if channel.permissions_for(self.guild.me).read_messages
        ]

        await asyncio.gather(
            *(
                self._process_channel(
                    channel,
                    scan_until,
                    lambda msg_count, react_count: (
                        message_counts.append(msg_count),
                        reaction_counts.append(react_count),
                    ),
                )
                for channel in channels
            )
        )

        total_message_count = sum(message_counts)
        total_reaction_count = sum(reaction_counts)

        self.logger.info(f"Database update completed. Total messages added: {total_message_count}")
        self.logger.info(f"Total reactions processed: {total_reaction_count}")

    async def _process_channel(
        self,
        channel: discord.TextChannel,
        scan_until: datetime.datetime,
        update_totals_callback: Callable[[int, int], None],
    ) -> None:
        """Process a single channel for new messages."""
        last_scanned = await self.db.get_last_scanned_timestamp(channel.id) or datetime.datetime(
            2017, 1, 1, tzinfo=datetime.UTC
        )
        self.logger.info(f"Updating channel: {channel.name} (ID: {channel.id})")
        self.logger.info(f"Last scanned timestamp for this channel: {last_scanned}")

        messages_to_insert = []
        history_messages_to_insert = []
        attachments_to_insert = []
        stats_to_update = []
        total_message_count = 0
        total_reaction_count = 0
        last_message_time = scan_until
        last_message_id = None

        async for message in channel.history(after=last_scanned, before=scan_until, limit=None):
            if message.author.id == self.bot.user.id:
                continue

            last_message_time = message.created_at
            last_message_id = message.id
            reaction_count = self._get_all_reacts(message)
            reply_info = get_reply_info(message)

            # Record ALL messages to message history (unfiltered, for AI purposes)
            history_messages_to_insert.append(
                HistoryMessage(
                    id=message.id,
                    channel_id=channel.id,
                    channel_name=channel.name,
                    author_id=message.author.id,
                    author_name=message.author.display_name,
                    content=message.content or "",
                    timestamp=message.created_at,
                    reply_to_id=reply_info[0],
                    reply_to_author=reply_info[1],
                    reply_to_content=reply_info[2],
                    reactions=self._serialize_reactions(message.reactions),
                )
            )

            # Batch insert history messages
            if len(history_messages_to_insert) >= 100:
                await self.message_history_db.add_messages(history_messages_to_insert)
                history_messages_to_insert.clear()

            # Skip messages with no content for stats/quality filtering
            if not message.content.strip():
                continue

            stats_to_update.append((message.author.id, reaction_count, message.created_at))
            await self._process_reactions(message)

            if len(stats_to_update) >= 100:
                await self.user_stats.batch_update_stats(stats_to_update)
                stats_to_update.clear()

            # Process attachments and quality messages for high-reaction content
            if reaction_count >= self.react_count_threshold:
                # Process attachments (images)
                for attachment in message.attachments:
                    if self.image_processor.is_image(attachment.filename):
                        exists = await self.db.attachment_exists(attachment.id)
                        if not exists:
                            output_dir = ATTACHMENTS_DIR / str(message.id)
                            result_path, _ = await self.image_processor.process_attachment(
                                url=attachment.url,
                                attachment_id=attachment.id,
                                filename=attachment.filename,
                                output_dir=output_dir,
                            )
                            if result_path:
                                local_path = str(result_path.relative_to(ATTACHMENTS_DIR))
                                attachments_to_insert.append(
                                    Attachment(
                                        id=attachment.id,
                                        message_id=message.id,
                                        message_content=message.content,
                                        author_id=message.author.id,
                                        timestamp=message.created_at,
                                        reaction_count=reaction_count,
                                        original_filename=attachment.filename,
                                        local_path=local_path,
                                    )
                                )

                # Quality message filtering for main database
                if self.message_filter.is_valid_message(message.content):
                    messages_to_insert.append(
                        Message(
                            id=message.id,
                            content=message.content,
                            timestamp=message.created_at,
                            reaction_count=reaction_count,
                            author_id=message.author.id,
                            reply_to_id=reply_info[0],
                            reply_to_author=reply_info[1],
                            reply_to_content=reply_info[2],
                        )
                    )

                    total_message_count += 1
                    total_reaction_count += reaction_count

            # Batch insert quality messages
            if len(messages_to_insert) >= 100:
                await self.db.add_messages(messages_to_insert)
                messages_to_insert.clear()

            # Batch insert attachments
            if len(attachments_to_insert) >= 50:
                await self.db.add_attachments(attachments_to_insert)
                attachments_to_insert.clear()

        # Insert remaining batches
        if history_messages_to_insert:
            await self.message_history_db.add_messages(history_messages_to_insert)
        if messages_to_insert:
            await self.db.add_messages(messages_to_insert)
        if attachments_to_insert:
            await self.db.add_attachments(attachments_to_insert)
        if stats_to_update:
            await self.user_stats.batch_update_stats(stats_to_update)

        await self.db.update_last_scanned_timestamp(channel.id, last_message_time)

        # Update message history scrape progress
        if last_message_id:
            await self.message_history_db.update_scrape_progress(channel.id, last_message_id)

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

    def _serialize_reactions(self, reactions: list) -> str:
        """Serialize message reactions to JSON string."""
        reaction_list = []
        for reaction in reactions:
            emoji_str = str(reaction.emoji)
            reaction_list.append({"emoji": emoji_str, "count": reaction.count})
        return json.dumps(reaction_list)

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
            clean_name = self._emoji_pattern.sub(r"", member.display_name)
            await self.user_stats.update_user_mapping(member.id, clean_name)

        self.last_username_update = current_time
        self.logger.info(f"Finished updating usernames at {current_time.isoformat()}.")
