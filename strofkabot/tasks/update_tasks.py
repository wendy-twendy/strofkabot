"""Background task manager for database and username updates."""

import asyncio
import datetime
import json
import logging
import re

import discord

from strofkabot.config import (
    ATTACHMENTS_DIR,
    PREDICTIONS_CHANNEL_ID,
    REACT_COUNT_THRESHOLD,
)
from strofkabot.discord_db import Attachment, Database, HistoryMessage, Message
from strofkabot.image_processor import ImageProcessor
from strofkabot.message_filter import MessageFilter
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

        # Image processor for downloading and compressing attachments
        self.image_processor = image_processor or ImageProcessor()

        # Comprehensive emoji pattern for username cleaning
        # Covers most Unicode emoji ranges including newer additions
        self._emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (regional indicators)
            "\U0001f700-\U0001f77f"  # alchemical symbols
            "\U0001f780-\U0001f7ff"  # geometric shapes extended
            "\U0001f800-\U0001f8ff"  # supplemental arrows-C
            "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
            "\U0001fa00-\U0001fa6f"  # chess symbols
            "\U0001fa70-\U0001faff"  # symbols and pictographs extended-A
            "\U0001fb00-\U0001fbff"  # symbols for legacy computing
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"  # enclosed characters
            "\U00002300-\U000023ff"  # misc technical
            "\U00002600-\U000026ff"  # misc symbols
            "\U00002700-\U000027bf"  # dingbats
            "\U0000fe00-\U0000fe0f"  # variation selectors
            "\U0001f000-\U0001f02f"  # mahjong tiles
            "\U0001f0a0-\U0001f0ff"  # playing cards
            "\U0000200d"  # zero width joiner (for ZWJ sequences)
            "\U0000203c\U00002049"  # exclamation marks
            "\U000020e3"  # combining enclosing keycap
            "\U00003030\U0000303d"  # wavy dash, part alternation mark
            "\U00003297\U00003299"  # circled ideographs
            "]+",
            flags=re.UNICODE,
        )

    def set_guild(self, guild: discord.Guild) -> None:
        """Set the guild for background tasks."""
        self.guild = guild

    async def close(self) -> None:
        """Close resources (no-op, database closed by LlumiBot)."""
        pass

    async def update_db(self) -> None:
        """Update the database with new messages from all channels."""
        if not self.guild:
            self.logger.warning("Guild not set. Skipping database update.")
            return

        current_time = datetime.datetime.now(datetime.UTC)
        scan_until = current_time - datetime.timedelta(days=1)

        channels = [
            channel
            for channel in self.guild.text_channels
            if channel.permissions_for(self.guild.me).read_messages
        ]

        # Gather returns results in order, avoiding concurrent list modification
        results = await asyncio.gather(
            *(self._process_channel(channel, scan_until) for channel in channels)
        )

        # Sum up results from all channels
        total_message_count = sum(msg_count for msg_count, _ in results)
        total_reaction_count = sum(react_count for _, react_count in results)

        self.logger.info(f"Database update completed. Total messages added: {total_message_count}")
        self.logger.info(f"Total reactions processed: {total_reaction_count}")

    async def _process_channel(
        self,
        channel: discord.TextChannel,
        scan_until: datetime.datetime,
    ) -> tuple[int, int]:
        """Process a single channel for new messages.

        Returns:
            Tuple of (message_count, reaction_count) added.
        """
        last_scanned = await self.db.get_last_scanned_timestamp(channel.id) or datetime.datetime(
            2017, 1, 1, tzinfo=datetime.UTC
        )
        self.logger.info(f"Updating channel: {channel.name} (ID: {channel.id})")
        self.logger.info(f"Last scanned timestamp for this channel: {last_scanned}")

        messages_to_insert = []
        history_messages_to_insert = []
        attachments_to_insert = []
        stats_to_update = []
        reactions_to_update = []
        replies_to_update = []
        total_message_count = 0
        total_reaction_count = 0
        last_message_time = last_scanned  # Start from last scanned, not scan_until
        last_message_id = None
        error_occurred = False

        try:
            async for message in channel.history(after=last_scanned, before=scan_until, limit=None):
                if message.author.id == self.bot.user.id:
                    continue

                last_message_time = message.created_at
                last_message_id = message.id
                reaction_count = self._get_all_reacts(message)
                reply_info = get_reply_info(message)
                replied_to_author_id = reply_info[3]

                # Collect reply stats if this message is a reply to someone else
                if replied_to_author_id and message.author.id != replied_to_author_id:
                    replies_to_update.append(
                        (
                            message.author.id,
                            replied_to_author_id,
                            message.created_at.year,
                            message.created_at.month,
                        )
                    )

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
                    await self.db.add_history_messages(history_messages_to_insert)
                    history_messages_to_insert.clear()

                # Skip messages with no content for stats/quality filtering
                if not message.content.strip():
                    continue

                stats_to_update.append((message.author.id, reaction_count, message.created_at))
                # Collect reactions for batching instead of writing immediately
                reaction_tuples = await self._collect_reactions(message)
                reactions_to_update.extend(reaction_tuples)

                if len(stats_to_update) >= 100:
                    await self.user_stats.batch_update_stats(stats_to_update)
                    stats_to_update.clear()

                # Batch reactions at the same threshold as stats for consistency
                if len(reactions_to_update) >= 100:
                    await self.user_stats.batch_update_reaction_stats(reactions_to_update)
                    reactions_to_update.clear()

                # Batch replies at the same threshold
                if len(replies_to_update) >= 100:
                    await self.db.batch_upsert_reply_stats(replies_to_update)
                    replies_to_update.clear()

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

        except Exception:
            error_occurred = True
            self.logger.exception(f"Error processing channel {channel.name} (ID: {channel.id})")

        # Always try to commit remaining batches (even after error)
        try:
            if history_messages_to_insert:
                await self.db.add_history_messages(history_messages_to_insert)
            if messages_to_insert:
                await self.db.add_messages(messages_to_insert)
            if attachments_to_insert:
                await self.db.add_attachments(attachments_to_insert)
            if stats_to_update:
                await self.user_stats.batch_update_stats(stats_to_update)
            if reactions_to_update:
                await self.user_stats.batch_update_reaction_stats(reactions_to_update)
            if replies_to_update:
                await self.db.batch_upsert_reply_stats(replies_to_update)
        except Exception:
            self.logger.exception(f"Error committing remaining batches for channel {channel.name}")

        # Update timestamp to last processed message (allows resume on next run)
        if last_message_time > last_scanned:
            await self.db.update_last_scanned_timestamp(channel.id, last_message_time)

        # Update message history scrape progress
        if last_message_id:
            await self.db.update_scrape_progress(channel.id, last_message_id)

        if error_occurred:
            self.logger.warning(
                f"Channel {channel.name} partially processed due to error. "
                f"Added {total_message_count} messages before failure."
            )
        else:
            # Update timestamp to scan_until only on successful completion
            await self.db.update_last_scanned_timestamp(channel.id, scan_until)
            self.logger.info(
                f"Channel {channel.name} updated. Added {total_message_count} messages."
            )

        return (total_message_count, total_reaction_count)

    async def _collect_reactions(
        self, message: discord.Message
    ) -> list[tuple[int, int, datetime.datetime]]:
        """Collect all reactions on a message for batch processing.

        Returns:
            List of (giver_id, receiver_id, timestamp) tuples.
        """
        reactions = []
        for reaction in message.reactions:
            async for user in reaction.users():
                if user.bot:
                    continue
                reactions.append((user.id, message.author.id, message.created_at))
        return reactions

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

    async def check_predictions(self, max_retries: int = 5) -> None:
        """Check and post due predictions.

        Args:
            max_retries: Maximum number of retry attempts before giving up on a prediction.
        """
        if not self.guild:
            self.logger.warning("Guild not set. Skipping prediction check.")
            return

        now = datetime.datetime.now(datetime.UTC)
        today = now.date()
        current_hour = now.hour

        due_predictions = await self.db.get_due_predictions(today, max_retries)

        if not due_predictions:
            return

        self.logger.info(f"Found {len(due_predictions)} due prediction(s) to check.")

        for prediction in due_predictions:
            is_overdue = prediction.target_date < today
            is_midday = 11 <= current_hour <= 13

            # Post today's predictions only around midday (11:00-13:00 UTC)
            # But always post overdue predictions (catch-up logic)
            if is_overdue or is_midday:
                try:
                    await self._post_prediction(prediction)
                    await self.db.mark_prediction_posted(prediction.id)
                    self.logger.info(
                        f"Posted prediction #{prediction.id} by {prediction.author_name}"
                    )
                except Exception:
                    # Increment retry count on failure
                    new_retry_count = await self.db.increment_prediction_retry(prediction.id)
                    if new_retry_count >= max_retries:
                        self.logger.error(
                            f"Prediction #{prediction.id} failed after {max_retries} attempts. "
                            f"Giving up (channel may be deleted or inaccessible)."
                        )
                    else:
                        self.logger.warning(
                            f"Error posting prediction #{prediction.id} "
                            f"(attempt {new_retry_count}/{max_retries})"
                        )

    async def _post_prediction(self, prediction) -> None:
        """Post a prediction to the predictions channel with voting reactions."""
        from strofkabot.discord_db import Prediction

        if not isinstance(prediction, Prediction):
            raise TypeError("Expected Prediction object")

        channel = self.bot.get_channel(PREDICTIONS_CHANNEL_ID)
        if not channel:
            self.logger.warning(
                f"Predictions channel {PREDICTIONS_CHANNEL_ID} not found for prediction {prediction.id}"
            )
            return

        # Build embed
        embed = discord.Embed(
            description=f"**{prediction.prediction_text}**",
            color=discord.Color.gold(),
        )

        # Try to get current user info for avatar
        member = self.guild.get_member(prediction.author_id)
        if member:
            embed.set_author(name=member.display_name, icon_url=member.display_avatar.url)
        else:
            embed.set_author(name=prediction.author_name)

        embed.set_footer(text=f"Predicted on {prediction.created_at.strftime('%B %d, %Y')}")

        message = await channel.send(embed=embed)
        await message.add_reaction("👍")
        await message.add_reaction("👎")
