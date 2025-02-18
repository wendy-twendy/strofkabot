import discord
from typing import List, Optional

async def get_member_names(guild: discord.Guild, member_ids: List[int]) -> List[str]:
    """Get display names for a list of member IDs."""
    member_names = []
    for user_id in member_ids:
        member = guild.get_member(user_id)
        if member:
            name = ''.join(char for char in member.display_name if ord(char) < 128) or f"User {user_id}"
            member_names.append(name)
        else:
            member_names.append(f"User {user_id}")
    return member_names

def get_reply_info(message: discord.Message) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Get information about a message reply."""
    reply_to_id = None
    reply_to_author = None
    reply_to_content = None

    if message.reference and message.reference.resolved:
        replied_msg = message.reference.resolved
        reply_to_id = replied_msg.id

        if isinstance(replied_msg, discord.DeletedReferencedMessage):
            reply_to_author = "Deleted User"
            reply_to_content = "Message was deleted"
        else:
            reply_to_author = replied_msg.author.display_name if replied_msg.author else "Unknown User"
            reply_to_content = replied_msg.content if hasattr(replied_msg, 'content') else "Content unavailable"

    return reply_to_id, reply_to_author, reply_to_content

async def get_non_bot_member_ids(guild: discord.Guild) -> List[int]:
    """Get IDs of all non-bot members in the guild."""
    return [member.id for member in guild.members if not member.bot]
