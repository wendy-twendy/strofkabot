import os
from pathlib import Path
import datetime
import re
import discord
from discord_db import DiscordDatabase, MessageData
from message_filter import MessageFilter


class LlumiBot(discord.Client):
    def __init__(self, database_location: Path, channel_name: str, discord_intents: discord.Intents):
        super().__init__(intents=discord_intents)
        self.db = DiscordDatabase(database_location, channel_name)
        self.channel_name = "General"
        self.guild_id = 413619835096924160
        self.channel_id = 430696233003122698
        self.guild = None
        self.channel = None
        self.react_count = 4
        print(f"Starting llumibot for channel {self.channel_id} on server {self.guild_id}.")
        self.message_filter = MessageFilter()

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        self.guild = self.get_guild(self.guild_id)
        self.channel = self._get_text_channel(self.channel_id)

        if self.channel is not None:
             await self.update_db()
        else:
            print(f"Channel id {self.channel_id} was not found on the server. Skipping Database Update")

    async def on_message(self, message):
        if message.author.id in [325678635388502016,301411562487545857]:
            if message.content == "!llumi":
                content = self.db.get_random_message().content
                await message.channel.send(content)
        
    async def update_db(self):
        while True:
            previous_scanned_timestamp = self._get_latest_timestamp()
            await self._update_db_block(after=previous_scanned_timestamp)
            if previous_scanned_timestamp == self._get_latest_timestamp():
                # No new message scanned. Update done
                print("Update done")
                break
            else:
                print(f"Scanning timestamp: {previous_scanned_timestamp}")

    async def _update_db_block(self, after, limit=100):
        ts_last_scanned_message = after
        async for message in self.channel.history(limit=limit,after=after):
            if message.created_at > ts_last_scanned_message:
                ts_last_scanned_message = message.created_at
            if len (message.reactions) >= self.react_count and self.message_filter.is_valid_message(message.content):
                print(f"Id: {message.id}, content: {message.content}, created at {message.created_at.isoformat()}" )
                self.db.add_message(MessageData(id=message.id , content= message.content, datetime= str(message.created_at) ))
        
        self.db.update_scanning_table(MessageData(datetime=str(ts_last_scanned_message)))

    def _get_all_reacts(self, message: discord.Message)-> int:
        count = 0
        for react in message.reactions:
            count += react.count

    def _get_latest_timestamp(self):
        sync_element = self.db.get_last_scanned_message()
        return datetime.datetime.fromisoformat(sync_element.datetime)

    def _get_text_channel(self, channel_id: str):
        text_channels = self.guild.text_channels
        for channel in text_channels:
            if channel.id == channel_id:
                return channel
            
        return None

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    DATABASE_FILE_LOCATION = "db.sqlite3"
    client = LlumiBot(database_location = Path(DATABASE_FILE_LOCATION), channel_name="llumi_3_reacts", discord_intents=intents)
    client.run(os.getenv("LLUMI_BOT_TOKEN"))
