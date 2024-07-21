""" discord_db 
    This module implement DiscordDatabase, a rudimentary interface 
    for storing discord messages in a key-value database.
"""
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class MessageData:
    id: int = 0
    content: str = ""
    datetime: str = ""
class DiscordDatabase():
    """ Wrapper around a key-value database for Discord message storage 
        Messages are stored by their id an datetime.
        See https://discordpy.readthedocs.io/en/latest/api.html#message 
        section *id* and *created_at*
    """

    def __init__(self, database_path: Path, channel_name: str, initialize: bool = False):
        """ Create a Discord database object from database_path 
        If database_path is doesn't exist, it will be created. 
        If inititalize is set to true, the old database will be discared
        """
        print("Debug: Create database at location")
        self.db_file = database_path
        if self.db_file.is_file() and initialize:
            self.db_file.unlink()
            print("Debug: Deleting existing database at location")

        self.scanning_table="scanning_table"
        self.scanning_table_initial_element= MessageData(datetime="2017-01-01 00:00:00.000000+00:00")

        self.con = sqlite3.connect(self.db_file)
        self.cursor = self.con.cursor()

        self.channel_name = channel_name
        self._create_channel_table(self.channel_name)
        self._create_scanning_table()

        # Initialiaze scanning table
        if self.is_message_stored(self.scanning_table,0) is False:
            self._add_element_to_table(self.scanning_table, self.scanning_table_initial_element)

    def add_message(self, message: MessageData):
        """ Add a message to the database
            Throws an exception if the message was not stored in the database
        """
        self._add_element_to_table(self.channel_name, message)

    def update_scanning_table(self, message: MessageData):
        """ Add the message element to the scanning table """
        self._delete_table_content(self.scanning_table)
        self._add_element_to_table(self.scanning_table, message)

    def get_random_message(self)->MessageData:
        """ Return the tupple of a message stored in the database 
            Each message has the same draw probability
        """

        random_message_query = f"SELECT * FROM {self.channel_name} ORDER BY RANDOM() LIMIT 1;"
        self.cursor.execute(random_message_query)
        random_row = self.cursor.fetchone()

        return self._get_message_from_tuple_message(random_row)
        

    def get_last_message(self)->MessageData:
        """ Return the last message added to the table """
        return self._get_last_table_element(self.channel_name, "id")

    def get_last_scanned_message(self)->MessageData:
        """ Return the most recent scanned message """
        return self._get_last_table_element(self.scanning_table, "id")

    def _get_last_table_element(self, table_name:str, desc_value: str)->MessageData:
        last_message_query = f"SELECT * FROM {table_name} ORDER BY {desc_value} DESC LIMIT 1;"
        self.cursor.execute(last_message_query)
        last_message = self.cursor.fetchone()

        return self._get_message_from_tuple_message(last_message)

    def is_message_stored(self, channel_name:str, message_id: int) -> bool:
        """ Returns true only if the message is stored in the database
            Return false
        """

        check_query = f"SELECT EXISTS(SELECT 1 FROM {channel_name} WHERE id = ? LIMIT 1);"
        self.cursor.execute(check_query, (message_id,))
        exists = self.cursor.fetchone()[0]

        return exists == 1

    def get_message_count(self) -> int:
        """ Return the count of messages stored in the table """     
        messages_count_query = f"SELECT COUNT(*) FROM {self.channel_name};"

        # Execute the query and fetch the result
        self.cursor.execute(messages_count_query)
        row_count = self.cursor.fetchone()[0]

        return row_count

    def _create_channel_table(self, channel_name: str):
        """ Create a dedicated table for the channel_namel on Strofka database
            If inititalize is set to true, the existing table will be discared
        """

        create_table_query = f"""
         CREATE TABLE IF NOT EXISTS {channel_name} (
            id INTEGER PRIMARY KEY,
            content STRING,
            datetime STRING
        );
        """

        self.cursor.execute(create_table_query)
        self.con.commit()
    
    def _create_scanning_table(self):
        """ Create a cell that stores the last scanned element messages
        """

        create_table_query = f"""
         CREATE TABLE IF NOT EXISTS {self.scanning_table} (
            id INTEGER PRIMARY KEY,
            content STRING,
            datetime STRING
        );
        """

        self.cursor.execute(create_table_query)
        self.con.commit()

    def _add_element_to_table(self, _table_name:str, message: MessageData):
        insert_query = f"""
            INSERT OR REPLACE INTO {_table_name} (id, content, datetime)
            VALUES (?,?,?);
        """
        self.cursor.execute(insert_query, (message.id, message.content, message.datetime))
        self.con.commit()

    def _delete_table_content(self, _table_name):
        self.cursor.execute(f"DELETE FROM {_table_name}")
        self.con.commit()

    def _get_message_from_tuple_message(self, tuple_message:tuple[int, str]) -> MessageData:
        return MessageData(id=tuple_message[0], content=tuple_message[1], datetime=tuple_message[2])
