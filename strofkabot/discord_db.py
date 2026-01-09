# discord_db.py

"""Database access layer for all SQLite operations.

This module serves as a facade, re-exporting the Database class and dataclasses
from the db package for backward compatibility.
"""

from strofkabot.db import Attachment, Database, Message, Prediction

__all__ = ["Database", "Message", "Attachment", "Prediction"]
