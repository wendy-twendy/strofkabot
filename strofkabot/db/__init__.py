# db/__init__.py

"""Database package composing all mixins into the unified Database class."""

from strofkabot.db.attachments import Attachment, AttachmentsMixin
from strofkabot.db.base import BaseDatabase
from strofkabot.db.message_history import HistoryMessage, MessageHistoryMixin
from strofkabot.db.messages import Message, MessagesMixin
from strofkabot.db.on_this_day import OnThisDayMixin
from strofkabot.db.predictions import Prediction, PredictionsMixin
from strofkabot.db.stats import StatsMixin


class Database(
    MessagesMixin,
    AttachmentsMixin,
    StatsMixin,
    PredictionsMixin,
    OnThisDayMixin,
    MessageHistoryMixin,
    BaseDatabase,
):
    """Unified database access layer combining all domain mixins."""

    pass


__all__ = [
    "Database",
    "Message",
    "Attachment",
    "Prediction",
    "HistoryMessage",
]
