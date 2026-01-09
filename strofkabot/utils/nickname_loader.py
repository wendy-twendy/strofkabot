"""Nickname loader utility for mapping user IDs to custom display names."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_nicknames(path: Path) -> dict[int, list[str]]:
    """Load nicknames from a YAML file.

    Args:
        path: Path to the nicknames YAML file.

    Returns:
        Dict mapping user IDs to lists of nicknames.
        Returns empty dict if file doesn't exist or is invalid.
    """
    if not path.exists():
        logger.warning("Nicknames file not found: %s", path)
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "nicknames" not in data:
            return {}

        nicknames = data["nicknames"]
        if not nicknames:
            return {}

        return {int(user_id): nicks for user_id, nicks in nicknames.items()}

    except (yaml.YAMLError, ValueError) as e:
        logger.warning("Failed to load nicknames file %s: %s", path, e)
        return {}


def get_display_name(
    user_id: int,
    fallback: str,
    nicknames: dict[int, list[str]] | None,
) -> str:
    """Get the display name for a user, preferring nickname if available.

    Args:
        user_id: Discord user ID.
        fallback: Name to use if no nickname is found.
        nicknames: Dict mapping user IDs to nickname lists.

    Returns:
        First nickname if found, otherwise the fallback name.
    """
    if not nicknames:
        return fallback

    nick_list = nicknames.get(user_id)
    if nick_list:
        return nick_list[0]

    return fallback
