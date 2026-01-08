"""Date arithmetic utilities."""

from datetime import datetime

import dateparser


def adjust_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Adjust year and month by an offset, handling year boundaries.

    Args:
        year: The starting year
        month: The starting month (1-12)
        offset: Number of months to add (positive) or subtract (negative)

    Returns:
        Tuple of (new_year, new_month)
    """
    new_month = month + offset
    new_year = year
    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -= 1
    return new_year, new_month


def parse_prediction_date(args: str) -> tuple[datetime | None, str]:
    """Parse date and text from prediction args.

    Args:
        args: String containing a date prefix followed by prediction text

    Returns:
        Tuple of (parsed datetime or None, remaining prediction text).
    """
    words = args.split()
    parsed_date = None
    date_word_count = 0

    # Try progressively longer prefixes as dates (up to 4 words)
    for i in range(1, min(len(words) + 1, 5)):
        candidate = " ".join(words[:i])
        result = dateparser.parse(
            candidate,
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": "UTC",
                "DATE_ORDER": "DMY",
            },
        )
        if result:
            parsed_date = result
            date_word_count = i

    prediction_text = " ".join(words[date_word_count:]) if date_word_count > 0 else ""
    return parsed_date, prediction_text
