"""Date arithmetic utilities."""


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
