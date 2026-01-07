from datetime import datetime, date

def adjust_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Adjust year and month by the given offset."""
    new_month = month + offset
    new_year = year
    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -= 1
    return new_year, new_month

def get_previous_month(date: date = None) -> tuple[int, int]:
    """Get the year and month of the previous month."""
    if date is None:
        date = datetime.now().date()
    return adjust_month(date.year, date.month, -1)
