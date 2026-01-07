"""Data fetching, calculation, and analysis functions."""

import datetime
from typing import Any

import discord
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


async def fetch_inflation_data(user_stats) -> tuple[list[dict], list[dict]]:
    """Fetch monthly and yearly reaction inflation data."""
    monthly_data = await user_stats.get_reaction_inflation_raw(monthly=True, limit=12)
    yearly_data = await user_stats.get_reaction_inflation_raw(monthly=False)
    return monthly_data, yearly_data


def calculate_monthly_inflation(monthly_data: list[dict]) -> list[dict]:
    """Calculate month-over-month inflation percentages."""
    monthly_inflation = []
    prev_avg = None
    for record in reversed(monthly_data):
        if prev_avg is not None:
            change = ((record["average_rpm"] - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0
            monthly_inflation.append(
                {
                    "month_year": f"{record['year']}-{record['month']:02d}",
                    "change_percentage": round(change, 2),
                    "average_rpm": record["average_rpm"],
                    "total_reactions": record.get("total_reactions", 0),
                    "total_messages": record.get("total_messages", 0),
                }
            )
        prev_avg = record["average_rpm"]
    return monthly_inflation


def calculate_yearly_inflation(yearly_data: list[dict]) -> list[dict]:
    """Calculate year-over-year inflation percentages."""
    yearly_inflation = []
    prev_avg = None
    for record in yearly_data:
        if prev_avg is not None:
            change = ((record["average_rpm"] - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0
            yearly_inflation.append(
                {
                    "year": record["year"],
                    "change_percentage": round(change, 2),
                    "average_rpm": record["average_rpm"],
                    "total_reactions": record.get("total_reactions", 0),
                    "total_messages": record.get("total_messages", 0),
                }
            )
        prev_avg = record["average_rpm"]
    return yearly_inflation


async def fetch_gdp_data(user_stats, show_all: bool = False) -> list[dict]:
    """Fetch total messages per month from user_stats_monthly.

    Args:
        user_stats: UserStats instance for database access.
        show_all: If True, returns all data without limit. If False, limits to 24 months.
    """
    limit = None if show_all else 24
    return await user_stats.get_gdp_data(limit=limit)


async def fetch_hdi_data(user_stats) -> list[dict]:
    """Fetch messages and calculate HDI (quality messages / total messages) per month."""
    return await user_stats.get_hdi_data(limit=24)


async def get_reaction_trade_data(user_stats, user_id: int, guild: discord.Guild) -> dict[str, Any]:
    """Fetch reaction trade data for a specific user from the past year."""
    one_year_ago = datetime.datetime.now() - datetime.timedelta(days=365)
    year, month = one_year_ago.year, one_year_ago.month

    # Get raw data from DAL
    raw_data = await user_stats.get_reaction_trade_data(user_id, year, month, limit=5)

    # Resolve Discord member names for exports
    exports = []
    for receiver_id, count in raw_data["exports"]:
        member = guild.get_member(receiver_id)
        name = member.display_name if member else f"User {receiver_id}"
        exports.append((name, count))

    # Resolve Discord member names for imports
    imports = []
    for giver_id, count in raw_data["imports"]:
        member = guild.get_member(giver_id)
        name = member.display_name if member else f"User {giver_id}"
        imports.append((name, count))

    return {
        "exports": exports,
        "imports": imports,
        "total_given": raw_data["total_given"],
        "total_received": raw_data["total_received"],
        "trade_balance": raw_data["trade_balance"],
    }


async def get_reaction_trade_data_for_month(
    user_stats, user_id: int, year: int, month: int, guild: discord.Guild
) -> dict[str, Any]:
    """Fetch reaction trade data for a specific user for a single month.

    Args:
        user_stats: UserStats instance for database access.
        user_id: The Discord user ID.
        year: The year to query.
        month: The month to query.
        guild: Discord guild for member name resolution.

    Returns:
        Dictionary with exports, imports, total_given, total_received, and trade_balance,
        with member names resolved.
    """
    # Get raw data from business logic layer
    raw_data = await user_stats.get_reaction_trade_data_for_month(user_id, year, month, limit=5)

    # Resolve Discord member names for exports
    exports = []
    for receiver_id, count in raw_data["exports"]:
        member = guild.get_member(receiver_id)
        name = member.display_name if member else f"User {receiver_id}"
        exports.append((name, count))

    # Resolve Discord member names for imports
    imports = []
    for giver_id, count in raw_data["imports"]:
        member = guild.get_member(giver_id)
        name = member.display_name if member else f"User {giver_id}"
        imports.append((name, count))

    return {
        "exports": exports,
        "imports": imports,
        "total_given": raw_data["total_given"],
        "total_received": raw_data["total_received"],
        "trade_balance": raw_data["trade_balance"],
    }


def calculate_reaction_percentage(reaction_graph: np.ndarray) -> np.ndarray:
    """Convert reaction counts to percentages per user."""
    row_sums = reaction_graph.sum(axis=1, keepdims=True)
    percentage = np.zeros_like(reaction_graph, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(reaction_graph, row_sums, out=percentage, where=row_sums != 0)
        percentage *= 100
    return percentage


def prepare_clustering_data(
    guild: discord.Guild, reactions_data: dict
) -> tuple[np.ndarray, list[str]]:
    """Prepare data for K-means clustering."""
    data = []
    member_names = []
    for user_id, stats in reactions_data.items():
        data.append([stats.get("given", 0), stats.get("received", 0)])
        member = guild.get_member(user_id)
        member_names.append(member.display_name if member else f"User {user_id}")
    return np.array(data), member_names


def perform_kmeans_clustering(data: np.ndarray, num_clusters: int = 4) -> np.ndarray:
    """Perform K-means clustering on reaction data."""
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data_normalized)
    return labels


async def fetch_hourly_activity_data(
    message_history_db, user_id: int, timezone_offset: int = 0
) -> np.ndarray | None:
    """Fetch hourly activity data for a user and format as 2D array.

    Args:
        message_history_db: MessageHistoryDatabase instance.
        user_id: Discord user ID.
        timezone_offset: Hours offset from UTC.

    Returns:
        7x24 numpy array where rows are days (Mon-Sun) and columns are hours (0-23).
        Returns None if no data found.
    """
    raw_data = await message_history_db.get_hourly_activity_by_user(user_id, timezone_offset)

    if not raw_data:
        return None

    # Initialize 7x24 matrix (days x hours)
    activity_matrix = np.zeros((7, 24), dtype=int)

    # SQLite %w: 0=Sunday, 1=Monday, ..., 6=Saturday
    # We want: 0=Monday, 1=Tuesday, ..., 6=Sunday
    # Conversion: (sqlite_day - 1) % 7 maps Sun(0)->6, Mon(1)->0, Tue(2)->1, etc.

    for sqlite_day, hour, count in raw_data:
        # Convert SQLite day (0=Sun) to display day (0=Mon)
        display_day = (sqlite_day - 1) % 7
        activity_matrix[display_day, hour] = count

    return activity_matrix
