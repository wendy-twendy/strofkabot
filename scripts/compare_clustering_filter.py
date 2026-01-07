#!/usr/bin/env python3
"""Compare clustering with and without low-activity user filtering."""

import asyncio
import sys
from pathlib import Path
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strofkabot.config import DATABASE_FILE_LOCATION
from strofkabot.discord_db import Database
from strofkabot.user_stats import UserStats
from strofkabot.utils.reaction_graph import (
    build_reaction_graph,
    compute_node_activity,
    detect_communities,
    get_rolling_start_month,
)


def filter_graph_by_activity(G, min_activity: int):
    """Remove nodes with activity below threshold."""
    activity = compute_node_activity(G)
    nodes_to_remove = [n for n, a in activity.items() if a < min_activity]
    filtered = G.copy()
    filtered.remove_nodes_from(nodes_to_remove)
    return filtered, nodes_to_remove, activity


def analyze_clusters(communities: dict[str, int]) -> dict:
    """Analyze cluster distribution."""
    if not communities:
        return {"num_clusters": 0, "sizes": [], "singletons": 0}

    # Count members per cluster
    cluster_sizes = {}
    for _user, cluster_id in communities.items():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    sizes = list(cluster_sizes.values())
    singletons = sum(1 for s in sizes if s == 1)

    return {
        "num_clusters": len(sizes),
        "sizes": sizes,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": mean(sizes),
        "std_size": stdev(sizes) if len(sizes) > 1 else 0,
        "singletons": singletons,
        "total_users": len(communities),
    }


async def main():
    months_to_test = [1, 3]  # Test both time windows
    thresholds = [0, 5, 10, 20, 50]  # 0 = no filtering

    # Connect to database
    db = Database(DATABASE_FILE_LOCATION)
    await db.initialize()
    user_stats = UserStats(db)

    for num_months in months_to_test:
        print(f"\n{'#'*60}")
        print(f"# TESTING {num_months} MONTH(S)")
        print(f"{'#'*60}")

        # Get rolling window start
        start_year, start_month = get_rolling_start_month(num_months)
        print(f"Analyzing {num_months} months from {start_year}-{start_month:02d}\n")

        # Fetch reaction network
        reaction_data = await user_stats.get_reaction_network_rolling(start_year, start_month)
        print(f"Total reaction edges: {len(reaction_data)}")

        # Build graph
        G = build_reaction_graph(reaction_data)
        print(f"Total users in graph: {G.number_of_nodes()}")

        if G.number_of_nodes() == 0:
            print("No users in graph, skipping...")
            continue

        # Get activity levels
        activity = compute_node_activity(G)
        sorted_activity = sorted(activity.items(), key=lambda x: x[1])

        print(f"\n{'='*60}")
        print("ACTIVITY DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Min: {sorted_activity[0][1]} ({sorted_activity[0][0]})")
        print(f"Max: {sorted_activity[-1][1]} ({sorted_activity[-1][0]})")
        print(f"Median: {sorted_activity[len(sorted_activity)//2][1]}")

        # Show users below each threshold
        for threshold in thresholds[1:]:
            below = [u for u, a in activity.items() if a < threshold]
            print(f"Users with <{threshold} reactions: {len(below)}")

        print(f"\n{'='*60}")
        print("CLUSTERING COMPARISON")
        print(f"{'='*60}")

        for threshold in thresholds:
            if threshold == 0:
                filtered_G = G
                removed = []
            else:
                filtered_G, removed, _ = filter_graph_by_activity(G, threshold)

            if filtered_G.number_of_nodes() == 0:
                print(f"\nThreshold {threshold}: No users remaining!")
                continue

            communities = detect_communities(filtered_G, resolution=1.5)
            stats = analyze_clusters(communities)

            print(f"\n--- Threshold: {threshold} (min reactions) ---")
            print(f"Users removed: {len(removed)}")
            print(f"Users clustered: {stats['total_users']}")
            print(f"Number of clusters: {stats['num_clusters']}")
            print(
                f"Cluster sizes: min={stats['min_size']}, max={stats['max_size']}, "
                f"avg={stats['avg_size']:.1f}, std={stats['std_size']:.1f}"
            )
            print(f"Singleton clusters (1 member): {stats['singletons']}")

            if removed and threshold > 0:
                print("Removed users (lowest activity):")
                removed_with_activity = [(u, activity[u]) for u in removed[:10]]
                for user, act in sorted(removed_with_activity, key=lambda x: x[1]):
                    print(f"  - {user}: {act} reactions")
                if len(removed) > 10:
                    print(f"  ... and {len(removed) - 10} more")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
