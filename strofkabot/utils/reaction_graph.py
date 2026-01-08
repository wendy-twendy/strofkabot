"""Reaction graph utilities for building and analyzing user interaction networks."""

from datetime import datetime
from math import cos, pi, sin, sqrt
from random import Random

import networkx as nx
import numpy as np


def build_reaction_graph(reaction_data: list[tuple[str, str, int]]) -> nx.DiGraph:
    """Build a directed graph from reaction data.

    Args:
        reaction_data: List of (giver_username, receiver_username, reaction_count) tuples.

    Returns:
        Directed graph with edge weights representing reaction counts.
    """
    G = nx.DiGraph()

    for giver, receiver, count in reaction_data:
        if G.has_edge(giver, receiver):
            G[giver][receiver]["weight"] += count
        else:
            G.add_edge(giver, receiver, weight=count)

    return G


def mutual_affinity(G: nx.DiGraph, user_a: str, user_b: str) -> float:
    """Calculate mutual affinity between two users.

    Formula:
        affinity(A, B) = sqrt(share_A->B * share_B->A)

        where:
            share_A->B = reactions A gave to B / total reactions A gave
            share_B->A = reactions B gave to A / total reactions B gave

    Args:
        G: Directed graph with reaction weights.
        user_a: First user.
        user_b: Second user.

    Returns:
        Mutual affinity score between 0 and 1.
    """
    # Get reaction counts
    a_to_b = G[user_a][user_b]["weight"] if G.has_edge(user_a, user_b) else 0
    b_to_a = G[user_b][user_a]["weight"] if G.has_edge(user_b, user_a) else 0

    # Get total outgoing reactions for each user
    a_total = sum(d["weight"] for _, _, d in G.out_edges(user_a, data=True))
    b_total = sum(d["weight"] for _, _, d in G.out_edges(user_b, data=True))

    if a_total == 0 or b_total == 0:
        return 0.0

    # Normalized shares
    share_a_to_b = a_to_b / a_total
    share_b_to_a = b_to_a / b_total

    # Geometric mean
    return sqrt(share_a_to_b * share_b_to_a)


def compute_all_affinities(G: nx.DiGraph) -> list[tuple[str, str, float]]:
    """Compute mutual affinity for all pairs with bidirectional edges.

    Args:
        G: Directed graph with reaction weights.

    Returns:
        Sorted list of (user_a, user_b, affinity) tuples, highest first.
    """
    affinities = []
    seen_pairs = set()

    for u, v in G.edges():
        # Only process bidirectional edges
        if G.has_edge(v, u):
            # Avoid duplicates (A,B) and (B,A)
            pair = tuple(sorted([u, v]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                affinity = mutual_affinity(G, u, v)
                affinities.append((pair[0], pair[1], affinity))

    # Sort by affinity descending
    affinities.sort(key=lambda x: x[2], reverse=True)
    return affinities


def compute_top_connection_per_user(
    affinities: list[tuple[str, str, float]], active_users: set[str]
) -> list[tuple[str, str, float]]:
    """Find the top connection for each active user.

    Args:
        affinities: List of (user_a, user_b, affinity) tuples from compute_all_affinities().
        active_users: Set of usernames considered active (e.g., 30+ messages).

    Returns:
        List of (user, partner, affinity) tuples sorted by affinity descending.
        Each active user appears at most once with their strongest connection.
    """
    user_top: dict[str, tuple[str, float]] = {}

    # affinities are already sorted by score descending
    for user_a, user_b, affinity in affinities:
        # Check if user_a is active and doesn't have a top connection yet
        if user_a in active_users and user_a not in user_top:
            user_top[user_a] = (user_b, affinity)
        # Check if user_b is active and doesn't have a top connection yet
        if user_b in active_users and user_b not in user_top:
            user_top[user_b] = (user_a, affinity)

    # Convert to list and sort by affinity descending
    result = [(user, partner, affinity) for user, (partner, affinity) in user_top.items()]
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def build_affinity_graph(
    G: nx.DiGraph,
    affinities: list[tuple[str, str, float]],
    min_affinity: float = 0.02,
    remove_isolated: bool = True,
) -> nx.Graph:
    """Build undirected graph with mutual affinity as edge weights.

    Args:
        G: Original directed graph (for node data).
        affinities: List of (user_a, user_b, affinity) tuples.
        min_affinity: Minimum affinity to include edge.
        remove_isolated: If True, remove nodes with no edges.

    Returns:
        Undirected graph with affinity weights.
    """
    affinity_G = nx.Graph()

    # Add edges with affinity weights (filtered by threshold)
    for user_a, user_b, affinity in affinities:
        if affinity >= min_affinity:
            affinity_G.add_edge(user_a, user_b, affinity=affinity)

    # Optionally add isolated nodes
    if not remove_isolated:
        affinity_G.add_nodes_from(G.nodes())

    return affinity_G


def detect_communities(G: nx.DiGraph, seed: int = 42, resolution: float = 1.0) -> dict[str, int]:
    """Detect communities using Louvain algorithm.

    Args:
        G: Directed graph (will be converted to undirected for Louvain).
        seed: Random seed for reproducibility.
        resolution: Higher values = more smaller communities (default 1.0).

    Returns:
        Dict mapping username -> community_id.
    """
    # Convert to undirected graph, summing weights
    simple_undirected = nx.Graph()

    # Add nodes in sorted order for determinism
    for node in sorted(G.nodes()):
        simple_undirected.add_node(node)

    # Combine edges by summing weights
    edge_weights = {}
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        weight = data.get("weight", 1)
        edge_weights[key] = edge_weights.get(key, 0) + weight

    # Add edges in sorted order for determinism
    for (u, v), weight in sorted(edge_weights.items()):
        simple_undirected.add_edge(u, v, weight=weight)

    # Run Louvain community detection
    communities = nx.community.louvain_communities(
        simple_undirected, weight="weight", seed=seed, resolution=resolution
    )

    # Sort communities by size (largest first) for consistent numbering
    sorted_communities = sorted(communities, key=lambda x: (-len(x), min(x)))

    # Build user -> community_id mapping
    community_map = {}
    for community_id, members in enumerate(sorted_communities):
        for member in members:
            community_map[member] = community_id

    return community_map


def compute_node_activity(G: nx.DiGraph) -> dict[str, float]:
    """Compute activity level for each node (reactions given + received).

    Args:
        G: Directed graph with reaction weights.

    Returns:
        Dict mapping username -> activity level.
    """
    activity = {}
    for node in G.nodes():
        given = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        received = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        activity[node] = given + received
    return activity


def compute_community_layout(
    G: nx.Graph,
    communities: dict[str, int] | None = None,
    isolated_nodes: list[str] | None = None,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute node positions using Kamada-Kawai layout with optional community seeding.

    Args:
        G: Undirected graph with affinity weights.
        communities: Dict mapping node -> community_id (None for no community seeding).
        isolated_nodes: List of nodes without edges to position around periphery.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping node -> (x, y) position.
    """
    rng = Random(seed)
    community_centers = {}

    if communities:
        # Find unique communities
        unique_communities = sorted(set(communities.values()))
        num_communities = len(unique_communities)

        # Place community centers on a circle
        for i, comm_id in enumerate(unique_communities):
            angle = 2 * pi * i / num_communities - pi / 2  # Start from top
            community_centers[comm_id] = (cos(angle) * 2, sin(angle) * 2)

        # Initialize node positions near their community center
        initial_pos = {}
        for node in G.nodes():
            comm_id = communities.get(node, 0)
            cx, cy = community_centers[comm_id]
            jitter_x = rng.uniform(-0.5, 0.5)
            jitter_y = rng.uniform(-0.5, 0.5)
            initial_pos[node] = (cx + jitter_x, cy + jitter_y)
    else:
        initial_pos = None

    # Use Kamada-Kawai for balanced layout
    pos = nx.kamada_kawai_layout(
        G,
        pos=initial_pos,
        weight="affinity",
        scale=1.0,
    )

    # Add isolated nodes around the periphery
    if isolated_nodes:
        # Find bounds of main graph
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            max_radius = max(max(abs(min(xs)), abs(max(xs))), max(abs(min(ys)), abs(max(ys))))
        else:
            max_radius = 1.0

        periphery_radius = max_radius * 1.15
        for i, node in enumerate(isolated_nodes):
            if communities:
                comm_id = communities.get(node, 0)
                cx, cy = community_centers.get(comm_id, (0, 0))
                base_angle = np.arctan2(cy, cx) if (cx != 0 or cy != 0) else 0
            else:
                base_angle = 2 * pi * i / len(isolated_nodes)

            angle_offset = (i / max(len(isolated_nodes), 1)) * pi * 0.3 - pi * 0.15
            angle = base_angle + angle_offset
            jitter = rng.uniform(-0.1, 0.1)
            pos[node] = (
                cos(angle) * (periphery_radius + jitter),
                sin(angle) * (periphery_radius + jitter),
            )

    return pos


def get_rolling_start_month(num_months: int = 1) -> tuple[int, int]:
    """Get the start year and month for rolling window.

    The window includes the current month and goes back num_months.
    E.g., if today is Jan 7 and num_months=3, returns November (Nov, Dec, Jan = 3 months).

    Args:
        num_months: Number of months to include in the window.

    Returns:
        Tuple of (start_year, start_month).
    """
    now = datetime.now()
    year = now.year
    # Include current month: go back (num_months - 1) from current
    month = now.month - num_months + 1

    while month <= 0:
        month += 12
        year -= 1

    return year, month


def format_period_string(start_year: int, start_month: int, num_months: int) -> str:
    """Format a human-readable period string.

    Args:
        start_year: Start year of the period.
        start_month: Start month of the period.
        num_months: Number of months in the period.

    Returns:
        Human-readable period string.
    """
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    if num_months == 1:
        return f"{month_names[start_month]} {start_year}"
    else:
        # Calculate end month from start + num_months - 1
        end_year = start_year
        end_month = start_month + num_months - 1
        while end_month > 12:
            end_month -= 12
            end_year += 1
        return f"{month_names[start_month]} {start_year} - {month_names[end_month]} {end_year}"
