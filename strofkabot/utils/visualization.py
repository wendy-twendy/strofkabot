"""Visualization and plotting functions."""

import io
from math import sqrt

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import ListedColormap


def create_monthly_inflation_plot(monthly_inflation: list[dict]) -> io.BytesIO:
    """Create a bar plot showing monthly reaction inflation."""
    plt.figure(figsize=(10, 6))
    months = [record["month_year"] for record in monthly_inflation]
    changes = [record["change_percentage"] for record in monthly_inflation]
    averages = [record["average_rpm"] for record in monthly_inflation]

    ax = sns.barplot(x=months, y=changes, hue=months, palette="viridis", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.title("Monthly Reaction Inflation (Last 12 Months)")
    plt.grid(True, linestyle="--", alpha=0.7)

    for i, v in enumerate(averages):
        ax.text(i, changes[i], f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def create_yearly_inflation_plot(yearly_inflation: list[dict]) -> io.BytesIO:
    """Create a bar plot showing yearly reaction inflation."""
    plt.figure(figsize=(8, 6))
    years = [record["year"] for record in yearly_inflation]
    changes_yearly = [record["change_percentage"] for record in yearly_inflation]
    averages_yearly = [record["average_rpm"] for record in yearly_inflation]

    ax = sns.barplot(x=years, y=changes_yearly, hue=years, palette="deep", legend=False)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xticks(rotation=0)
    plt.xlabel("Year")
    plt.ylabel("YoY Inflation (%)")
    plt.title("Yearly Reaction Inflation")
    plt.grid(True, linestyle="--", alpha=0.7)

    for i, v in enumerate(averages_yearly):
        ax.text(i, changes_yearly[i], f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def determine_figure_size(num_users: int) -> float:
    """Calculate appropriate figure size based on number of users."""
    return max(24, num_users * 0.6)


def generate_reaction_matrix_plot(
    data: np.ndarray, labels: list[str], fig_size: float
) -> io.BytesIO:
    """Generate a heatmap of reaction interactions between users."""
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        data,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlGnBu",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.xticks(rotation=90, ha="center", fontsize=16)
    plt.yticks(rotation=0, va="center", fontsize=16)

    plt.title("Reaction Matrix (Percentage of Reactions Given)", fontsize=22, pad=20)
    plt.xlabel("Reactions Received From", fontsize=18, labelpad=15)
    plt.ylabel("Reactions Given To", fontsize=18, labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf


def generate_cluster_plot(
    data: np.ndarray, labels: np.ndarray, member_names: list[str]
) -> io.BytesIO:
    """Generate a scatter plot of user clusters based on reactions."""
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(sns.color_palette("hsv", np.unique(labels).size).as_hex())
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=cmap, legend="full")
    plt.xlabel("Reactions Given")
    plt.ylabel("Reactions Received")
    plt.title("User Clusters Based on Reactions")
    plt.legend(title="Cluster")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def create_gdp_plot(data: list[dict]) -> io.BytesIO:
    """Create a line plot of server GDP (messages per month)."""
    data = data[::-1]
    n_points = len(data)

    # Dynamic figure width based on data size
    fig_width = min(24, 12 + max(0, n_points - 24) * 0.15)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    messages = [record["total_messages"] for record in data]

    ax.plot(
        range(n_points),
        messages,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#ff7f0e",
    )
    ax.fill_between(range(n_points), messages, alpha=0.2, color="#ff7f0e")

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Messages")
    ax.set_title("Server GDP (Total Messages per Month)")

    # Determine x-axis label interval based on data size
    if n_points <= 12:
        x_interval = 1
    elif n_points <= 36:
        x_interval = 3
    elif n_points <= 60:
        x_interval = 6
    else:
        x_interval = 12

    # Set x-ticks at intervals
    tick_indices = list(range(0, n_points, x_interval))
    if (n_points - 1) not in tick_indices:
        tick_indices.append(n_points - 1)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([months[i] for i in tick_indices], rotation=45, ha="right")

    # Determine which count labels to show
    if n_points <= 24:
        label_indices = set(range(n_points))
    else:
        # Interval-based labels
        if n_points <= 48:
            label_interval = 3
        elif n_points <= 84:
            label_interval = 6
        else:
            label_interval = 12
        label_indices = set(range(0, n_points, label_interval))
        # Always include first, last, max, min
        label_indices.add(0)
        label_indices.add(n_points - 1)
        label_indices.add(messages.index(max(messages)))
        label_indices.add(messages.index(min(messages)))

    # Draw count labels only for selected indices
    max_val = max(messages)
    for i in label_indices:
        ax.text(
            i,
            messages[i] + (max_val * 0.02),
            str(messages[i]),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close()
    return buf


def create_hdi_plot(data: list[dict]) -> io.BytesIO:
    """Create a line plot of server HDI (quality messages ratio)."""
    plt.figure(figsize=(12, 6))
    data = data[::-1]
    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    hdi_values = [record["hdi_ratio"] for record in data]

    plt.plot(
        months, hdi_values, marker="o", linestyle="-", linewidth=2, markersize=8, color="#2ecc71"
    )
    plt.fill_between(months, hdi_values, alpha=0.2, color="#2ecc71")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("HDI Ratio (Quality/Total Messages)")
    plt.title("Server HDI (Quality Messages Ratio per Month)")

    for i, v in enumerate(hdi_values):
        plt.text(i, v + (max(hdi_values) * 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    plt.close()
    return buf


def create_reaction_graph_plot(
    affinity_graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    communities: dict[str, int] | None,
    activity: dict[str, float],
    period_str: str = "",
) -> io.BytesIO:
    """Generate network visualization of reaction interactions.

    Args:
        affinity_graph: Undirected graph with affinity edge weights.
        positions: Dict mapping node -> (x, y) position.
        communities: Dict mapping node -> community_id (or None for no communities).
        activity: Dict mapping node -> activity level (for node sizing).
        period_str: Human-readable time period string.

    Returns:
        BytesIO buffer containing the PNG image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    # Color palette for communities
    community_palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
    ]

    if communities:
        unique_communities = sorted(set(communities.values()))
        community_colors = {
            comm_id: community_palette[i % len(community_palette)]
            for i, comm_id in enumerate(unique_communities)
        }
    else:
        unique_communities = []
        community_colors = {}

    # Normalize activity for node sizing
    graph_nodes = set(affinity_graph.nodes())
    node_activities = {n: activity.get(n, 0) for n in graph_nodes}
    max_activity = max(node_activities.values()) if node_activities else 1
    min_activity = min(node_activities.values()) if node_activities else 0
    activity_range = max_activity - min_activity if max_activity > min_activity else 1

    # Draw edges
    for u, v, data in affinity_graph.edges(data=True):
        affinity = data.get("affinity", 0)

        # Edge styling scales with affinity
        width = 0.8 + 3 * affinity
        alpha = 0.2 + 0.45 * affinity

        if communities:
            u_comm = communities.get(u, -1)
            v_comm = communities.get(v, -1)
            same_community = u_comm == v_comm and u_comm != -1

            if same_community:
                color = community_colors[u_comm]
                alpha *= 0.6
            else:
                color = "#7f8c8d"
        else:
            color = "#5d6d7e"

        x = [positions[u][0], positions[v][0]]
        y = [positions[u][1], positions[v][1]]
        ax.plot(x, y, color=color, linewidth=width, alpha=alpha, zorder=1, solid_capstyle="round")

    # Draw nodes
    for node in affinity_graph.nodes():
        x, y = positions[node][0], positions[node][1]

        if communities:
            comm_id = communities.get(node, 0)
            color = community_colors.get(comm_id, "#95a5a6")
        else:
            color = "#3498db"

        # Node size based on activity
        normalized_activity = (node_activities.get(node, 0) - min_activity) / activity_range
        size = 120 + 280 * sqrt(normalized_activity)

        ax.scatter(x, y, s=size, c=[color], edgecolors="white", linewidths=1.5, zorder=2, alpha=0.9)

    # Draw labels with adjustText
    texts = []
    for node in affinity_graph.nodes():
        x, y = positions[node][0], positions[node][1]
        text = ax.text(
            x,
            y + 0.03,
            node,
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=4,
            fontweight="medium",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
        texts.append(text)

    # Adjust text positions
    adjust_text(
        texts,
        ax=ax,
        expand=(1.2, 1.2),
        force_text=(0.3, 0.3),
        force_static=(0.15, 0.15),
        force_pull=(0.5, 0.5),
        only_move={"points": "y", "texts": "xy"},
    )

    # Create legend for communities
    if communities and unique_communities:
        legend_handles = []
        for comm_id in unique_communities:
            color = community_colors[comm_id]
            members = [n for n, c in communities.items() if c == comm_id and n in graph_nodes]
            handle = plt.scatter(
                [], [], c=[color], s=120, label=f"Community {comm_id} ({len(members)} users)"
            )
            legend_handles.append(handle)

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=11,
            framealpha=0.9,
            edgecolor="gray",
        )

    # Count connected vs isolated
    connected_count = len([n for n in affinity_graph.nodes() if affinity_graph.degree(n) > 0])
    isolated_count = affinity_graph.number_of_nodes() - connected_count

    if communities:
        title = f"Reaction Network: Mutual Affinity & Communities ({affinity_graph.number_of_nodes()} users"
    else:
        title = f"Reaction Network: Mutual Affinity ({affinity_graph.number_of_nodes()} users"
    if isolated_count > 0:
        title += f", {isolated_count} isolated"
    title += ")"

    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

    # Add period as subtitle
    if period_str:
        ax.text(
            0.5,
            0.98,
            f"Period: {period_str}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="top",
            style="italic",
            color="#555",
        )

    ax.axis("off")
    ax.margins(0.08)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close()
    return buf


def create_activity_heatmap(
    data: np.ndarray,
    username: str,
    timezone_label: str = "UTC",
) -> io.BytesIO:
    """Create an hourly activity heatmap for a user.

    Args:
        data: 7x24 numpy array (days x hours) with message counts.
              Rows: days (0=Monday through 6=Sunday).
              Columns: hours (0-23).
        username: Display name for the plot title.
        timezone_label: Timezone label for display (e.g., "UTC", "CET").

    Returns:
        BytesIO buffer containing the PNG image.
    """
    plt.figure(figsize=(14, 5))

    # Day labels (Monday first, Sunday last)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hour_labels = [f"{h:02d}" for h in range(24)]

    # Create heatmap using seaborn
    sns.heatmap(
        data,
        xticklabels=hour_labels,
        yticklabels=day_labels,
        cmap="YlOrRd",
        cbar_kws={"label": "Messages", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )

    # Styling
    plt.xlabel(f"Hour ({timezone_label})", fontsize=12)
    plt.ylabel("Day of Week", fontsize=12)
    plt.title(f"Activity Heatmap: {username}", fontsize=14, pad=15)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf
