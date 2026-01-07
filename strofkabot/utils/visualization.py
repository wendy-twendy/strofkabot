"""Visualization and plotting functions."""

import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


def create_monthly_inflation_plot(monthly_inflation: list[dict]) -> io.BytesIO:
    """Create a bar plot showing monthly reaction inflation."""
    plt.figure(figsize=(10, 6))
    months = [record['month_year'] for record in monthly_inflation]
    changes = [record['change_percentage'] for record in monthly_inflation]
    averages = [record['average_rpm'] for record in monthly_inflation]

    ax = sns.barplot(x=months, y=changes, hue=months, palette="viridis", legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.title("Monthly Reaction Inflation (Last 12 Months)")
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(averages):
        ax.text(i, changes[i], f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def create_yearly_inflation_plot(yearly_inflation: list[dict]) -> io.BytesIO:
    """Create a bar plot showing yearly reaction inflation."""
    plt.figure(figsize=(8, 6))
    years = [record['year'] for record in yearly_inflation]
    changes_yearly = [record['change_percentage'] for record in yearly_inflation]
    averages_yearly = [record['average_rpm'] for record in yearly_inflation]

    ax = sns.barplot(x=years, y=changes_yearly, hue=years, palette="deep", legend=False)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=0)
    plt.xlabel("Year")
    plt.ylabel("YoY Inflation (%)")
    plt.title("Yearly Reaction Inflation")
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(averages_yearly):
        ax.text(i, changes_yearly[i], f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def determine_figure_size(num_users: int) -> float:
    """Calculate appropriate figure size based on number of users."""
    return max(24, num_users * 0.6)


def generate_reaction_matrix_plot(
    data: np.ndarray,
    labels: list[str],
    fig_size: float
) -> io.BytesIO:
    """Generate a heatmap of reaction interactions between users."""
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        data, xticklabels=labels, yticklabels=labels,
        cmap="YlGnBu", square=True, cbar_kws={'shrink': .8}
    )

    plt.xticks(rotation=90, ha='center', fontsize=16)
    plt.yticks(rotation=0, va='center', fontsize=16)

    plt.title("Reaction Matrix (Percentage of Reactions Given)", fontsize=22, pad=20)
    plt.xlabel("Reactions Received From", fontsize=18, labelpad=15)
    plt.ylabel("Reactions Given To", fontsize=18, labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


def generate_cluster_plot(
    data: np.ndarray,
    labels: np.ndarray,
    member_names: list[str]
) -> io.BytesIO:
    """Generate a scatter plot of user clusters based on reactions."""
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(sns.color_palette("hsv", np.unique(labels).size).as_hex())
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=cmap, legend='full')
    plt.xlabel("Reactions Given")
    plt.ylabel("Reactions Received")
    plt.title("User Clusters Based on Reactions")
    plt.legend(title='Cluster')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def create_gdp_plot(data: list[dict]) -> io.BytesIO:
    """Create a line plot of server GDP (messages per month)."""
    plt.figure(figsize=(12, 6))
    data = data[::-1]
    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    messages = [record['total_messages'] for record in data]

    plt.plot(months, messages, marker='o', linestyle='-', linewidth=2,
             markersize=8, color='#ff7f0e')
    plt.fill_between(months, messages, alpha=0.2, color='#ff7f0e')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Month")
    plt.ylabel("Total Messages")
    plt.title("Server GDP (Total Messages per Month)")

    for i, v in enumerate(messages):
        plt.text(i, v + (max(messages) * 0.02), str(v),
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf


def create_hdi_plot(data: list[dict]) -> io.BytesIO:
    """Create a line plot of server HDI (quality messages ratio)."""
    plt.figure(figsize=(12, 6))
    data = data[::-1]
    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    hdi_values = [record['hdi_ratio'] for record in data]

    plt.plot(months, hdi_values, marker='o', linestyle='-', linewidth=2,
             markersize=8, color='#2ecc71')
    plt.fill_between(months, hdi_values, alpha=0.2, color='#2ecc71')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Month")
    plt.ylabel("HDI Ratio (Quality/Total Messages)")
    plt.title("Server HDI (Quality Messages Ratio per Month)")

    for i, v in enumerate(hdi_values):
        plt.text(i, v + (max(hdi_values) * 0.02), f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()
    return buf
