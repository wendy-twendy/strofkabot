"""Utility functions for the Strofkabot Discord bot."""

from strofkabot.utils.data_processing import (
    calculate_monthly_inflation,
    calculate_reaction_percentage,
    calculate_yearly_inflation,
    fetch_gdp_data,
    fetch_hdi_data,
    fetch_inflation_data,
    get_reaction_trade_data,
    perform_kmeans_clustering,
    prepare_clustering_data,
)
from strofkabot.utils.date_utils import adjust_month
from strofkabot.utils.discord_helpers import (
    get_member_names,
    get_non_bot_member_ids,
    get_reply_info,
    parse_rpm_args,
    send_leaderboard,
    send_most_liked_stats,
    send_personal_stats,
)
from strofkabot.utils.visualization import (
    create_gdp_plot,
    create_hdi_plot,
    create_monthly_inflation_plot,
    create_yearly_inflation_plot,
    determine_figure_size,
    generate_cluster_plot,
    generate_reaction_matrix_plot,
)

__all__ = [
    # Date utilities
    'adjust_month',
    # Visualization
    'create_monthly_inflation_plot',
    'create_yearly_inflation_plot',
    'determine_figure_size',
    'generate_reaction_matrix_plot',
    'generate_cluster_plot',
    'create_gdp_plot',
    'create_hdi_plot',
    # Data processing
    'fetch_inflation_data',
    'calculate_monthly_inflation',
    'calculate_yearly_inflation',
    'fetch_gdp_data',
    'fetch_hdi_data',
    'get_reaction_trade_data',
    'calculate_reaction_percentage',
    'prepare_clustering_data',
    'perform_kmeans_clustering',
    # Discord helpers
    'parse_rpm_args',
    'send_leaderboard',
    'send_personal_stats',
    'get_reply_info',
    'get_member_names',
    'get_non_bot_member_ids',
    'send_most_liked_stats',
]
