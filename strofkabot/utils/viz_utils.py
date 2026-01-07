import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Dict, Any
import networkx as nx
import pandas as pd

def create_reaction_matrix_plot(data: pd.DataFrame, labels: List[str], fig_size: float) -> io.BytesIO:
    """Create a heatmap of reaction interactions."""
    plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, 
                     cmap="YlGnBu", square=True, cbar_kws={'shrink': .8})
    
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

def create_gdp_plot(data: List[Dict[str, Any]]) -> io.BytesIO:
    """Create a GDP (message count) plot."""
    plt.figure(figsize=(12, 6))
    data = data[::-1]  # Reverse to show oldest to newest
    months = [f"{record['year']}-{record['month']:02d}" for record in data]
    messages = [record['total_messages'] for record in data]
    
    plt.plot(months, messages, marker='o', linestyle='-', linewidth=2, markersize=8, color='#ff7f0e')
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
