"""
visualization.py
================
Reusable plotting helpers used across notebooks and evaluation scripts.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Default colour palette
_PALETTE = "husl"
_STYLE   = "seaborn-v0_8-darkgrid"

try:
    plt.style.use(_STYLE)
except OSError:
    pass  # Older matplotlib – ignore missing style


# ---------------------------------------------------------------------------
# Class / label distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side bar + pie chart for a label series.

    Parameters
    ----------
    y : pd.Series
        Integer or string labels.
    title : str
        Overall figure title.
    class_names : list[str] | None
        Human-readable labels.  Uses ``y.unique()`` sorted if None.
    figsize : tuple
        Matplotlib figure size.
    save_path : str | None
        Save path if provided.
    """
    counts = y.value_counts().sort_index()

    if class_names and len(class_names) == len(counts):
        index_labels = class_names
    else:
        index_labels = [str(i) for i in counts.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Bar chart
    bars = ax1.bar(index_labels, counts.values,
                   color=sns.color_palette(_PALETTE, len(counts)))
    ax1.set_xlabel("Class", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Sample Counts")

    for bar, count in zip(bars, counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=9,
        )

    # Pie chart
    ax2.pie(
        counts.values,
        labels=index_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette(_PALETTE, len(counts)),
    )
    ax2.set_title("Proportions")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Class distribution plot saved to %s", save_path)

    plt.show()

    # Print stats
    ratio = counts.max() / counts.min()
    print(f"Imbalance ratio (majority/minority): {ratio:.2f}")


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    figsize: Tuple[int, int] = (14, 11),
    threshold: float = 0.85,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot a lower-triangle correlation heatmap and return the correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (numeric columns used automatically).
    columns : list[str] | None
        Subset of columns to include.  All numeric columns if None.
    title : str
        Axes title.
    figsize : tuple
        Matplotlib figure size.
    threshold : float
        Pairs with |r| above this value are printed to stdout.
    save_path : str | None
        Save path if provided.

    Returns
    -------
    pd.DataFrame – full correlation matrix.
    """
    cols = columns or df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=len(cols) <= 20,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"shrink": 0.7},
        square=True,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Correlation matrix saved to %s", save_path)

    plt.show()

    # Print highly correlated pairs
    high = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                high.append((corr.columns[i], corr.columns[j], r))

    if high:
        print(f"\nHighly correlated pairs (|r| >= {threshold}):")
        for a, b, r in sorted(high, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {a}  <->  {b}  : {r:.3f}")
    else:
        print(f"No pairs with |r| >= {threshold}")

    return corr


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importances",
    figsize: Tuple[int, int] = (9, 7),
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart for feature importances.

    Parameters
    ----------
    importances : pd.Series
        Index = feature names, values = importance scores.
    top_n : int
        Number of top features to display.
    """
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=figsize)
    top.plot(kind="barh", color="steelblue")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to %s", save_path)

    plt.show()


# ---------------------------------------------------------------------------
# Reconstruction error histogram (Autoencoder)
# ---------------------------------------------------------------------------


def plot_reconstruction_error(
    normal_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
    title: str = "Autoencoder Reconstruction Error",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Overlapping histogram of reconstruction errors for normal and attack traffic.

    Parameters
    ----------
    normal_errors : array
        MSE reconstruction errors for normal samples.
    attack_errors : array
        MSE reconstruction errors for attack samples.
    threshold : float
        Decision boundary line.
    """
    plt.figure(figsize=figsize)

    plt.hist(normal_errors, bins=80, alpha=0.6, color="steelblue", label="Normal",  density=True)
    plt.hist(attack_errors, bins=80, alpha=0.6, color="tomato",    label="Attack",  density=True)
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"Threshold = {threshold:.5f}")

    plt.xlabel("Reconstruction Error (MSE)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reconstruction error plot saved to %s", save_path)

    plt.show()
