"""
metrics.py
==========
Evaluation utilities used across all notebooks:
  - evaluate_model()         : returns a dict of classification metrics
  - plot_roc_curve()         : ROC-AUC plot for one or many models
  - plot_confusion_matrix()  : annotated heatmap
  - plot_training_history()  : Keras history loss / accuracy curves
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like | None
        Predicted probabilities for the positive class (enables AUC metrics).
    verbose : bool
        Print the sklearn classification report when True.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
                    confusion_matrix, classification_report
    """
    metrics: Dict = {}

    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")

    metrics["confusion_matrix"]      = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
    )

    if verbose:
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1']:.4f}")
        if y_prob is not None:
            print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
            print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
        print()
        print(metrics["classification_report"])

    logger.info(
        "evaluate_model  acc=%.4f  f1=%.4f  roc_auc=%.4f",
        metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------


def plot_roc_curve(
    models_probs: Dict[str, tuple],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for multiple models on a single axes.

    Parameters
    ----------
    models_probs : dict
        ``{ 'Model Name': (y_true, y_prob), ... }``
    figsize : tuple
        Matplotlib figure size.
    save_path : str | None
        If provided, saves the figure to this path.

    Example
    -------
    >>> plot_roc_curve({
    ...     'XGBoost': (y_test, xgb_probs),
    ...     'Random Forest': (y_test, rf_probs),
    ... })
    """
    plt.figure(figsize=figsize)

    for name, (y_true, y_prob) in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score   = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves – Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)

    plt.show()


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (6, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot an annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground truth and predicted labels.
    class_names : list[str] | None
        Labels for axes.  Defaults to [0, 1].
    title : str
        Axes title.
    figsize : tuple
        Matplotlib figure size.
    save_path : str | None
        Save figure to this path if provided.
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = class_names or ["Normal", "Attack"]

    # Normalised version for colour scale
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("Actual", fontsize=11)
    plt.xlabel("Predicted", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    plt.show()

    # Print raw numbers
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None,) * 4
    if tn is not None:
        print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"False Positive Rate : {fp/(fp+tn):.4f}")
        print(f"False Negative Rate : {fn/(fn+tp):.4f}")


# ---------------------------------------------------------------------------
# Keras training history plot
# ---------------------------------------------------------------------------


def plot_training_history(
    history,
    title: str = "Training History",
    figsize: tuple = (13, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss and accuracy curves from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
        Returned by ``model.fit(...)``.
    title : str
        Overall figure title.
    figsize : tuple
        Matplotlib figure size.
    save_path : str | None
        Save figure to this path if provided.
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    has_accuracy = "accuracy" in hist
    n_plots = 2 if has_accuracy else 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n_plots == 1:
        axes = [axes]

    # Loss
    axes[0].plot(epochs, hist["loss"],     label="Train Loss")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    if has_accuracy:
        axes[1].plot(epochs, hist["accuracy"],     label="Train Acc")
        axes[1].plot(epochs, hist["val_accuracy"], label="Val Acc", linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training history saved to %s", save_path)

    plt.show()
