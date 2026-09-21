"""MuleGuard evaluation and benchmark suite."""
from .metrics import compute_classification_metrics, compute_latency_percentiles
from .per_pattern import evaluate_per_pattern_recall
from .visualize import (
    plot_ablation_barchart,
    plot_confusion_matrices,
    plot_per_pattern_barchart,
    plot_pr_curves,
    plot_roc_curves,
)

__all__ = [
    "compute_classification_metrics",
    "compute_latency_percentiles",
    "evaluate_per_pattern_recall",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_matrices",
    "plot_per_pattern_barchart",
    "plot_ablation_barchart",
]
