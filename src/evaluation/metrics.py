"""Comprehensive Evaluation Metrics for Mule Account Detection."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Computes Precision, Recall, F1, ROC-AUC, and PR-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # In case all labels are single class
    has_both_classes = len(np.unique(y_true)) > 1

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    roc_auc = roc_auc_score(y_true, y_prob) if has_both_classes else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if has_both_classes else 0.0

    return {
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "accuracy": float(round(acc, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
    }


def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes p50, p95, and p99 inference latency percentiles in ms."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(round(np.percentile(arr, 50), 2)),
        "p95_ms": float(round(np.percentile(arr, 95), 2)),
        "p99_ms": float(round(np.percentile(arr, 99), 2)),
        "mean_ms": float(round(np.mean(arr), 2)),
    }
