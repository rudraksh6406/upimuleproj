import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics, find_best_threshold
from src.evaluation.statistical import paired_model_comparison


def test_metrics_include_false_positive_rate_and_counts():
    metrics = compute_classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.2]))
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1


def test_validation_threshold_selection():
    threshold, metrics = find_best_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.5, 0.9]))
    assert 0 < threshold < 1
    assert metrics["f1_score"] == 1.0


def test_paired_comparison():
    result = paired_model_comparison([0.8, 0.82, 0.81], [0.7, 0.72, 0.71])
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["n_runs"] == 3
