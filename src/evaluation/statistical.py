"""Statistical tests and confidence intervals for repeated experiments."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy import stats


def paired_model_comparison(candidate: Iterable[float], baseline: Iterable[float]) -> Dict[str, float]:
    """Compare paired metric observations using a t-test and Cohen's dz.

    The same seeds/splits must be used for each pair. At least two runs are
    required; constant zero differences are reported without producing NaNs.
    """
    a = np.asarray(list(candidate), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("candidate and baseline must be equal-length 1-D arrays with >=2 runs")
    delta = a - b
    std = delta.std(ddof=1)
    if np.allclose(delta, 0):
        t_stat, p_value, effect = 0.0, 1.0, 0.0
    elif std < np.finfo(float).eps:
        # Every paired run has the same non-zero improvement.
        t_stat, p_value, effect = float(np.sign(delta.mean()) * np.inf), 0.0, float(np.sign(delta.mean()) * np.inf)
    else:
        t_stat, p_value = stats.ttest_rel(a, b)
        effect = float(delta.mean() / std)
    sem = stats.sem(delta)
    margin = float(stats.t.ppf(0.975, len(delta) - 1) * sem) if sem > 0 else 0.0
    return {
        "candidate_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "mean_difference": float(delta.mean()),
        "ci95_low": float(delta.mean() - margin),
        "ci95_high": float(delta.mean() + margin),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": effect,
        "n_runs": int(len(a)),
    }
