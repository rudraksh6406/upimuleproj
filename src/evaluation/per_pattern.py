"""Per-Pattern Breakdown and Evaluation for Mule Detection."""
from typing import Any, Dict, List, Set
import numpy as np


def evaluate_per_pattern_recall(
    accounts: List[Dict[str, Any]],
    predicted_probs: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Computes recall for each of the 9 specific mule pattern types.
    
    Returns:
        Dict mapping pattern_name -> {total_mules, detected_mules, recall}
    """
    pattern_stats: Dict[str, Dict[str, int]] = {}

    for acc in accounts:
        pattern = acc.get("mule_pattern")
        if not pattern or not acc.get("is_mule"):
            continue
            
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {"total": 0, "detected": 0}
            
        pattern_stats[pattern]["total"] += 1
        vpa = acc["vpa"]
        prob = predicted_probs.get(vpa, 0.0)
        if prob >= threshold:
            pattern_stats[pattern]["detected"] += 1

    results = {}
    for pattern, st in pattern_stats.items():
        total = st["total"]
        det = st["detected"]
        rec = (det / total) if total > 0 else 0.0
        results[pattern] = {
            "total_mules": total,
            "detected_mules": det,
            "recall": round(rec, 4),
        }

    return results
