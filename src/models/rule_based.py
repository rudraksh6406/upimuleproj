"""Rule-Based Fraud Detection Heuristics Baseline."""
from typing import Dict, List, Tuple
import numpy as np


class RuleBasedBaseline:
    """Heuristic rule-based detection reflecting current legacy banking and payment gateway rules."""

    def __init__(
        self,
        velocity_threshold_1h: int = 5,
        turnover_threshold: float = 0.85,
        dormancy_threshold_days: float = 45.0,
        fan_out_threshold: int = 4,
    ):
        self.velocity_thresh = velocity_threshold_1h
        self.turnover_thresh = turnover_threshold
        self.dormancy_thresh = dormancy_threshold_days
        self.fan_out_thresh = fan_out_threshold

    def evaluate_rules(self, x_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates rules on feature matrix.
        Features expected indices:
        - index 10: device_count
        - index 11: dormancy_days
        - index 16: turnover_ratio
        - index 17: max_velocity_1h
        - index 9: out_cps_count
        """
        n_samples = x_features.shape[0]
        scores = np.zeros(n_samples, dtype=np.float32)
        predictions = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            f = x_features[i]
            out_cps = f[9]
            dev_count = f[10]
            dormancy = f[11]
            turnover = f[16]
            vel_1h = f[17]

            rules_triggered = 0
            
            # Rule 1: High Velocity Burst
            if vel_1h >= self.velocity_thresh:
                rules_triggered += 1
                
            # Rule 2: High Pass-Through Turnover (Layering / Conduit)
            if turnover >= self.turnover_thresh:
                rules_triggered += 1
                
            # Rule 3: Dormancy reactivated
            if dormancy >= self.dormancy_thresh and vel_1h >= 2:
                rules_triggered += 1
                
            # Rule 4: Rapid Fan-out
            if out_cps >= self.fan_out_thresh and vel_1h >= 3:
                rules_triggered += 1
                
            # Rule 5: Multi-Device Fraud ring
            if dev_count >= 3:
                rules_triggered += 1

            # Score normalized by 5 rules
            scores[i] = min(1.0, rules_triggered / 3.0)
            predictions[i] = 1 if rules_triggered >= 2 else 0

        return scores, predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores, _ = self.evaluate_rules(X)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, preds = self.evaluate_rules(X)
        return preds
