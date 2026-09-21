"""Gradient Boosting Baseline for UPI Mule Detection with XGBoost / Sklearn fallback."""
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier


class XGBoostBaseline:
    """Gradient Boosting baseline trained on graph-aggregated tabular node features."""

    def __init__(self, max_depth: int = 5, n_estimators: int = 150, learning_rate: float = 0.05):
        self.scaler = StandardScaler()
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                scale_pos_weight=10.0,  # handle class imbalance
                random_state=42,
                eval_metric="logloss",
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_depth=max_depth,
                max_iter=n_estimators,
                learning_rate=learning_rate,
                class_weight="balanced",
                random_state=42
            )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
