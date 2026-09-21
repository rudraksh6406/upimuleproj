"""Logistic Regression Baseline for UPI Mule Detection."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionBaseline:
    """Standard Logistic Regression baseline with class balancing."""

    def __init__(self, c_param: float = 1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=c_param,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
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
