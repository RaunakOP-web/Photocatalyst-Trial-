import numpy as np
import pandas as pd

class BlendingEnsemble:
    """Wrapper that blends predictions from a set of base models using given weights."""
    def __init__(self, models: dict, weights: dict, active_features: list):
        self.models = models
        self.weights = weights
        self.active_features = active_features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_base = X[self.active_features].copy()
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w > 0.0:
                preds += w * model.predict(X_base)
        return preds
