import numpy as np
import pandas as pd

class StackingEnsemble:
    """Wrapper that stacks predictions of base models and uses a meta-learner (e.g. Ridge)."""
    def __init__(self, base_models: dict, meta_learner, active_features: list):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.active_features = active_features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_base = X[self.active_features].copy()
        base_preds = []
        for name in sorted(self.base_models.keys()):
            base_preds.append(self.base_models[name].predict(X_base))
        X_meta = np.column_stack(base_preds)
        return self.meta_learner.predict(X_meta)
