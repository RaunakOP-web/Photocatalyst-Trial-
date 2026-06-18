import numpy as np
import pandas as pd

class RoutingEnsemble:
    """Combines a specialist and generalist model using host material routing."""
    def __init__(self, specialist, generalist, route_group="tio2"):
        self.specialist = specialist
        self.generalist = generalist
        self.route_group = route_group

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        if "host_material" in X.columns:
            mask = X["host_material"].astype(str).str.strip().str.lower() == self.route_group
        else:
            mask = np.zeros(len(X), dtype=bool)

        if mask.any():
            preds[mask] = self.specialist.predict(X[mask])
        if (~mask).any():
            preds[~mask] = self.generalist.predict(X[~mask])
        return preds
