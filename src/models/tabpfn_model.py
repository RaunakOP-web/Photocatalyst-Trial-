import os
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TabPFNModelPlaceholder:
    def __init__(self):
        logger.info("Initializing TabPFN fallback placeholder...")

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        # Default placeholder returns zero prediction
        import numpy as np
        return np.zeros(len(X))
