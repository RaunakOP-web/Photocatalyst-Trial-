import pytest
import numpy as np
import pandas as pd
from src.ensemble.blending import BlendingEnsemble
from src.evaluation.metrics import calculate_metrics, calculate_original_scale_metrics

class MockModel:
    def predict(self, X):
        return np.ones(len(X)) * 2.0

def test_blending_ensemble():
    models = {
        "model_a": MockModel(),
        "model_b": MockModel()
    }
    weights = {
        "model_a": 0.3,
        "model_b": 0.7
    }
    active_features = ["feat1", "feat2"]
    
    ensemble = BlendingEnsemble(models, weights, active_features)
    X = pd.DataFrame({
        "feat1": [1, 2],
        "feat2": [3, 4],
        "other": [5, 6]
    })
    
    preds = ensemble.predict(X)
    assert len(preds) == 2
    # Weighted average of mock models returning 2.0: 0.3*2.0 + 0.7*2.0 = 2.0
    assert np.allclose(preds, 2.0)

def test_metrics_calculation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert "R2" in metrics
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert metrics["R2"] > 0.9

def test_original_scale_metrics():
    # log1p scale targets
    y_true_log = np.log1p(np.array([10.0, 20.0, 30.0]))
    y_pred_log = np.log1p(np.array([11.0, 19.0, 31.0]))
    
    metrics = calculate_original_scale_metrics(y_true_log, y_pred_log)
    assert "R2_orig" in metrics
    assert "MAE_orig" in metrics
    assert metrics["R2_orig"] > 0.9
