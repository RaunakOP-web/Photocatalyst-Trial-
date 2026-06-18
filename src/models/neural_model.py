from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_neural_model():
    """Returns a scikit-learn Pipeline containing a StandardScaler and MLPRegressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=0.001, max_iter=500, random_state=42))
    ])
