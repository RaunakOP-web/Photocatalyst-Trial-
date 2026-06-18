import os
import yaml

CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))

def load_yaml(filename):
    path = os.path.join(CONFIGS_DIR, filename)
    if not os.path.exists(path):
        # Fallback to root if called from elsewhere or when run locally
        path = os.path.join("configs", filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_train_config():
    return load_yaml("train.yaml")

def get_features_config():
    return load_yaml("features.yaml")

def get_models_config():
    return load_yaml("models.yaml")
