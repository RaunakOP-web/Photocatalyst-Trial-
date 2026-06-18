import os
import json
import joblib
import pandas as pd

def safe_load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}")
    return pd.read_csv(path)

def safe_save_csv(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)

def safe_load_joblib(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def safe_save_joblib(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)

def safe_load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
