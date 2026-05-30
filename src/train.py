"""
train.py
Trains XGBoost and LightGBM models on preprocessed data.
Saves the best model and cross-validation results.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

PROC_DIR   = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "data/results"
RANDOM_STATE = 42

def load_splits():
    X_train = pd.read_csv(f"{PROC_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{PROC_DIR}/y_train.csv").squeeze()
    X_test  = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_test  = pd.read_csv(f"{PROC_DIR}/y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test

def get_models():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    # Convert back from log scale for interpretable metrics
    her_pred = np.expm1(preds)
    her_true = np.expm1(y_test)
    return {
        "R2_log_scale":   round(r2_score(y_test, preds), 4),
        "R2_original":    round(r2_score(her_true, her_pred), 4),
        "MAE_log_scale":  round(mean_absolute_error(y_test, preds), 4),
        "MAE_umol_g_h":   round(mean_absolute_error(her_true, her_pred), 2),
    }

def main():
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_splits()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_score = -np.inf
    best_name  = None
    best_model = None

    for name, model in get_models().items():
        print(f"Training {name}...")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="r2", n_jobs=-1
        )

        model.fit(X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)

        results[name] = {
            "CV_R2_mean": round(cv_scores.mean(), 4),
            "CV_R2_std":  round(cv_scores.std(),  4),
            **test_metrics,
        }

        print(f"  CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test R² (log): {test_metrics['R2_log_scale']}")
        print(f"  Test R² (original HER): {test_metrics['R2_original']}")
        print(f"  Test MAE (µmol/g/h): {test_metrics['MAE_umol_g_h']}\n")

        joblib.dump(model, f"{MODELS_DIR}/{name.lower()}_model.joblib")

        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_name  = name
            best_model = model

    # Save best model separately for inference
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib")
    with open(f"{MODELS_DIR}/best_model_name.txt", "w") as f:
        f.write(best_name)

    # Save results
    with open(f"{RESULTS_DIR}/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best model: {best_name} (CV R²={best_score:.4f})")
    print(f"Saved to models/ and data/results/")
    print("\nTraining complete. Run src/evaluate.py next.")

if __name__ == "__main__":
    main()
