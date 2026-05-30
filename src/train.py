"""
train.py
Performs hyperparameter tuning with Optuna, final fits with early stopping,
Ridge baseline, LOMO-CV, and metrics saving.
"""

import os
import yaml
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Disable optuna logs unless warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths = CFG["paths"]
data_cfg = CFG["data"]
tuning_cfg = CFG["tuning"]

def main():
    os.makedirs(paths["models_dir"], exist_ok=True)
    os.makedirs(paths["results_dir"], exist_ok=True)

    # Load data
    proc_dir = paths["proc_dir"]
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()
    
    # Load sample weights
    sample_weights = pd.read_csv(
        os.path.join(proc_dir, "sample_weights_train.csv"),
        header=None
    ).squeeze()

    # Load host material for LOMO-CV
    df_clean = pd.read_csv(os.path.join(proc_dir, "df_clean.csv"), index_col=0)
    # The indexes in df_clean align to original dataframe, so we align using training indices
    # Wait, df_clean has index stored, and when preprocess.py saved it:
    # df.to_csv(..., index=True)
    # Since X_train has lost the index (it was saved with index=False), let's align them.
    # Actually, y_train.index will have the same length and align if we keep track.
    # Wait, y_train was saved with index=False too!
    # Let's check how we can align the indices.
    # Ah, in preprocess.py:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
    # The index of X_train is the original index! But when saved to CSV with index=False, the index is lost.
    # To fix this, we can load df_clean, split it using the exact same train_test_split parameters!
    # Yes! That is extremely clean and guarantees 100% correct index alignment!
    # Let's do that:
    print("Aligning df_clean train indices for LOMO-CV...")
    strat_bins = pd.qcut(df_clean["log_HER"], 10, labels=False, duplicates="drop")
    _, _, _, y_test_aligned = train_test_split(
        df_clean[[]], df_clean["log_HER"],
        test_size=data_cfg["test_size"],
        stratify=strat_bins,
        random_state=data_cfg["random_state"]
    )
    df_clean_train = df_clean.drop(y_test_aligned.index)
    host_materials_train = df_clean_train["host_material"].fillna("unknown").values
    host_materials_test = df_clean.loc[y_test_aligned.index, "host_material"].fillna("unknown").values

    # 3a. Optuna hyperparameter tuning
    print("Step 3a: Beginning Optuna hyperparameter tuning...")
    
    # ── XGBoost Tuning ──────────────────────────────────
    print("Tuning XGBoost...")
    def xgb_objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "tree_method":      "hist",
            "verbosity":        0,
            "random_state":     data_cfg["random_state"]
        }
        
        kf = KFold(n_splits=tuning_cfg["cv_folds"], shuffle=True, random_state=data_cfg["random_state"])
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights.iloc[train_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val)
            scores.append(r2_score(y_val, preds))
        return np.mean(scores)

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(
        xgb_objective,
        n_trials=tuning_cfg["n_trials"],
        timeout=tuning_cfg["timeout_seconds"]
    )
    best_xgb_params = xgb_study.best_params
    print(f"  XGBoost best CV R2: {xgb_study.best_value:.4f}")
    with open(os.path.join(paths["results_dir"], "best_params_XGBoost.json"), "w") as f:
        json.dump(best_xgb_params, f, indent=2)

    # ── LightGBM Tuning ─────────────────────────────────
    print("Tuning LightGBM...")
    def lgb_objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "verbose":           -1,
            "random_state":      data_cfg["random_state"]
        }
        
        kf = KFold(n_splits=tuning_cfg["cv_folds"], shuffle=True, random_state=data_cfg["random_state"])
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights.iloc[train_idx]
            
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val)
            scores.append(r2_score(y_val, preds))
        return np.mean(scores)

    lgb_study = optuna.create_study(direction="maximize")
    lgb_study.optimize(
        lgb_objective,
        n_trials=tuning_cfg["n_trials"],
        timeout=tuning_cfg["timeout_seconds"]
    )
    best_lgb_params = lgb_study.best_params
    print(f"  LightGBM best CV R2: {lgb_study.best_value:.4f}")
    with open(os.path.join(paths["results_dir"], "best_params_LightGBM.json"), "w") as f:
        json.dump(best_lgb_params, f, indent=2)

    # 3b. Final fit with early stopping
    print("Step 3b: Performing final model fits with early stopping validation...")
    X_tr, X_ev, y_tr, y_ev, w_tr, w_ev = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.10,
        random_state=data_cfg["random_state"]
    )
    
    # XGBoost Final Fit
    xgb_final_params = {
        **best_xgb_params,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": data_cfg["random_state"]
    }
    xgb_model = XGBRegressor(**xgb_final_params)
    xgb_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_ev, y_ev)],
        verbose=False
    )
    
    # LightGBM Final Fit
    lgb_final_params = {
        **best_lgb_params,
        "verbose": -1,
        "random_state": data_cfg["random_state"]
    }
    lgb_model = LGBMRegressor(**lgb_final_params)
    lgb_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_ev, y_ev)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # 3c. Ridge Baseline
    print("Step 3c: Training Ridge baseline model...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train, sample_weight=sample_weights)

    # 3d. Leave-One-Material-Out CV (LOMO-CV)
    print("Step 3d: Executing Leave-One-Material-Out CV (LOMO-CV)...")
    
    models = {
        "XGBoost":  XGBRegressor(**{k: v for k, v in xgb_final_params.items() if k != "early_stopping_rounds"}),
        "LightGBM": LGBMRegressor(**lgb_final_params),
        "Ridge":    Ridge(alpha=1.0)
    }
    
    lomo_results = {}
    for name, model in models.items():
        # Setup GroupKFold with 5 splits based on host_material groups
        gkf = GroupKFold(n_splits=5)
        lomo_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=host_materials_train):
            X_tr_lomo, X_val_lomo = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_lomo, y_val_lomo = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr_lomo = sample_weights.iloc[train_idx]
            
            model.fit(X_tr_lomo, y_tr_lomo, sample_weight=w_tr_lomo)
            preds = model.predict(X_val_lomo)
            lomo_scores.append(r2_score(y_val_lomo, preds))
            
        lomo_results[name] = {
            "mean": float(np.mean(lomo_scores)),
            "std":  float(np.std(lomo_scores))
        }
        print(f"  {name} LOMO-CV R2: {lomo_results[name]['mean']:.4f} ± {lomo_results[name]['std']:.4f}")

    # 3e & 3f. Compute metrics, save models, save results
    print("Step 3e/f: Evaluation and serialization...")
    
    # Fits for metrics computation (using fully trained final models)
    trained_models = {
        "XGBoost":  xgb_model,
        "LightGBM": lgb_model,
        "Ridge":    ridge_model
    }
    
    # 5-fold CV metrics for results table (standard KFold R2 mean/std)
    cv_results = {}
    for name, model in trained_models.items():
        kf = KFold(n_splits=5, shuffle=True, random_state=data_cfg["random_state"])
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr_cv = sample_weights.iloc[train_idx]
            
            # Use raw constructor parameters for fitting fold-by-fold
            if name == "XGBoost":
                fold_model = XGBRegressor(**{k: v for k, v in xgb_final_params.items() if k != "early_stopping_rounds"})
            elif name == "LightGBM":
                fold_model = LGBMRegressor(**lgb_final_params)
            else:
                fold_model = Ridge(alpha=1.0)
                
            fold_model.fit(X_tr_cv, y_tr_cv, sample_weight=w_tr_cv)
            preds = fold_model.predict(X_val_cv)
            scores.append(r2_score(y_val_cv, preds))
        cv_results[name] = {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores))
        }

    metrics_report = {}
    best_lomo_score = -np.inf
    best_model_name = None
    best_model_obj = None
    
    for name, model in trained_models.items():
        # Evaluate on test set
        preds_log = model.predict(X_test)
        preds_orig = np.expm1(preds_log)
        y_test_orig = np.expm1(y_test)
        
        # Clip negative predictions to zero for original scale HER
        preds_orig = np.clip(preds_orig, 0, None)
        
        test_r2_log = r2_score(y_test, preds_log)
        test_r2_orig = r2_score(y_test_orig, preds_orig)
        test_mae_log = mean_absolute_error(y_test, preds_log)
        test_mae_orig = mean_absolute_error(y_test_orig, preds_orig)
        test_rmse_orig = root_mean_squared_error(y_test_orig, preds_orig)
        
        metrics_report[name] = {
            "CV_R2_mean":             cv_results[name]["mean"],
            "CV_R2_std":              cv_results[name]["std"],
            "LOMO_CV_R2_mean":        lomo_results[name]["mean"],
            "LOMO_CV_R2_std":         lomo_results[name]["std"],
            "Test_R2_log":            float(test_r2_log),
            "Test_R2_original":       float(test_r2_orig),
            "Test_MAE_log":           float(test_mae_log),
            "Test_MAE_umol_g_h":      float(test_mae_orig),
            "Test_RMSE_umol_g_h":     float(test_rmse_orig)
        }
        
        # Save model
        joblib.dump(model, os.path.join(paths["models_dir"], f"{name.lower()}_model.joblib"))
        
        # Select best model based on LOMO_CV_R2_mean
        if lomo_results[name]["mean"] > best_lomo_score:
            best_lomo_score = lomo_results[name]["mean"]
            best_model_name = name
            best_model_obj = model
            
    # Save best model
    joblib.dump(best_model_obj, os.path.join(paths["models_dir"], "best_model.joblib"))
    with open(os.path.join(paths["models_dir"], "best_model_name.txt"), "w") as f:
        f.write(best_model_name)
        
    # Save metrics JSON
    with open(os.path.join(paths["results_dir"], "training_results.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    print(f"\nStep 3f: Finalized models and saved results. Best model by LOMO-CV: {best_model_name}")

if __name__ == "__main__":
    main()
