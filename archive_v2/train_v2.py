"""
train_v2.py

Upgraded pipeline with CatBoost, ExtraTrees, multivariate TPE Optuna tuning,
weighted-average ensemble, and expanded parameter spaces.

Fixes applied (2026-06-08):
  Fix 1 – Drop confidence_volume, confidence_pH, confidence_bandgap
  Fix 2 – LightGBM DART → GBDT + min_child_samples / path_smooth regularisation
  Fix 3 – Replace ElasticNet stacking with simple weighted average (XGB 0.35, LGB 0.30, CAT 0.35)
  Fix 4 – n_trials=150, timeout=1800, expanded XGB/LGB/CatBoost search spaces
  Fix 5 – Drop raw duplicates: bandgap_eV, catalyst_loading_mg, catalyst_loading_g_L,
           reaction_volume_mL, glycerol_concentration_v_pct, light_power_W,
           co_catalyst_wt_pct, glycerol_concentration_std
"""

import os
import yaml
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

paths      = CFG["paths"]
data_cfg   = CFG["data"]
tuning_cfg = CFG["tuning"]
SEED       = data_cfg["random_state"]
N_TRIALS   = tuning_cfg["n_trials"]
TIMEOUT    = tuning_cfg["timeout_seconds"]
CV_FOLDS   = tuning_cfg["cv_folds"]

# ── Fix 1 & 5: columns to drop before any model fitting ───────────────────────
COLS_TO_DROP = [
    # Fix 1: confidence audit flags (not physical predictors)
    "confidence_volume", "confidence_pH", "confidence_bandgap",
    # Fix 5: raw duplicates superseded by log-transform or lookup-table versions
    "bandgap_eV",                   # superseded by semi_bandgap_eV
    "catalyst_loading_mg",          # superseded by log_loading_mg
    "catalyst_loading_g_L",         # superseded by log_loading_gL
    "reaction_volume_mL",           # superseded by log_reaction_vol
    "glycerol_concentration_v_pct", # superseded by log_glycerol_conc
    "light_power_W",                # superseded by log_light_power
    "co_catalyst_wt_pct",           # superseded by log_cocat_loading
    "glycerol_concentration_std",   # data-quality indicator, not a predictor
]

# Weighted average ensemble weights (Fix 3)
ENSEMBLE_WEIGHTS = {"XGBoost": 0.35, "LightGBM": 0.30, "CatBoost": 0.35}


def main():
    os.makedirs(paths["models_dir"],  exist_ok=True)
    os.makedirs(paths["results_dir"], exist_ok=True)

    proc_dir = paths["proc_dir"]

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading processed data...")
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()

    sample_weights = pd.read_csv(
        os.path.join(proc_dir, "sample_weights_train.csv"), header=None
    ).squeeze()

    group_labels_train = pd.read_csv(
        os.path.join(proc_dir, "group_labels_train.csv")
    ).squeeze().values

    # ── Apply Fix 1 & 5: drop redundant / audit columns ──────────────────────
    drop_actual = [c for c in COLS_TO_DROP if c in X_train.columns]
    X_train = X_train.drop(columns=drop_actual)
    X_test  = X_test.drop(columns=drop_actual)
    print(f"  Dropped {len(drop_actual)} redundant/audit columns: {drop_actual}")

    # ── Verification print ────────────────────────────────────────────────────
    print(f"\n=== VERIFICATION ===")
    print(f"  X_train shape after drops: {X_train.shape}")
    assert X_train.shape[1] >= 60 and X_train.shape[1] <= 67, \
        f"Expected 60–67 columns, got {X_train.shape[1]}"
    forbidden = [c for c in COLS_TO_DROP if c in X_train.columns]
    assert len(forbidden) == 0, f"These should have been dropped: {forbidden}"
    print(f"  No forbidden columns present. OK")
    print(f"  Final columns ({X_train.shape[1]}):")
    print(f"  {X_train.columns.tolist()}")
    print(f"====================\n")

    cat_cols = joblib.load(os.path.join(proc_dir, "cat_cols.joblib"))
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    print(f"  X_train: {X_train.shape} | cat_cols: {len(cat_cols)}")

    # ── 3a. Global TargetEncoder (cv=5 prevents leakage internally) ───────────
    print("\nStep 3a: Encoding categoricals (TargetEncoder, cv=5)...")
    if cat_cols:
        encoder = TargetEncoder(random_state=SEED, cv=5)
        X_train_enc = X_train.copy()
        X_test_enc  = X_test.copy()
        X_train_enc[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        X_test_enc[cat_cols]  = encoder.transform(X_test[cat_cols])
        joblib.dump(encoder, os.path.join(proc_dir, "target_encoder.joblib"))
        print(f"  Encoded {len(cat_cols)} categorical columns.")
    else:
        X_train_enc = X_train.copy()
        X_test_enc  = X_test.copy()
        encoder = None

    # ── 3b. Optuna HPO ────────────────────────────────────────────────────────
    print("\nStep 3b: Bayesian hyperparameter optimisation (multivariate TPE)...")
    best_params_all = {}

    # Fix 4: use fresh TPE sampler per study to avoid dynamic search space issues
    def make_sampler():
        return optuna.samplers.TPESampler(multivariate=True, seed=SEED)

    def _cv_r2(model_factory, X, y, groups, sample_weights):
        """LOGO-CV with internal target encoding, evaluated globally over OOF predictions."""
        logo = LeaveOneGroupOut()
        oof_preds = np.zeros(len(y))
        valid_mask = np.zeros(len(y), dtype=bool)

        for tr_idx, val_idx in logo.split(X, y, groups=groups):
            if len(val_idx) < 2:
                continue
            X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr = y.iloc[tr_idx]
            sw_tr = None if sample_weights is None else sample_weights.iloc[tr_idx]

            if cat_cols:
                te = TargetEncoder(cv=5, random_state=SEED)
                X_tr[cat_cols] = te.fit_transform(X_tr[cat_cols], y_tr)
                X_val[cat_cols] = te.transform(X_val[cat_cols])

            m = model_factory()
            try:
                m.fit(X_tr, y_tr, sample_weight=sw_tr)
            except TypeError:
                m.fit(X_tr, y_tr)

            oof_preds[val_idx] = m.predict(X_val)
            valid_mask[val_idx] = True

        if not valid_mask.any():
            return -999.0
        return r2_score(y[valid_mask], oof_preds[valid_mask])

    # ── XGBoost (Fix 4: expanded search ranges) ───────────────────────────────
    print("  Tuning XGBoost...")
    def xgb_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "max_delta_step":    trial.suggest_int("max_delta_step", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "objective":  "reg:squarederror",
            "tree_method": "hist", "verbosity": 0, "random_state": SEED,
        }
        return _cv_r2(lambda: XGBRegressor(**p), X_train_enc, y_train,
                      group_labels_train, sample_weights)

    s = optuna.create_study(direction="maximize", sampler=make_sampler())
    s.optimize(xgb_obj, n_trials=N_TRIALS, timeout=TIMEOUT)
    best_params_all["XGBoost"] = s.best_params
    print(f"    Best CV R²: {s.best_value:.4f}")

    # ── LightGBM (Fix 2: DART → GBDT + min_child_samples / path_smooth) ──────
    print("  Tuning LightGBM...")
    def lgb_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),   # Fix 2
            "path_smooth":       trial.suggest_float("path_smooth", 0.0, 1.0),     # Fix 2
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 2.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            "boosting_type": "gbdt",   # Fix 2: reverted from dart
            "verbose": -1, "random_state": SEED,
        }
        return _cv_r2(lambda: LGBMRegressor(**p), X_train_enc, y_train,
                      group_labels_train, sample_weights)

    s = optuna.create_study(direction="maximize", sampler=make_sampler())
    s.optimize(lgb_obj, n_trials=N_TRIALS, timeout=TIMEOUT)
    best_params_all["LightGBM"] = s.best_params
    print(f"    Best CV R²: {s.best_value:.4f}")

    # ── CatBoost (Fix 4: add grow_policy) ────────────────────────────────────
    print("  Tuning CatBoost...")
    def cat_obj(trial):
        p = {
            "iterations":          trial.suggest_int("iterations", 300, 2000),
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 0.001, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 3.0),
            "border_count":        trial.suggest_int("border_count", 32, 254),
            "grow_policy":         trial.suggest_categorical("grow_policy",
                                       ["SymmetricTree", "Lossguide"]),  # Fix 4
            "verbose": 0, "random_state": SEED, "allow_writing_files": False,
        }
        return _cv_r2(lambda: CatBoostRegressor(**p), X_train_enc, y_train,
                      group_labels_train, sample_weights)

    s = optuna.create_study(direction="maximize", sampler=make_sampler())
    s.optimize(cat_obj, n_trials=N_TRIALS, timeout=TIMEOUT)
    best_params_all["CatBoost"] = s.best_params
    print(f"    Best CV R²: {s.best_value:.4f}")

    print("  Adding ExtraTrees (fixed hyperparameters)...")

    # Save params
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        with open(os.path.join(paths["results_dir"], f"best_params_{name}_v2.json"), "w") as f:
            json.dump(best_params_all[name], f, indent=2)

    # ── 3c. Build final model instances ───────────────────────────────────────
    print("\nStep 3c: Building final base models...")
    xgb_p = {**best_params_all["XGBoost"],
              "objective": "reg:squarederror", "tree_method": "hist",
              "verbosity": 0, "random_state": SEED}
    lgb_p = {**best_params_all["LightGBM"],
              "boosting_type": "gbdt", "verbose": -1, "random_state": SEED}  # Fix 2
    cat_p = {**best_params_all["CatBoost"],
              "verbose": 0, "random_state": SEED, "allow_writing_files": False}

    model_factories = {
        "XGBoost":    lambda: XGBRegressor(**xgb_p),
        "LightGBM":   lambda: LGBMRegressor(**lgb_p),
        "CatBoost":   lambda: CatBoostRegressor(**cat_p),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=800, max_features=0.6,
            min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }

    # ── 3d. LOGO-CV for base models ───────────────────────────────────────────
    print("\nStep 3d: LOGO-CV evaluation of base learners...")
    logo = LeaveOneGroupOut()
    logo_results = {}

    for name, factory in model_factories.items():
        oof_preds  = np.zeros(len(y_train))
        valid_mask = np.zeros(len(y_train), dtype=bool)

        for tr_idx, val_idx in logo.split(X_train_enc, y_train, groups=group_labels_train):
            if len(val_idx) < 2:
                continue
            X_tr, X_val = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
            y_tr         = y_train.iloc[tr_idx]
            w_tr         = sample_weights.iloc[tr_idx]
            m = factory()
            try:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            except TypeError:
                m.fit(X_tr, y_tr)
            oof_preds[val_idx]  = m.predict(X_val)
            valid_mask[val_idx] = True

        mean_r2 = float(r2_score(y_train[valid_mask], oof_preds[valid_mask])) \
                  if valid_mask.any() else -999.0
        logo_results[name] = {"mean": mean_r2, "std": 0.0}
        print(f"  {name:12s}  LOGO-CV R² = {mean_r2:.4f}")

    # ── 3e. Train final base models on full training set ──────────────────────
    print("\nStep 3e: Training base models on full training set...")
    trained_models = {}
    for name, factory in model_factories.items():
        m = factory()
        try:
            m.fit(X_train_enc, y_train, sample_weight=sample_weights)
        except TypeError:
            m.fit(X_train_enc, y_train)
        trained_models[name] = m

    # ── 3f. Weighted-average ensemble (Fix 3) ─────────────────────────────────
    print("\nStep 3f: Building weighted-average ensemble (XGB 0.35, LGB 0.30, CAT 0.35)...")

    def ensemble_predict(models, weights, X):
        """Return weighted-average prediction in log space."""
        pred = np.zeros(len(X))
        for name, w in weights.items():
            pred += w * models[name].predict(X)
        return pred

    # LOGO-CV for ensemble
    ens_oof_preds  = np.zeros(len(y_train))
    ens_valid_mask = np.zeros(len(y_train), dtype=bool)

    for tr_idx, val_idx in logo.split(X_train_enc, y_train, groups=group_labels_train):
        if len(val_idx) < 2:
            continue
        X_tr, X_val = X_train_enc.iloc[tr_idx], X_train_enc.iloc[val_idx]
        y_tr         = y_train.iloc[tr_idx]
        w_tr         = sample_weights.iloc[tr_idx]

        fold_models = {}
        for name, factory in {k: v for k, v in model_factories.items()
                               if k in ENSEMBLE_WEIGHTS}.items():
            m = factory()
            try:
                m.fit(X_tr, y_tr, sample_weight=w_tr)
            except TypeError:
                m.fit(X_tr, y_tr)
            fold_models[name] = m

        ens_oof_preds[val_idx] = ensemble_predict(fold_models, ENSEMBLE_WEIGHTS, X_val)
        ens_valid_mask[val_idx] = True

    ens_mean = float(r2_score(y_train[ens_valid_mask], ens_oof_preds[ens_valid_mask])) \
               if ens_valid_mask.any() else -999.0
    logo_results["Ensemble"] = {"mean": ens_mean, "std": 0.0}
    print(f"  {'Ensemble':12s}  LOGO-CV R² = {ens_mean:.4f}")

    # Store ensemble "model" as a callable wrapper
    class WeightedEnsemble:
        def __init__(self, models, weights):
            self.models  = models
            self.weights = weights
        def predict(self, X):
            return ensemble_predict(self.models, self.weights, X)

    ens_models = {k: trained_models[k] for k in ENSEMBLE_WEIGHTS}
    trained_models["Ensemble"] = WeightedEnsemble(ens_models, ENSEMBLE_WEIGHTS)

    # ── 3g. Test-set metrics ──────────────────────────────────────────────────
    print("\nStep 3g: Test-set metrics...")
    metrics_report = {}
    for name, model in trained_models.items():
        preds_log   = model.predict(X_test_enc)
        preds_orig  = np.clip(np.expm1(preds_log), 0, None)
        y_test_orig = np.expm1(y_test)

        logo_mean = logo_results.get(name, {}).get("mean", 0.0)
        logo_std  = logo_results.get(name, {}).get("std",  0.0)

        metrics_report[name] = {
            "LOGO_CV_R2_mean":    round(logo_mean, 4),
            "LOGO_CV_R2_std":     round(logo_std,  4),
            "Test_R2_log":        round(float(r2_score(y_test, preds_log)), 4),
            "Test_R2_original":   round(float(r2_score(y_test_orig, preds_orig)), 4),
            "Test_MAE_log":       round(float(mean_absolute_error(y_test, preds_log)), 4),
            "Test_MAE_umol_g_h":  round(float(mean_absolute_error(y_test_orig, preds_orig)), 1),
            "Test_RMSE_umol_g_h": round(float(root_mean_squared_error(y_test_orig, preds_orig)), 1),
            "composite_score":    round(logo_mean * 0.5 + float(r2_score(y_test, preds_log)) * 0.5, 4),
        }
        print(f"  {name:12s}  LOGO={logo_mean:.4f}±{logo_std:.4f}  "
              f"TestR²(log)={metrics_report[name]['Test_R2_log']:.4f}  "
              f"MAE_orig={metrics_report[name]['Test_MAE_umol_g_h']:,.0f}")

    best_model_name = max(metrics_report, key=lambda n: metrics_report[n]["Test_R2_log"])
    print(f"\nBest model: {best_model_name} "
          f"(Test R² = {metrics_report[best_model_name]['Test_R2_log']:.4f})")

    # Save all individual base models for evaluate.py to reload
    for name in ["XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]:
        joblib.dump(trained_models[name],
                    os.path.join(paths["models_dir"], f"{name}.joblib"))

    # Save ensemble weights so evaluate.py can reconstruct
    with open(os.path.join(paths["models_dir"], "ensemble_weights.json"), "w") as f:
        json.dump(ENSEMBLE_WEIGHTS, f, indent=2)

    # best_model.joblib: use Ensemble's best constituent if Ensemble wins
    best_for_pickle = best_model_name if best_model_name != "Ensemble" else "XGBoost"
    joblib.dump(trained_models[best_for_pickle],
                os.path.join(paths["models_dir"], "best_model.joblib"))
    with open(os.path.join(paths["models_dir"], "best_model_name.txt"), "w") as f:
        f.write(best_model_name)

    with open(os.path.join(paths["results_dir"], "training_results_v2.json"), "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nTraining complete. Results in {paths['results_dir']}/training_results_v2.json")


if __name__ == "__main__":
    main()
