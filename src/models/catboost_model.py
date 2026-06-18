import optuna
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from src.data.validation import preprocess_and_engineer_folds

def tune_catboost(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=50):
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42)
    
    def objective(trial):
        params = {
            "iterations":         trial.suggest_int("iterations", 500, 2000),
            "depth":              trial.suggest_int("depth", 4, 9),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "verbose":            0,
            "random_seed":        42,
            "allow_writing_files":False
        }
        
        gkf = GroupKFold(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights.iloc[train_idx]
            
            # Preprocess and engineer fold features leakage-free
            X_tr_proc, X_val_proc = preprocess_and_engineer_folds(X_tr, X_val, y_tr, cat_cols)
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr_proc, y_tr, sample_weight=w_tr, verbose=0)
            preds = model.predict(X_val_proc)
            scores.append(r2_score(y_val, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

