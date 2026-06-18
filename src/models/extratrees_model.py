import optuna
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from src.data.validation import preprocess_and_engineer_folds

def tune_extratrees(X_train, y_train, sample_weights, groups, cat_cols, cv_folds, n_trials=50):
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15, seed=42)
    
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 1000),
            "max_features":      trial.suggest_float("max_features", 0.2, 0.6),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 2, 15),
            "max_depth":         trial.suggest_int("max_depth", 10, 30),
            "bootstrap":         True,
            "random_state":      42,
            "n_jobs":            1
        }
        
        gkf = GroupKFold(n_splits=cv_folds)
        scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = sample_weights.iloc[train_idx]
            
            # Preprocess and engineer fold features leakage-free
            X_tr_proc, X_val_proc = preprocess_and_engineer_folds(X_tr, X_val, y_tr, cat_cols)
            
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr_proc, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val_proc)
            scores.append(r2_score(y_val, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

