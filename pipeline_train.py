import patches
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from automatminer import MatPipe, AutoFeaturizer, DataCleaner, FeatureReducer
from automatminer.automl.adaptors import TPOTAdaptor
from tpot.config import regressor_config_dict

# Load cleaned dataset
csv_path = "catalyst_dataset_clean.csv"
df = pd.read_csv(csv_path)

# Filter columns to only include features and target
df_train = df[["composition", "Bandgap_eV", "VB_eV_vs_NHE", "CB_eV_vs_NHE", "BET_m2_g", "HER_clean"]].copy()

print(f"Dataset loaded. Total samples: {len(df_train)}")

# Define cross-validation strategy (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_r2s = []
fold_maes = []

print("\n--- Starting 5-Fold Cross-Validation ---")

for fold, (train_idx, test_idx) in enumerate(kf.split(df_train)):
    print(f"\nEvaluating Fold {fold + 1}/5...")
    
    train_fold = df_train.iloc[train_idx].copy()
    test_fold = df_train.iloc[test_idx].copy()
    
    # Initialize pipeline components
    autofeat = AutoFeaturizer(preset='express', n_jobs=1)
    cleaner = DataCleaner()
    reducer = FeatureReducer()
    
    # Filter the config_dict to keep only tree-based models and basic selectors/preprocessors
    allowed_keys = [
        'sklearn.ensemble.ExtraTreesRegressor',
        'sklearn.ensemble.GradientBoostingRegressor',
        'sklearn.ensemble.RandomForestRegressor',
        'xgboost.XGBRegressor',
        'sklearn.preprocessing.MinMaxScaler',
        'sklearn.preprocessing.StandardScaler',
        'sklearn.preprocessing.RobustScaler',
        'sklearn.preprocessing.Normalizer',
        'sklearn.decomposition.PCA',
        'sklearn.feature_selection.VarianceThreshold',
        'sklearn.feature_selection.SelectFromModel'
    ]
    custom_config = {k: v for k, v in regressor_config_dict.items() if k in allowed_keys}
    
    # Configure TPOT to run quickly and prioritize tree-based methods
    learner = TPOTAdaptor(
        generations=5,
        population_size=15,
        random_state=42 + fold,
        n_jobs=1,  # Keep single threaded to avoid Windows multiprocessing locks
        verbosity=0,
        config_dict=custom_config
    )
    
    # Create MatPipe
    pipe = MatPipe(autofeaturizer=autofeat, cleaner=cleaner, reducer=reducer, learner=learner)
    
    # Train MatPipe on training fold
    pipe.fit(train_fold, "HER_clean")
    
    # Predict on test fold
    predictions = pipe.predict(test_fold)
    
    # Evaluate
    y_true = test_fold["HER_clean"].values
    y_pred = predictions["HER_clean predicted"].values
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    fold_r2s.append(r2)
    fold_maes.append(mae)
    
    print(f"Fold {fold + 1} Metrics: R^2 = {r2:.3f}, MAE = {mae:.1f} umol g^-1 h^-1")

# Print overall cross-validation summary
avg_r2 = np.mean(fold_r2s)
avg_mae = np.mean(fold_maes)
print("\n=== Cross-Validation Results Summary ===")
print(f"Average R^2: {avg_r2:.3f}")
print(f"Average MAE: {avg_mae:.1f} umol g^-1 h^-1")

print("\n--- Training Final Model on Full Dataset ---")

# Initialize pipeline components for final model
autofeat_final = AutoFeaturizer(preset='express', n_jobs=1)
cleaner_final = DataCleaner()
reducer_final = FeatureReducer()

# Configure final TPOT regressor using tree-based custom config
allowed_keys = [
    'sklearn.ensemble.ExtraTreesRegressor',
    'sklearn.ensemble.GradientBoostingRegressor',
    'sklearn.ensemble.RandomForestRegressor',
    'xgboost.XGBRegressor',
    'sklearn.preprocessing.MinMaxScaler',
    'sklearn.preprocessing.StandardScaler',
    'sklearn.preprocessing.RobustScaler',
    'sklearn.preprocessing.Normalizer',
    'sklearn.decomposition.PCA',
    'sklearn.feature_selection.VarianceThreshold',
    'sklearn.feature_selection.SelectFromModel'
]
custom_config_final = {k: v for k, v in regressor_config_dict.items() if k in allowed_keys}

learner_final = TPOTAdaptor(
    generations=8,         # Slightly more generations for the final fit
    population_size=20,
    random_state=42,
    n_jobs=1,
    verbosity=2,
    config_dict=custom_config_final
)

# Create final MatPipe
pipe_final = MatPipe(autofeaturizer=autofeat_final, cleaner=cleaner_final, reducer=reducer_final, learner=learner_final)

# Fit on all data
pipe_final.fit(df_train, "HER_clean")

# -----------------------------------------------------------------
# Strip non-picklable DEAP/IO state from TPOT backend before saving.
# Only fitted_pipeline_ is needed for pipe.predict() at screening time.
# DEAP objects (toolbox, pset, pareto front, population), stdout handle,
# tqdm progress bar, and search-time buffers are all un-picklable and
# unnecessary after training is complete.
# -----------------------------------------------------------------
try:
    backend = pipe_final.learner.backend
    _NON_PICKLABLE_ATTRS = [
        '_file',            # sys.stdout  → TextIOWrapper, cannot pickle
        '_pbar',            # tqdm bar    → cannot pickle
        '_toolbox',         # DEAP Toolbox with bound methods/closures
        '_pset',            # DEAP PrimitiveSet
        '_pareto_front',    # DEAP ParetoFront (holds pareto_eq ref)
        '_pop',             # DEAP population list
        'pretest_X',        # numpy arrays held only for warm-start
        'pretest_y',
        '_start_datetime',
        '_last_pipeline_write',
        '_memory',          # joblib Memory object (if set)
    ]
    for attr in _NON_PICKLABLE_ATTRS:
        if hasattr(backend, attr):
            setattr(backend, attr, None)
    # evaluated_individuals_ is a plain dict – keep it (has no pickle issues)
    print("TPOT backend cleaned of non-picklable DEAP/IO state.")
except Exception as e:
    print(f"Warning: Could not fully clean TPOT backend before save: {e}")

# Save final model
model_path = "matpipe_model.joblib"
joblib.dump(pipe_final, model_path)
print(f"\nFinal model successfully saved to {model_path}")

# Output selected ML algorithm details
try:
    best_pipeline = pipe_final.learner.backend.fitted_pipeline_
    print(f"Best selected pipeline: {best_pipeline}")
except Exception as e:
    print(f"Could not retrieve backend pipeline details: {e}")

