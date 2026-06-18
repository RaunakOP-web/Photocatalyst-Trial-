import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

class DescriptorGNNHybridModel:
    def __init__(self, gnn_ensemble, active_features):
        self.gnn_ensemble = gnn_ensemble
        self.active_features = active_features
        self.regressor = None
        
    def fit(self, train_loader, X_train_tab, y_train, catboost_params=None):
        # 1. Extract GNN embeddings
        print("  Extracting GNN embeddings for training hybrid model...")
        train_embeds = self.gnn_ensemble.get_embeddings(train_loader)
        
        # 2. Concatenate with tabular features
        X_tab = X_train_tab[self.active_features].values
        X_combined = np.hstack([train_embeds, X_tab])
        
        # 3. Fit CatBoost regressor
        if catboost_params is None:
            catboost_params = {
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 6,
                "verbose": 0,
                "random_seed": 42
            }
            
        print("  Fitting CatBoost Regressor on hybrid GNN-descriptor embeddings...")
        self.regressor = CatBoostRegressor(**catboost_params)
        self.regressor.fit(X_combined, y_train.values)
        
    def predict(self, loader, X_tab_df):
        # 1. Extract GNN embeddings
        test_embeds = self.gnn_ensemble.get_embeddings(loader)
        
        # 2. Concatenate with tabular features
        X_tab = X_tab_df[self.active_features].values
        X_combined = np.hstack([test_embeds, X_tab])
        
        # 3. Predict
        return self.regressor.predict(X_combined)
