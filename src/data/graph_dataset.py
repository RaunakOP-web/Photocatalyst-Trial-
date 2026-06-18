import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from pymatgen.core import Element

from src.data.structure_generator import formula_to_structure

class CrystalGraphDataset(Dataset):
    def __init__(self, X_df, y_series, groups, active_features=None, cutoff=6.0, num_gaussians=30):
        super().__init__()
        self.X_df = X_df
        self.y_series = y_series
        self.groups = list(groups)
        
        self.cutoff = cutoff
        self.num_gaussians = num_gaussians
        self.active_features = active_features
        
        # Build Gaussian filter grid
        self.mu = np.linspace(0.0, cutoff, num_gaussians)
        self.sigma = (cutoff / num_gaussians) * 1.5
        
        # Precompute structures and GNN features to save time
        self.structures = [formula_to_structure(g) for g in self.groups]
        
    def len(self):
        return len(self.X_df)
        
    def get_element_features(self, symbol):
        try:
            el = Element(symbol)
            # 6-dimensional basic feature vector
            z = float(el.Z)
            x = float(el.X) if el.X is not None else 0.0
            row = float(el.row)
            group = float(el.group)
            r = float(el.atomic_radius) if el.atomic_radius is not None else 1.0
            valence = float(el.valence) if el.valence is not None else 0.0
            return [z, x, row, group, r, valence]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
    def get(self, idx):
        struct = self.structures[idx]
        
        # 1. Node features (x)
        node_features = []
        for site in struct:
            el_feat = self.get_element_features(site.specie.symbol)
            node_features.append(el_feat)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # 2. Edge index and Edge features (distances)
        edge_index = []
        edge_attr = []
        
        all_neighbors = struct.get_all_neighbors(self.cutoff, include_index=True)
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                d = neighbor.nn_distance
                
                # Gaussian expansion
                rbf = np.exp(-((d - self.mu) ** 2) / (self.sigma ** 2))
                
                edge_index.append([i, j])
                edge_attr.append(rbf)
                
        if len(edge_index) == 0:
            # dummy self-loops if no neighbors
            edge_index = [[i, i] for i in range(len(struct))]
            edge_attr = [np.zeros(self.num_gaussians) for _ in range(len(struct))]
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        
        # 3. Targets
        y = torch.tensor([self.y_series.iloc[idx]], dtype=torch.float)
        
        # 4. Tabular descriptor features (if specified)
        if self.active_features is not None:
            tab_feats = self.X_df.iloc[idx][self.active_features].values.astype(np.float32)
            tab_feats = torch.tensor(tab_feats, dtype=torch.float).unsqueeze(0)
        else:
            tab_feats = torch.zeros((1, 1))
            
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            tab_feats=tab_feats,
            group=self.groups[idx]
        )
