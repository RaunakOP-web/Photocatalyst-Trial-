import os
import copy
import torch
import numpy as np

class GNNElementEnsemble:
    def __init__(self, model_class, model_kwargs, num_seeds=5):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.num_seeds = num_seeds
        self.seeds = [42 + i for i in range(num_seeds)]
        self.models = []
        
    def fit(self, train_loader, val_loader=None, epochs=50, lr=0.005, weight_decay=1e-5, device="cpu"):
        self.models = []
        for seed in self.seeds:
            print(f"  Training ensemble member with seed {seed}...")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = self.model_class(**self.model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.MSELoss()
            
            best_val_loss = float("inf")
            best_model_weights = None
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch.num_graphs
                    
                train_loss /= len(train_loader.dataset)
                
                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            loss = criterion(out, batch.y)
                            val_loss += loss.item() * batch.num_graphs
                    val_loss /= len(val_loader.dataset)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    best_model_weights = copy.deepcopy(model.state_dict())
                    
            model.load_state_dict(best_model_weights)
            self.models.append(model)
            
    def predict(self, loader, device="cpu"):
        """Predicts mean and standard deviation (uncertainty) for all graphs in loader."""
        all_preds = []
        for model in self.models:
            model.eval()
            member_preds = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out = model(batch)
                    member_preds.extend(out.cpu().numpy().tolist())
            all_preds.append(member_preds)
            
        all_preds = np.array(all_preds) # shape: (num_seeds, num_samples)
        mean_preds = all_preds.mean(axis=0)
        std_preds = all_preds.std(axis=0)
        
        return mean_preds, std_preds
        
    def get_embeddings(self, loader, device="cpu"):
        """Extract average graph embeddings from the ensemble members."""
        all_embeds = []
        for model in self.models:
            model.eval()
            member_embeds = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    emb = model.get_embeddings(batch)
                    member_embeds.extend(emb.cpu().numpy().tolist())
            all_embeds.append(member_embeds)
        
        all_embeds = np.array(all_embeds) # shape: (num_seeds, num_samples, embedding_dim)
        mean_embeds = all_embeds.mean(axis=0)
        return mean_embeds
