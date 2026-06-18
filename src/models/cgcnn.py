import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class CGCNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.fc_filter = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.fc_core = nn.Linear(2 * node_dim + edge_dim, node_dim)
        
        self.bn_filter = nn.BatchNorm1d(node_dim)
        self.bn_core = nn.BatchNorm1d(node_dim)
        
        self.act_filter = nn.Sigmoid()
        self.act_core = nn.Softplus()
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        filt = self.act_filter(self.bn_filter(self.fc_filter(z)))
        core = self.act_core(self.bn_core(self.fc_core(z)))
        return filt * core
        
    def update(self, aggr_out, x):
        return x + aggr_out

class CGCNN(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=30, node_dim=64, num_convs=3, out_dim=1):
        super().__init__()
        self.embedding = nn.Linear(in_node_dim, node_dim)
        
        self.convs = nn.ModuleList([
            CGCNNConv(node_dim, in_edge_dim) for _ in range(num_convs)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial projection
        h = self.embedding(x)
        
        # Message passing layers
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            
        # Global pooling
        g = global_mean_pool(h, batch)
        
        # Output prediction
        out = self.fc(g).squeeze(-1)
        return out
        
    def get_embeddings(self, data):
        """Extract graph embedding representation for hybrid model training."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.embedding(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        g = global_mean_pool(h, batch)
        return g
