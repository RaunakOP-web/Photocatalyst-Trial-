import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class MEGNetBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim):
        super().__init__()
        # Edge update function
        self.edge_phi = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + global_dim, edge_dim),
            nn.BatchNorm1d(edge_dim),
            nn.ReLU()
        )
        
        # Node update function
        self.node_phi = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.ReLU()
        )
        
        # Global update function
        self.global_phi = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        
        # Get global state corresponding to each edge
        # batch[row] gives the graph index of each edge source node
        u_edge = u[batch[row]]
        
        # 1. Update Edges
        edge_input = torch.cat([x[row], x[col], edge_attr, u_edge], dim=-1)
        edge_attr_new = self.edge_phi(edge_input) + edge_attr
        
        # Aggregate edges for each node
        # We can use scatter_mean or a manual sum/mean using torch.zeros_like
        edge_agg = torch.zeros_like(x)
        ones = torch.ones_like(col, dtype=torch.float).unsqueeze(-1)
        # Safe aggregation without requiring torch-scatter
        node_edge_sum = torch.zeros((x.size(0), edge_attr_new.size(1)), device=x.device)
        node_edge_count = torch.zeros((x.size(0), 1), device=x.device)
        
        node_edge_sum.index_add_(0, row, edge_attr_new)
        node_edge_count.index_add_(0, row, ones)
        
        node_edge_mean = node_edge_sum / (node_edge_count + 1e-6)
        
        # 2. Update Nodes
        u_node = u[batch]
        node_input = torch.cat([x, node_edge_mean, u_node], dim=-1)
        x_new = self.node_phi(node_input) + x
        
        # 3. Update Globals
        # Mean pool nodes per graph
        x_graph = global_mean_pool(x_new, batch)
        
        # Mean pool edges per graph
        # Map each edge to its graph index
        edge_batch = batch[row]
        num_graphs = u.size(0)
        edge_graph_sum = torch.zeros((num_graphs, edge_attr_new.size(1)), device=x.device)
        edge_graph_count = torch.zeros((num_graphs, 1), device=x.device)
        edge_ones = torch.ones_like(edge_batch, dtype=torch.float).unsqueeze(-1)
        
        edge_graph_sum.index_add_(0, edge_batch, edge_attr_new)
        edge_graph_count.index_add_(0, edge_batch, edge_ones)
        
        edge_graph_mean = edge_graph_sum / (edge_graph_count + 1e-6)
        
        global_input = torch.cat([x_graph, edge_graph_mean, u], dim=-1)
        u_new = self.global_phi(global_input) + u
        
        return x_new, edge_attr_new, u_new

class MEGNet(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=30, in_global_dim=78, node_dim=64, edge_dim=64, global_dim=64, num_blocks=3, out_dim=1):
        super().__init__()
        self.node_embedding = nn.Linear(in_node_dim, node_dim)
        self.edge_embedding = nn.Linear(in_edge_dim, edge_dim)
        self.global_embedding = nn.Linear(in_global_dim, global_dim)
        
        self.blocks = nn.ModuleList([
            MEGNetBlock(node_dim, edge_dim, global_dim) for _ in range(num_blocks)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(node_dim + global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # In MEGNet, the global state is initialized from the experimental descriptors
        u = data.tab_feats
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            
        h_x = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr)
        h_u = self.global_embedding(u)
        
        # Message passing blocks
        for block in self.blocks:
            h_x, h_e, h_u = block(h_x, edge_index, h_e, h_u, batch)
            
        # Global readout: concatenate mean node features and global features
        x_graph = global_mean_pool(h_x, batch)
        g = torch.cat([x_graph, h_u], dim=-1)
        
        # Output prediction
        out = self.fc(g).squeeze(-1)
        return out
        
    def get_embeddings(self, data):
        """Extract GNN embedding representation for hybrid model training."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        u = data.tab_feats
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            
        h_x = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr)
        h_u = self.global_embedding(u)
        
        for block in self.blocks:
            h_x, h_e, h_u = block(h_x, edge_index, h_e, h_u, batch)
            
        x_graph = global_mean_pool(h_x, batch)
        g = torch.cat([x_graph, h_u], dim=-1)
        return g
