import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv

class GATv2Model(nn.Module):
    def __init__(self, in_features, edge_dim, hidden_features, out_features, num_heads, num_layers, dropout):
        super(GATv2Model, self).__init__()
        #Input layer
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_features, hidden_features, edge_dim=edge_dim, heads=num_heads, dropout=dropout))
        #Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_features * num_heads, hidden_features, edge_dim=edge_dim, heads=num_heads, dropout=dropout))
        
        #GAT output layer
        self.layers.append(GATv2Conv(hidden_features * num_heads, out_features, edge_dim=edge_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = nn.functional.relu(x)
        return x  
