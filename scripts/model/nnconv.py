"""
GNN model using NNConv layers with edge features.
Improved version with better regularization and pooling strategies.
"""


import torch
from torch.nn import Linear, LayerNorm, Sequential, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool


class NNConvModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=256, num_layers=5, dropout=0.1):
        super().__init__()
        torch.manual_seed(117)
        
        # Store hyperparameters
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # More layers for better representation
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # Edge network dimensions
        edge_dim = 4  # QM9 has 4 edge features
        
        # First layer with deeper edge network
        edge_nn_1 = Sequential(
            Linear(edge_dim, hidden_channels * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels),
            ReLU(),
            Linear(hidden_channels, input_channels * hidden_channels)
        )
        self.convs.append(NNConv(input_channels, hidden_channels, edge_nn_1, aggr='add'))
        self.layer_norms.append(LayerNorm(hidden_channels))
        
        # Hidden layers with deeper edge networks
        for _ in range(num_layers - 1):
            edge_nn = Sequential(
                Linear(edge_dim, hidden_channels * 2),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_channels * 2, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels * hidden_channels)
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_nn, aggr='add'))
            self.layer_norms.append(LayerNorm(hidden_channels))
        
        # Output layers with dropout - using concatenated pooling
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)  # *2 for concat pooling
        self.lin2 = Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr):
        # Graph convolution layers with residual connections
        for i, (conv, ln) in enumerate(zip(self.convs, self.layer_norms)):
            x_new = conv(x, edge_index, edge_attr)
            x_new = ln(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Skip connection after first layer
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Combine mean and add pooling for richer representation
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_add], dim=-1)
        
        # Final prediction layers with deeper network
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        
        return x