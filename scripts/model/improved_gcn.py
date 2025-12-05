"""
Improved GCN model with more layers, residual connections, and enhanced pooling.
"""


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


class ImprovedGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels=128, num_layers=5):
        super().__init__()
        torch.manual_seed(117)
        
        # More layers for better representation
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_channels, hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        # Output layers with dropout
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.dropout = 0.3

    def forward(self, x, edge_index, batch):
        # Graph convolution layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            
            # Skip connection after first layer
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling (try both mean and sum)
        x = global_mean_pool(x, batch)
        
        # Final prediction layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x
