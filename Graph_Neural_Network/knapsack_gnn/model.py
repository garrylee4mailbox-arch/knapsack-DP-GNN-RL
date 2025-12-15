import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class KnapsackGNN(nn.Module):
    """A simple 2-layer GCN for node-level selection prediction."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        logits = self.lin(x).squeeze(-1)  # [num_nodes]
        return logits
