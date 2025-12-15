# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Q for actions 0/1
        )

    def forward(self, x):
        return self.net(x)
