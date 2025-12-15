from typing import List

import torch
from torch_geometric.data import Data


def build_knapsack_graph(
    weights: List[int],
    values: List[int],
    capacity: int,
    solution: List[int],
) -> Data:
    """Convert one knapsack instance into a PyG graph (Data).

    Node features:
        [weight_norm, value_norm, ratio_norm]

    Edges:
        Fully-connected directed edges (including self-loops): O(n^2).
        WARNING: This is not suitable for very large n (e.g., 5k+).
        In the next iteration, replace with a sparse graph (e.g., kNN graph).

    Labels:
        Node-level 0/1 selection from DP.
    """
    n = len(weights)
    w = torch.tensor(weights, dtype=torch.float32)
    v = torch.tensor(values, dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)  # [n]

    ratio = v / (w + 1e-8)

    w_norm = w / (w.max() + 1e-8)
    v_norm = v / (v.max() + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)

    x = torch.stack([w_norm, v_norm, ratio_norm], dim=1)  # [n, 3]

    idx = torch.arange(n)
    row, col = torch.meshgrid(idx, idx, indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)  # [2, n^2]

    y = sol.unsqueeze(1)  # [n, 1]

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        weights=w,
        values=v,
        capacity=torch.tensor([capacity], dtype=torch.float32),
    )
