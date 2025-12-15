from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset

from .dp import solve_knapsack_dp
from .graph_builder import build_knapsack_graph
from .io_excel import load_excel_knapsack_instances


class KnapsackDataset(InMemoryDataset):
    """In-memory dataset built from Excel instances.

    Each Excel instance is converted to a graph and labeled using DP.
    """

    def __init__(self, excel_path: str, transform=None, pre_transform=None):
        self._excel_path = excel_path
        super().__init__(root=".", transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self._generate()

    def _generate(self):
        data_list = []
        instances = load_excel_knapsack_instances(self._excel_path)

        for inst in instances:
            weights = inst["weights"]
            values = inst["values"]
            capacity = inst["capacity"]

            solution = solve_knapsack_dp(weights, values, capacity)
            graph = build_knapsack_graph(weights, values, capacity, solution)
            data_list.append(graph)

        return self.collate(data_list)


def _build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build a k-NN edge_index (directed) based on Euclidean distance."""
    n = x.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k, n - 1)
    dist = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))
    knn_idx = dist.topk(k_eff, largest=False).indices  # [n, k_eff]

    row = torch.arange(n).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    col = knn_idx.reshape(-1)
    return torch.stack([row, col], dim=0)


def _build_sparse_graph(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
    solution: Sequence[int],
    k: int = 16,
    use_knn: bool = True,
) -> Data:
    """Construct a sparse PyG Data object for one instance."""
    w = torch.tensor(weights, dtype=torch.float32)
    v = torch.tensor(values, dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)

    ratio = v / (w + 1e-8)
    w_norm = w / (w.max() + 1e-8)
    v_norm = v / (v.max() + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)
    cap_norm = torch.full_like(w_norm, float(capacity) / (w.sum() + 1e-8))

    x = torch.stack([w_norm, v_norm, ratio_norm, cap_norm], dim=1)  # [n, 4]

    if use_knn:
        edge_index = _build_knn_edges(x, k=k)
    else:
        # Simple sparse fallback: connect each node to the next k nodes in a ring.
        n = x.size(0)
        k_eff = min(k, max(1, n - 1))
        src = torch.arange(n).repeat_interleave(k_eff)
        offsets = torch.arange(1, k_eff + 1) % n
        dst = (torch.arange(n).unsqueeze(1) + offsets) % n
        edge_index = torch.stack([src, dst.reshape(-1)], dim=0)

    y = sol.unsqueeze(1)
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        wts=w.to(dtype=torch.float32),
        vals=v.to(dtype=torch.float32),
        cap=torch.tensor([capacity], dtype=torch.float32),
    )


class GeneratedKnapsack01Dataset(InMemoryDataset):
    """Dataset for generated NPZ knapsack instances (data_generate_01.py)."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        k: int = 16,
        use_knn: bool = True,
        use_cache: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.k = k
        self.use_knn = use_knn
        self.use_cache = use_cache
        # Explicitly avoid loading cached Data objects (torch.load on PyG Data can fail on 2.6+).
        self._cache_path = self.root_dir / "processed_dataset.pt"
        super().__init__(root=".", transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self._load_or_generate()

    def _npz_files(self) -> List[Path]:
        files = sorted(self.root_dir.glob("instance_*.npz"))
        if not files:
            raise FileNotFoundError(f"No instance_*.npz files found in {self.root_dir}")
        return files

    def _load_or_generate(self):
        if self._cache_path.exists():
            # Safely ignore or remove old cache to avoid torch.load pickling issues.
            try:
                if not self.use_cache:
                    self._cache_path.unlink()
            except OSError:
                pass

        data_list: List[Data] = []
        for path in self._npz_files():
            with np.load(path) as arrs:
                weights = arrs["weights"].tolist()
                values = arrs["values"].tolist()
                capacity = int(arrs["capacity"].item())
                solution = arrs["solution"].tolist()
            graph = _build_sparse_graph(
                weights=weights,
                values=values,
                capacity=capacity,
                solution=solution,
                k=self.k,
                use_knn=self.use_knn,
            )
            data_list.append(graph)

        data, slices = self.collate(data_list)
        return data, slices


def split_dataset_by_instances(
    dataset: InMemoryDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset by instance index (no node-level shuffling)."""
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_train + n_val + n_test))

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )
