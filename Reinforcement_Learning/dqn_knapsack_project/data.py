# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

@dataclass
class KnapsackInstance:
    instance_id: int
    weights: np.ndarray  # shape [n]
    values: np.ndarray   # shape [n]
    capacity: int

def _to_1d_int_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    # allow float in npz -> cast
    return arr.astype(np.int64)

def _to_int(x) -> int:
    # handles numpy scalar
    return int(np.asarray(x).reshape(()))

def load_npz_instances(npz_path: str) -> List[KnapsackInstance]:
    """Load knapsack instances from an .npz file.

    Supports formats:
    - Object arrays: weights[i] is an array, values[i] is an array, capacity[i] scalar
    - Dense arrays: weights shape [N, n], values shape [N, n], capacity shape [N]
    """
    data = np.load(npz_path, allow_pickle=True)

    # Common key variants
    def pick_key(candidates):
        for k in candidates:
            if k in data.files:
                return k
        return None

    k_w = pick_key(["weights", "w", "W"])
    k_v = pick_key(["values", "v", "V"])
    k_c = pick_key(["capacity", "cap", "C"])

    if k_w is None or k_v is None or k_c is None:
        raise KeyError(f".npz must contain weights/values/capacity keys; found keys={data.files}")

    W = data[k_w]
    V = data[k_v]
    C = data[k_c]

    instances: List[KnapsackInstance] = []

    # Object array case
    if W.dtype == object or V.dtype == object:
        N = len(W)
        for i in range(N):
            wi = _to_1d_int_array(W[i])
            vi = _to_1d_int_array(V[i])
            ci = _to_int(C[i] if np.asarray(C).ndim > 0 else C)
            if wi.shape[0] != vi.shape[0]:
                raise ValueError(f"Instance {i}: weights and values length mismatch: {wi.shape[0]} vs {vi.shape[0]}")
            instances.append(KnapsackInstance(i, wi, vi, ci))
        return instances

    # Dense array case
    W = np.asarray(W)
    V = np.asarray(V)
    C = np.asarray(C)

    # Allow single-instance files saved as 1D arrays (directory-per-instance datasets)
    if W.ndim == 1 and V.ndim == 1:
        W = W[None, :]
        V = V[None, :]
    elif W.ndim == 1 and V.ndim == 2:
        raise ValueError(f"weights ndim=1 but values ndim=2; expected both 1D or both 2D in {npz_path}")
    elif W.ndim == 2 and V.ndim == 1:
        raise ValueError(f"weights ndim=2 but values ndim=1; expected both 1D or both 2D in {npz_path}")

    if W.ndim != 2 or V.ndim != 2:
        raise ValueError(f"Dense format expects weights/values as 2D arrays; got W.ndim={W.ndim}, V.ndim={V.ndim}")
    if W.shape != V.shape:
        raise ValueError(f"Dense format expects weights/values same shape; got {W.shape} vs {V.shape}")

    # Allow scalar capacity for single-instance files (directory-per-instance datasets)
    if C.ndim == 0:
        if W.shape[0] == 1:
            C = np.array([_to_int(C)])
        else:
            raise ValueError(f"capacity is scalar but batch size is {W.shape[0]} in {npz_path}; expected 1D capacity per instance")
    if C.ndim != 1 or C.shape[0] != W.shape[0]:
        raise ValueError(f"Dense format expects capacity shape [{W.shape[0]}]; got {C.shape} for N={W.shape[0]}")

    N, n = W.shape
    for i in range(N):
        instances.append(KnapsackInstance(i, _to_1d_int_array(W[i]), _to_1d_int_array(V[i]), _to_int(C[i])))
    return instances

def split_instances(instances: List[KnapsackInstance], seed: int, train_ratio: float, val_ratio: float):
    """Deterministic shuffle + split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(instances))
    rng.shuffle(idx)

    n = len(instances)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n_train, 1)
    n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else max(n_val, 0)

    train_ids = idx[:n_train]
    val_ids = idx[n_train:n_train + n_val]
    test_ids = idx[n_train + n_val:]

    train = [instances[i] for i in train_ids]
    val = [instances[i] for i in val_ids]
    test = [instances[i] for i in test_ids]
    return train, val, test
