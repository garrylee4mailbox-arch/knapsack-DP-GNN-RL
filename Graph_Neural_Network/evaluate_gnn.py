# Per-instance GNN evaluation (aligned with DP/DQN CSV schema)
# Run directly (no CLI args). Uses hard-coded paths.

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from knapsack_gnn.model import KnapsackGNN


# Hard-coded defaults
DATASET_DIR = Path(r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\dataset\knapsack01_medium")
MODEL_PATH = Path(r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\knapsack-DP-GNN-DL\results\GNN\gnn.pt")
OUT_CSV = Path(r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\knapsack-DP-GNN-DL\results\GNN\gnn_eval_results.csv")


def mark(msg: str):
    print(f"[GNN-EVAL] {msg}", flush=True)


def greedy_feasible_decode(probs: torch.Tensor, weights: torch.Tensor, capacity: float) -> torch.Tensor:
    """Greedy decode: pick items by descending prob while staying within capacity."""
    idx = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total_w = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total_w + w_i <= capacity:
            x_hat[i] = 1.0
            total_w += w_i
    return x_hat


def build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build a directed k-NN edge_index based on Euclidean distance."""
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


def build_graph(weights: np.ndarray, values: np.ndarray, capacity: int, k: int = 16) -> Data:
    """Replicate training-time graph construction (features + kNN edges)."""
    w = torch.tensor(weights, dtype=torch.float32)
    v = torch.tensor(values, dtype=torch.float32)
    ratio = v / (w + 1e-8)
    w_norm = w / (w.max() + 1e-8)
    v_norm = v / (v.max() + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)
    cap_norm = torch.full_like(w_norm, float(capacity) / (w.sum() + 1e-8))
    x = torch.stack([w_norm, v_norm, ratio_norm, cap_norm], dim=1)  # [n, 4]
    edge_index = build_knn_edges(x, k=k)
    return Data(
        x=x,
        edge_index=edge_index,
        wts=w,
        vals=v,
        cap=torch.tensor([capacity], dtype=torch.float32),
    )


def load_instance(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    arr = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in arr.files:
                return arr[k]
        return None

    W = pick(["weights", "w", "W"])
    V = pick(["values", "v", "V"])
    C = pick(["capacity", "cap", "C"])
    if W is None or V is None or C is None:
        raise KeyError(f"Missing weights/values/capacity in {npz_path}, found keys={arr.files}")
    W = np.asarray(W).reshape(-1)
    V = np.asarray(V).reshape(-1)
    C = int(np.asarray(C).reshape(()))
    if W.shape != V.shape:
        raise ValueError(f"weights and values shape mismatch in {npz_path}: {W.shape} vs {V.shape}")
    return W.astype(np.float32), V.astype(np.float32), C


def load_model(model_path: Path, in_dim: int, device: torch.device) -> KnapsackGNN:
    model = KnapsackGNN(in_dim=in_dim, hidden_dim=64).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    files = sorted(DATASET_DIR.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(f"No instance_*.npz files found in {DATASET_DIR}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset dir: {DATASET_DIR}")
    mark(f"Model path: {MODEL_PATH}")
    mark(f"Output CSV: {OUT_CSV}")

    # Build one sample graph to infer input dim (should be 4)
    sample_w, sample_v, sample_c = load_instance(files[0])
    sample_graph = build_graph(sample_w, sample_v, sample_c, k=16)
    in_dim = sample_graph.num_node_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, in_dim=in_dim, device=device)

    results = []
    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        graph = build_graph(W, V, C, k=16)
        graph = graph.to(device)

        t0 = time.perf_counter()
        logits = model(graph)
        probs = torch.sigmoid(logits).detach().cpu()
        x_hat = greedy_feasible_decode(probs, graph.wts.cpu(), float(C))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = float((x_hat * graph.wts.cpu()).sum().item())
        total_value = float((x_hat * graph.vals.cpu()).sum().item())
        feasible = 1 if total_weight <= float(C) + 1e-6 else 0
        selected_indices = [int(i) for i, val in enumerate(x_hat.tolist()) if val == 1.0]

        results.append(
            {
                "instance_file": path.name,
                "n_items": int(len(W)),
                "capacity": float(C),
                "total_weight": total_weight,
                "total_value": total_value,
                "feasible": feasible,
                "inference_time_ms": infer_ms,
                "selected_items": json.dumps(selected_indices),
            }
        )
        if (idx + 1) % 100 == 0:
            mark(f"Evaluated {idx+1}/{len(files)} instances")

    avg_value = float(np.mean([r["total_value"] for r in results]))
    feasibility_rate = float(np.mean([r["feasible"] for r in results]))
    avg_time_ms = float(np.mean([r["inference_time_ms"] for r in results]))
    mark(f"avg_value={avg_value:.2f} feasibility_rate={feasibility_rate:.3f} avg_inference_time_ms={avg_time_ms:.2f}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_file",
                "n_items",
                "capacity",
                "total_weight",
                "total_value",
                "feasible",
                "inference_time_ms",
                "selected_items",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    mark(f"Wrote per-instance results to {OUT_CSV}")


if __name__ == "__main__":
    main()
