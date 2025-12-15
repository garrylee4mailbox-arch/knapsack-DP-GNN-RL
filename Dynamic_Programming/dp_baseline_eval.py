# DP baseline evaluation for 0/1 Knapsack (per-instance CSV aligned with DQN output)
# Run directly (no CLI args). Uses hard-coded dataset/output paths.

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


DATASET_DIR = Path(r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\dataset\knapsack01_medium")
OUT_CSV = Path(r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\knapsack-DP-GNN-DL\results\DP\dp_results.csv")


def mark(msg: str):
    print(f"[DP-EVAL] {msg}", flush=True)


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

    W = np.asarray(W).astype(int).reshape(-1)
    V = np.asarray(V).astype(int).reshape(-1)
    C = int(np.asarray(C).reshape(()))
    if W.shape != V.shape:
        raise ValueError(f"weights and values shape mismatch in {npz_path}: {W.shape} vs {V.shape}")
    return W, V, C


def solve_knapsack_dp(weights: np.ndarray, values: np.ndarray, capacity: int) -> List[int]:
    n = len(weights)
    # dp[w] is max value; keep choice to backtrack
    dp = [0] * (capacity + 1)
    choice = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        w_i = int(weights[i])
        v_i = int(values[i])
        for w in range(capacity, w_i - 1, -1):
            take = dp[w - w_i] + v_i
            if take > dp[w]:
                dp[w] = take
                choice[i][w] = True

    # backtrack
    selected = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if choice[i][w]:
            selected.append(i)
            w -= int(weights[i])
    selected.reverse()
    return selected


def main():
    files = sorted(DATASET_DIR.glob("instance_*.npz"))
    if not files:
        raise FileNotFoundError(f"No instance_*.npz files found in {DATASET_DIR}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mark(f"Dataset dir: {DATASET_DIR}")
    mark(f"Output CSV: {OUT_CSV}")

    results = []
    for idx, path in enumerate(files):
        W, V, C = load_instance(path)
        t0 = time.perf_counter()
        selected_idx = solve_knapsack_dp(W, V, C)
        t_ms = (time.perf_counter() - t0) * 1000.0

        total_weight = int(W[selected_idx].sum()) if selected_idx else 0
        total_value = int(V[selected_idx].sum()) if selected_idx else 0

        results.append(
            {
                "instance_file": path.name,
                "n_items": int(W.shape[0]),
                "capacity": int(C),
                "total_weight": total_weight,
                "total_value": total_value,
                "feasible": 1,
                "inference_time_ms": t_ms,
                "selected_items": json.dumps(selected_idx),
            }
        )
        if (idx + 1) % 100 == 0:
            mark(f"Processed {idx+1}/{len(files)} instances")

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

    mark(f"Wrote DP results to {OUT_CSV}")
    avg_value = float(np.mean([r["total_value"] for r in results]))
    avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
    mark(f"Avg value={avg_value:.2f}, Avg time={avg_time:.2f} ms over {len(results)} instances")


if __name__ == "__main__":
    main()
