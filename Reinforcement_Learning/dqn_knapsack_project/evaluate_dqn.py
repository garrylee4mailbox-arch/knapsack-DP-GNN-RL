# Evaluation script for trained DQN on 0/1 Knapsack
# Run directly (no CLI args). Uses hard-coded dataset/model/output paths.

from __future__ import annotations
import os
import time
import csv
import json
from typing import List, Tuple

import numpy as np
import torch

from data import load_npz_instances, KnapsackInstance
from env import KnapsackEnv
from model import QNetwork

# Hard-coded defaults for click-to-run usage
DATASET_DIR = r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\dataset\knapsack01_medium"
MODEL_PATH = r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\knapsack-DP-GNN-DL\results\DQN\dqn.pt"
OUT_CSV = r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\knapsack-DP-GNN-DL\results\DQN\eval_results.csv"


def mark(msg: str):
    print(f"[EVAL] {msg}", flush=True)


def load_instances_from_directory(dataset_dir: str) -> Tuple[List[KnapsackInstance], List[str]]:
    """Load all .npz files (sorted) from a directory; keeps per-file instance ordering."""
    files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(".npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found in dataset directory: {dataset_dir}")

    all_instances: List[KnapsackInstance] = []
    file_names_per_instance: List[str] = []
    for fname in files:
        path = os.path.join(dataset_dir, fname)
        inst_list = load_npz_instances(path)
        all_instances.extend(inst_list)
        file_names_per_instance.extend([fname] * len(inst_list))
    return all_instances, file_names_per_instance


def select_action_greedy(q_values: np.ndarray, valid_mask: np.ndarray) -> int:
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def run_inference(model: QNetwork, device: torch.device, inst: KnapsackInstance) -> dict:
    env = KnapsackEnv(inst.weights, inst.values, inst.capacity, eps=1e-8)
    s = env.reset()
    done = False
    t0 = time.perf_counter()
    while not done:
        valid_mask = env.valid_actions_mask()
        with torch.no_grad():
            q = model(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
        a = select_action_greedy(q, valid_mask)
        out = env.step(a)
        s = out.next_state
        done = out.done
    t_ms = (time.perf_counter() - t0) * 1000.0

    total_weight = env.compute_solution_weight()
    total_value = env.compute_solution_value()
    feasible = 1 if total_weight <= float(inst.capacity) + 1e-6 else 0
    selected_indices = [i for i, val in enumerate(env.selection.tolist()) if val == 1]

    return {
        "instance_file": "",
        "n_items": int(len(inst.weights)),
        "capacity": int(inst.capacity),
        "total_weight_selected": float(total_weight),
        "total_value_selected": float(total_value),
        "feasible": feasible,
        "inference_time_ms": t_ms,
        "selected_items": json.dumps(selected_indices),
    }


def main():
    mark(f"Loading model from {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state_dim = int(ckpt["state_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    model_state = ckpt["model_state"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(state_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    mark(f"Loading dataset from {DATASET_DIR}")
    instances, file_names_per_instance = load_instances_from_directory(DATASET_DIR)
    mark(f"Loaded {len(instances)} instances from {len(file_names_per_instance)} instances/files")

    results = []
    for idx, inst in enumerate(instances):
        res = run_inference(model, device, inst)
        res["instance_file"] = file_names_per_instance[idx] if idx < len(file_names_per_instance) else f"instance_{idx:04d}"
        results.append(res)
        if idx % 100 == 0:
            mark(f"Evaluated {idx}/{len(instances)} instances")

    avg_value = float(np.mean([r["total_value_selected"] for r in results]))
    feasibility_rate = float(np.mean([r["feasible"] for r in results]))
    avg_time_ms = float(np.mean([r["inference_time_ms"] for r in results]))

    mark(f"avg_value={avg_value:.2f} feasibility_rate={feasibility_rate:.3f} avg_inference_time_ms={avg_time_ms:.2f}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance_file",
            "n_items",
            "capacity",
            "total_weight_selected",
            "total_value_selected",
            "feasible",
            "inference_time_ms",
            "selected_items",
        ])
        writer.writeheader()
        writer.writerows(results)

    mark(f"Wrote per-instance results to {OUT_CSV}")


if __name__ == "__main__":
    main()
