# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
import argparse
import csv
import os
import time
import random

import numpy as np
import torch

from config import DQNConfig
from data import load_npz_instances, split_instances
from env import KnapsackEnv
from model import QNetwork
from dp_baseline import dp_opt_value

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def greedy_action(q_values: np.ndarray, valid_mask: np.ndarray) -> int:
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_dp_capacity", type=int, default=20000,
                        help="Skip DP if capacity exceeds this (write -1 in dp_opt_value)")
    args = parser.parse_args()

    cfg = DQNConfig()
    set_seed(cfg.seed)

    instances = load_npz_instances(args.npz_path)
    train_set, val_set, test_set = split_instances(instances, cfg.seed, cfg.train_ratio, cfg.val_ratio)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dim = int(ckpt["state_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    device = torch.device(args.device)
    model = QNetwork(state_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = []
    t0 = time.time()
    for inst in test_set:
        env = KnapsackEnv(inst.weights, inst.values, inst.capacity, eps=cfg.eps)
        s = env.reset()
        done = False

        infer_start = time.time()
        while not done:
            mask = env.valid_actions_mask().astype(np.float32)
            with torch.no_grad():
                q = model(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
            a = greedy_action(q, mask)
            out = env.step(a)
            s = out.next_state
            done = out.done
        infer_time = time.time() - infer_start

        dqn_val = float(env.compute_solution_value())
        dqn_w = float(env.compute_solution_weight())
        feasible = (dqn_w <= float(inst.capacity) + 1e-6)

        # DP oracle value (optional skip for huge capacity)
        if inst.capacity <= args.max_dp_capacity:
            dp_val = int(dp_opt_value(inst.weights, inst.values, inst.capacity))
        else:
            dp_val = -1

        ratio = float(dqn_val / dp_val) if dp_val > 0 else float('nan')

        rows.append({
            "instance_id": inst.instance_id,
            "n_items": int(len(inst.weights)),
            "capacity": int(inst.capacity),
            "dp_opt_value": int(dp_val),
            "dqn_value": float(dqn_val),
            "optimality_ratio": float(ratio),
            "weight_used": float(dqn_w),
            "feasible": bool(feasible),
            "inference_time_sec": float(infer_time),
        })

    total_eval_time = time.time() - t0

    fieldnames = [
        "instance_id", "n_items", "capacity",
        "dp_opt_value", "dqn_value", "optimality_ratio",
        "weight_used", "feasible", "inference_time_sec"
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote results to: {args.out_csv}")
    print(f"Eval time (total): {total_eval_time:.2f}s for {len(rows)} instances")

if __name__ == "__main__":
    main()
