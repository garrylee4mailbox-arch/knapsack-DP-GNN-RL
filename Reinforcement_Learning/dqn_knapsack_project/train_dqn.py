# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
import argparse
import os
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import DQNConfig
from data import load_npz_instances, split_instances, KnapsackInstance
from env import KnapsackEnv
from model import QNetwork
from replay import ReplayBuffer, Transition

# DEBUG: limit number of files loaded; set to an int (e.g., 20) when diagnosing slow I/O
DEBUG_MAX_FILES = None

# Default paths for click-to-run usage (dataset auto-loading + output)
DEFAULT_DATASET_DIR = r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\dataset\knapsack01_medium"
# Compute project root relative to this file to avoid user-specific hard-coding
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "results" / "DQN")
# CHECKPOINT: frequency and filenames
CHECKPOINT_EVERY = 10_000
LATEST_CKPT = "checkpoint_latest.pt"

# DEBUG: lightweight logging helpers
def mark(msg: str):
    print(f"[DQN] {msg}", flush=True)

def timed(msg: str, fn):
    t0 = time.time()
    mark(f"{msg} (start)")
    out = fn()
    mark(f"{msg} (done in {time.time() - t0:.2f}s)")
    return out

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    t = step / max(1, cfg.eps_decay_steps)
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)

def pick_action(q_values: np.ndarray, valid_mask: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    # valid_mask shape [2] in {0,1}
    if rng.random() < eps:
        valid_actions = np.where(valid_mask > 0.5)[0]
        return int(rng.choice(valid_actions))
    # greedy with mask
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))

def load_instances_from_directory(dataset_dir: str) -> Tuple[List[KnapsackInstance], List[str]]:
    """Auto-load every .npz instance file in the dataset directory (sorted for determinism)."""
    files = sorted(
        [f for f in os.listdir(dataset_dir) if f.lower().endswith(".npz")]
    )
    if not files:
        raise FileNotFoundError(f"No .npz files found in dataset directory: {dataset_dir}")

    # DEBUG: optionally load only a subset to diagnose slow loading
    if DEBUG_MAX_FILES is not None:
        files = files[:DEBUG_MAX_FILES]

    all_instances: List[KnapsackInstance] = []
    for fname in files:
        idx = len(all_instances)
        total_files = len(files)
        if idx % 20 == 0:
            print(f"[DQN] loading file {idx} / {total_files}: {fname}")  # DEBUG
        t0 = time.time()  # DEBUG
        path = os.path.join(dataset_dir, fname)
        inst_list = load_npz_instances(path)
        all_instances.extend(inst_list)
        dur = time.time() - t0  # DEBUG
        if dur > 0.5:
            print(f"[DQN] slow file detected: {fname} ({dur:.2f}s)")  # DEBUG

    # Re-index to keep ids unique/ordered across concatenated files.
    for idx, inst in enumerate(all_instances):
        inst.instance_id = idx
    return all_instances, [os.path.join(dataset_dir, f) for f in files]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_steps", type=int, default=50_000, help="Override train steps (default 50k)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Directory of per-instance .npz files")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for checkpoints/metadata")
    args = parser.parse_args()

    cfg = DQNConfig()
    if args.train_steps is not None:
        cfg.train_steps = int(args.train_steps)
    mark(f"train_steps={cfg.train_steps}")  # DEBUG

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    mark(f"output_dir = {os.path.abspath(args.out_dir)}")  # DEBUG
    mark("start loading dataset...")  # DEBUG
    instances, dataset_files = load_instances_from_directory(args.dataset_dir)
    mark(f"finished loading {len(instances)} instances from {len(dataset_files)} files")  # DEBUG

    train_set, val_set, test_set = split_instances(instances, cfg.seed, cfg.train_ratio, cfg.val_ratio)

    # step A: build env and state dim
    def init_env0():
        env0_local = KnapsackEnv(train_set[0].weights, train_set[0].values, train_set[0].capacity, eps=cfg.eps)
        sd = env0_local.reset().shape[0]
        return env0_local, sd
    env0, state_dim = timed("step A: building environments", init_env0)  # DEBUG

    # step B: init networks/optimizer
    def init_models():
        dev = torch.device(args.device)
        on = QNetwork(state_dim, hidden_dim=args.hidden_dim).to(dev)
        tgt = QNetwork(state_dim, hidden_dim=args.hidden_dim).to(dev)
        tgt.load_state_dict(on.state_dict())
        tgt.eval()
        opt_local = torch.optim.Adam(on.parameters(), lr=cfg.lr)
        return dev, on, tgt, opt_local
    device, online, target, opt = timed("step B: initializing networks/optimizer", init_models)  # DEBUG

    # CHECKPOINT: optional resume from latest
    ckpt_latest_path = os.path.join(args.out_dir, LATEST_CKPT)
    step_count = 0
    updates = 0
    if os.path.exists(ckpt_latest_path):
        mark("checkpoint detected; attempting resume")  # CHECKPOINT
        ckpt = torch.load(ckpt_latest_path, map_location=device)
        online.load_state_dict(ckpt["online_state"])
        target.load_state_dict(ckpt["target_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        step_count = int(ckpt.get("step", 0))
        updates = int(ckpt.get("updates", 0))
        mark(f"Resumed from checkpoint at step={step_count}")  # CHECKPOINT
        mark("Replay buffer not restored; training will warm up again")  # CHECKPOINT

    # step C: build replay buffer
    buffer = timed("step C: building replay buffer", lambda: ReplayBuffer(cfg.buffer_size, seed=cfg.seed))  # DEBUG

    mark("step D: starting training loop...")  # DEBUG
    # training loop: sample instances uniformly; each rollout is one episode for that instance
    start_time = time.time()
    last_heartbeat = start_time  # DEBUG
    last_loss = None  # DEBUG

    while step_count < cfg.train_steps:
        inst = train_set[int(rng.integers(0, len(train_set)))]
        env = KnapsackEnv(inst.weights, inst.values, inst.capacity, eps=cfg.eps)
        s = env.reset()
        done = False

        if step_count == 0:
            mark("step E: first training iteration reached")  # DEBUG

        while not done and step_count < cfg.train_steps:
            valid_mask = env.valid_actions_mask()  # [2]
            eps = epsilon_by_step(step_count, cfg)
            with torch.no_grad():
                q = online(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
            a = pick_action(q, valid_mask, eps, rng)

            out = env.step(a)
            s2 = out.next_state
            done = out.done
            valid_mask2 = env.valid_actions_mask() if not done else np.array([1, 0], dtype=np.int64)

            buffer.push(Transition(s=s, a=int(a), r=float(out.reward), s2=s2, done=bool(done), mask2=valid_mask2))
            s = s2
            step_count += 1

            # optimize
            if len(buffer) >= cfg.min_buffer_size and step_count % 1 == 0:
                s_b, a_b, r_b, s2_b, d_b, m2_b = buffer.sample(cfg.batch_size)

                s_t = torch.from_numpy(s_b).to(device)
                a_t = torch.from_numpy(a_b).to(device)
                r_t = torch.from_numpy(r_b).to(device)
                s2_t = torch.from_numpy(s2_b).to(device)
                d_t = torch.from_numpy(d_b).to(device)
                m2_t = torch.from_numpy(m2_b).to(device)

                q_sa = online(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    q2 = target(s2_t)  # [B,2]
                    # mask invalid actions at next state
                    q2 = q2 + (m2_t - 1.0) * 1e9  # invalid -> -1e9
                    max_q2 = q2.max(dim=1).values
                    y = r_t + (1.0 - d_t) * cfg.gamma * max_q2

                loss = F.smooth_l1_loss(q_sa, y)
                last_loss = float(loss.item())  # DEBUG

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip_norm)
                opt.step()
                updates += 1

                if step_count % cfg.target_update_steps == 0:
                    target.load_state_dict(online.state_dict())

                # CHECKPOINT: periodic save
                if step_count % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(args.out_dir, f"checkpoint_step_{step_count}.pt")
                    torch.save({
                        "online_state": online.state_dict(),
                        "target_state": target.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "step": step_count,
                        "updates": updates,
                        "config": cfg.__dict__,
                    }, ckpt_path)
                    torch.save({
                        "online_state": online.state_dict(),
                        "target_state": target.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "step": step_count,
                        "updates": updates,
                        "config": cfg.__dict__,
                    }, ckpt_latest_path)
                    mark(f"Checkpoint saved at step={step_count}")  # CHECKPOINT

            # DEBUG: heartbeat to confirm loop is alive
            if step_count == 0 or step_count % 500 == 0:
                mark(f"step={step_count} eps={eps:.3f} buf={len(buffer)} loss={last_loss}")  # DEBUG
                last_heartbeat = time.time()

        # optional: quick heartbeat logging
        if step_count % 20_000 == 0:
            elapsed = time.time() - start_time
            print(f"steps={step_count} updates={updates} eps={epsilon_by_step(step_count, cfg):.3f} elapsed={elapsed:.1f}s")

    train_time = time.time() - start_time

    ckpt_path = os.path.join(args.out_dir, "dqn.pt")
    meta_path = os.path.join(args.out_dir, "train_meta.json")

    torch.save({
        "state_dim": state_dim,
        "hidden_dim": args.hidden_dim,
        "model_state": online.state_dict(),
        "config": cfg.__dict__,
    }, ckpt_path)

    # CHECKPOINT: final latest checkpoint
    torch.save({
        "online_state": online.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": opt.state_dict(),
        "step": step_count,
        "updates": updates,
        "config": cfg.__dict__,
    }, os.path.join(args.out_dir, LATEST_CKPT))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_time_sec": train_time,
            "train_steps": cfg.train_steps,
            "updates": updates,
            "seed": cfg.seed,
            "dataset_dir": os.path.normpath(args.dataset_dir),
            "dataset_files": dataset_files,
            "split": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        }, f, indent=2)

    print(f"Saved model to: {ckpt_path}")
    print(f"Saved meta  to: {meta_path}")

if __name__ == "__main__":
    main()
