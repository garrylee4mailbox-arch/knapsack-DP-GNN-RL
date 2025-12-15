# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

## DQN Knapsack (0/1) – Modular Reference Implementation

This folder contains a modular Deep Q-Network (DQN) implementation for **0/1 Knapsack** designed to be comparable with a GNN baseline and a DP oracle.

### Key Design Choices
- **Cross-instance training**: train once on a training split, evaluate on a test split.
- **Feasibility guaranteed**: invalid action (take when overweight) is masked out and never executed.
- **State contains generalizable features**: current item features + remaining capacity + summary stats of remaining items.

### Files
- `data.py` – `.npz` dataset loader + train/val/test split
- `env.py` – knapsack environment (sequential decision)
- `model.py` – Q-network (MLP)
- `replay.py` – replay buffer
- `dp_baseline.py` – DP oracle (value only) for evaluation ratios
- `train_dqn.py` – training script (saves model + training metadata)
- `eval_dqn.py` – evaluation script (writes `results_dqn.csv`)
- `metrics.py` – helper metrics

### Run
```bash
python train_dqn.py --npz_path path/to/dataset.npz --out_dir out_dqn
python eval_dqn.py --npz_path path/to/dataset.npz --ckpt out_dqn/dqn.pt --out_csv out_dqn/results_dqn.csv
```

### Notes on `.npz` formats
The loader supports common patterns:
1) Object arrays: `weights[i]` and `values[i]` are arrays, `capacity[i]` scalar.
2) 2D arrays: `weights` and `values` are shape `[N, n_items]`, `capacity` shape `[N]`.
