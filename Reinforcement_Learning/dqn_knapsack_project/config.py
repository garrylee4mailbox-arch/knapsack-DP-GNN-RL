# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from dataclasses import dataclass

@dataclass
class DQNConfig:
    # RL
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    min_buffer_size: int = 5_000
    target_update_steps: int = 2_000
    train_steps: int = 200_000
    warmup_steps: int = 2_000

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    # Optimization stability
    grad_clip_norm: float = 5.0

    # Dataset split
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # remaining is test

    # Feature normalization/eps
    eps: float = 1e-8
