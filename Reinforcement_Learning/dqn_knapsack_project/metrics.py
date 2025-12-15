# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import time

@dataclass
class EvalRow:
    instance_id: int
    n_items: int
    capacity: int
    dp_opt_value: int
    dqn_value: float
    optimality_ratio: float
    weight_used: float
    feasible: bool
    inference_time_sec: float

def is_feasible(weights: np.ndarray, selection: np.ndarray, capacity: int) -> bool:
    tot_w = float(np.sum(np.asarray(weights) * np.asarray(selection)))
    return tot_w <= float(capacity) + 1e-6
