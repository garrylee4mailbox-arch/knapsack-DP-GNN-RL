# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class StepOutput:
    next_state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class KnapsackEnv:
    """Sequential knapsack environment for a single instance.

    At step i, you decide action a in {0,1} for item i.
    Feasibility is enforced by masking: if w[i] > cap, action=1 is invalid.
    """
    def __init__(self, weights: np.ndarray, values: np.ndarray, capacity: int, eps: float = 1e-8):
        self.w = np.asarray(weights).astype(np.float32)
        self.v = np.asarray(values).astype(np.float32)
        self.W = float(capacity)
        self.eps = eps

        self.n = int(self.w.shape[0])
        self.reset()

    def reset(self):
        self.i = 0
        self.cap = float(self.W)
        self.total_value = 0.0
        self.total_weight = 0.0
        self.selection = np.zeros(self.n, dtype=np.int64)
        return self._state()

    def valid_actions_mask(self) -> np.ndarray:
        # mask over actions [a=0, a=1]
        if self.i >= self.n:
            return np.array([1, 0], dtype=np.int64)
        can_take = (self.w[self.i] <= self.cap + self.eps)
        return np.array([1, 1 if can_take else 0], dtype=np.int64)

    def step(self, action: int) -> StepOutput:
        if self.i >= self.n:
            # already done
            return StepOutput(self._state(), 0.0, True, {"terminal": True})

        mask = self.valid_actions_mask()
        # enforce feasibility: if invalid, force a=0
        if action == 1 and mask[1] == 0:
            action = 0

        reward = 0.0
        if action == 1:
            self.cap -= float(self.w[self.i])
            self.total_weight += float(self.w[self.i])
            self.total_value += float(self.v[self.i])
            self.selection[self.i] = 1
            reward = float(self.v[self.i])

        self.i += 1
        done = (self.i >= self.n)
        return StepOutput(self._state(), reward, done, {
            "i": self.i,
            "cap": self.cap,
            "total_value": self.total_value,
            "total_weight": self.total_weight,
        })

    def _state(self) -> np.ndarray:
        """Generalizable feature vector.

        Includes:
        - current item normalized weight/value/ratio (0 if terminal)
        - remaining capacity normalized
        - progress i/n
        - summary stats for remaining items (mean/std/max of w,v,ratio)
        """
        # Normalize denominators
        w_max = float(np.max(self.w)) if self.n > 0 else 1.0
        v_max = float(np.max(self.v)) if self.n > 0 else 1.0

        cap_norm = self.cap / (self.W + self.eps)
        i_norm = (float(self.i) / float(self.n)) if self.n > 0 else 0.0

        if self.i >= self.n:
            cur = np.zeros(3, dtype=np.float32)
            rem_stats = np.zeros(9, dtype=np.float32)
        else:
            wi = self.w[self.i]
            vi = self.v[self.i]
            ratio = vi / (wi + self.eps)
            cur = np.array([
                wi / (w_max + self.eps),
                vi / (v_max + self.eps),
                ratio / (v_max / (w_max + self.eps) + self.eps),
            ], dtype=np.float32)

            w_rem = self.w[self.i:]
            v_rem = self.v[self.i:]
            r_rem = v_rem / (w_rem + self.eps)

            def stats(x):
                return (float(np.mean(x)), float(np.std(x)), float(np.max(x)))

            wm, ws, wx = stats(w_rem / (w_max + self.eps))
            vm, vs, vx = stats(v_rem / (v_max + self.eps))
            rm, rs, rx = stats(r_rem / (v_max / (w_max + self.eps) + self.eps))
            rem_stats = np.array([wm, ws, wx, vm, vs, vx, rm, rs, rx], dtype=np.float32)

        state = np.concatenate([cur, np.array([cap_norm, i_norm], dtype=np.float32), rem_stats], axis=0)
        return state.astype(np.float32)

    def compute_solution_value(self) -> float:
        return float(self.total_value)

    def compute_solution_weight(self) -> float:
        return float(self.total_weight)
