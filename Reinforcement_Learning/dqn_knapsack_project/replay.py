# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import random

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool
    mask2: np.ndarray  # valid action mask at next state

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def push(self, t: Transition):
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.data, k=batch_size)
        # stack arrays
        s = np.stack([b.s for b in batch], axis=0)
        a = np.asarray([b.a for b in batch], dtype=np.int64)
        r = np.asarray([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch], axis=0)
        done = np.asarray([b.done for b in batch], dtype=np.float32)
        mask2 = np.stack([b.mask2 for b in batch], axis=0).astype(np.float32)
        return s, a, r, s2, done, mask2
