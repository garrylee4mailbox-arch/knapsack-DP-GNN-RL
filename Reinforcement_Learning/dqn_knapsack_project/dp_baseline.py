# Auto-generated DQN-for-0/1-Knapsack project skeleton
# Files are modular by design.

from __future__ import annotations
import numpy as np

def dp_opt_value(weights: np.ndarray, values: np.ndarray, capacity: int) -> int:
    """Classic 0/1 knapsack DP for optimal value.

    Complexity: O(n * capacity). Suitable for moderate capacities.
    If capacity is extremely large (e.g., > 20000) this may be slow.
    """
    w = np.asarray(weights).astype(np.int64)
    v = np.asarray(values).astype(np.int64)
    W = int(capacity)
    n = int(w.shape[0])

    dp = np.zeros(W + 1, dtype=np.int64)
    for i in range(n):
        wi = int(w[i]); vi = int(v[i])
        if wi > W:
            continue
        # iterate backwards for 0/1
        for c in range(W, wi - 1, -1):
            cand = dp[c - wi] + vi
            if cand > dp[c]:
                dp[c] = cand
    return int(dp[W])
