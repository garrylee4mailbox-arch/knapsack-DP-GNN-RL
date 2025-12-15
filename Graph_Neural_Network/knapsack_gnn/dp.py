from typing import List

def solve_knapsack_dp(weights: List[int], values: List[int], capacity: int) -> List[int]:
    """Classic 0/1 knapsack DP.

    Returns:
        A 0/1 list indicating whether each item is selected in an optimal solution.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    keep = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                cand = dp[i - 1][c - w] + v
                if cand > dp[i][c]:
                    dp[i][c] = cand
                    keep[i][c] = 1

    res = [0] * n
    c = capacity
    for i in range(n, 0, -1):
        if keep[i][c] == 1:
            res[i - 1] = 1
            c -= weights[i - 1]
    return res
