# 0–1 Knapsack Problem Using Traditional Two-Dimensional Dynamic Programming

## 1. Problem Description

The **0–1 Knapsack Problem** is a classical optimization problem in dynamic programming.

Given:
- A set of `n` items
- Each item can be **either selected or not selected** (hence the name *0–1*)
- Each item `i` has:
  - Weight `weight[i]`
  - Value `value[i]`
- A knapsack with maximum capacity `W`

**Objective:**
Select a subset of items such that the total weight does not exceed `W`, while the total value is maximized.

---

## 2. Motivation for Dynamic Programming

The 0–1 knapsack problem exhibits two important properties:

1. **Optimal Substructure**  
   The optimal solution for the first `i` items depends on optimal solutions of the first `i−1` items.

2. **Overlapping Subproblems**  
   The same subproblems `(i, w)` are repeatedly evaluated during the decision process.

Because of these characteristics, dynamic programming is an efficient and appropriate approach.

---

## 3. Two-Dimensional DP State Definition

A two-dimensional DP table is defined as:

```
dp[i][w]
```

**Meaning:**  
The maximum total value achievable using the **first `i` items** with a knapsack capacity of **`w`**.

Where:
- `i = 0, 1, ..., n`
- `w = 0, 1, ..., W`

---

## 4. State Transition Equation

For item `i` (indexed as `i−1` in arrays), there are two possible choices:

### Case 1: Do not select item `i`

```
dp[i][w] = dp[i-1][w]
```

### Case 2: Select item `i` (only if `weight[i-1] ≤ w`)

```
dp[i][w] = dp[i-1][w - weight[i-1]] + value[i-1]
```

### Final Transition Formula

```
dp[i][w] = max(
    dp[i-1][w],
    dp[i-1][w - weight[i-1]] + value[i-1]
)
```

---

## 5. Initialization

The DP table is initialized as follows:

- `dp[0][w] = 0` for all `w`  
  (No items yield zero value regardless of capacity)

- `dp[i][0] = 0` for all `i`  
  (Zero capacity yields zero value)

---

## 6. Time and Space Complexity

- **Time Complexity:** `O(nW)`
- **Space Complexity:** `O(nW)`

This confirms that the solution follows the traditional two-dimensional dynamic programming approach.

---

## 7. Algorithm Implementation Overview

The function `knapsack_01_with_path` implements the above DP formulation and additionally recovers the selected items.

### 7.1 DP Table Construction

```python
dp = [[0] * (capacity + 1) for _ in range(n + 1)]
```

- Rows represent the number of items considered
- Columns represent knapsack capacity

The table directly corresponds to the defined state `dp[i][w]`.

---

### 7.2 Filling the DP Table

```python
for i in range(1, n + 1):
    wi = weights[i - 1]
    vi = values[i - 1]
    for w in range(capacity + 1):
        dp[i][w] = dp[i - 1][w]
        if wi <= w:
            dp[i][w] = max(dp[i][w], dp[i - 1][w - wi] + vi)
```

Explanation:
- The outer loop iterates through items
- The inner loop iterates through all capacities
- Each state considers both selecting and not selecting the current item

---

## 8. Recovering the Selected Items (Backtracking)

After the DP table is completed, the maximum value is stored at:

```
dp[n][capacity]
```

To determine which items are selected, backtracking is performed.

### Backtracking Logic

```python
selected = [0] * n
w = capacity

for i in range(n, 0, -1):
    if dp[i][w] != dp[i - 1][w]:
        selected[i - 1] = 1
        w -= weights[i - 1]
```

If the value at `dp[i][w]` differs from `dp[i-1][w]`, item `i` must have been selected.
The remaining capacity is updated accordingly.

---

## 9. Output Format

The algorithm returns:

- The maximum achievable value
- A binary list indicating item selection

Example:
```
Selected = [1, 0, 1, 1]
```

This means items 1, 3, and 4 are included in the optimal solution.

---

## 10. Experimental Design and Engineering Considerations

In the `main` function:

- Input data is read from Excel files
- Multiple test cases are processed sequentially
- Execution time is measured using `time.perf_counter()`
- Results (maximum value, runtime, and selection path) are exported to an Excel file

This design supports reproducible experiments and performance evaluation.

---

## 11. Conclusion

This implementation applies a **traditional two-dimensional dynamic programming approach** to solve the 0–1 knapsack problem.

Key advantages of the solution include:
- Clear state definition and transition logic
- Correct and efficient backtracking to recover selected items
- Practical support for batch testing and result export

Therefore, the solution is both **theoretically sound** and **practically applicable**, making it suitable for academic assignments and experimental reports.

