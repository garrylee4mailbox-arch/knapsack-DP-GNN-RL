# Knapsack Optimization Project (DP / GNN / DL)
Wenzhou-Kean University · CPS Project · 2025
- [Knapsack Optimization Project (DP / GNN / DL)](#knapsack-optimization-project-dp--gnn--dl)
  - [0. Documentation and Meetings](#0-documentation-and-meetings)
  - [📄 会前必读文件](#-会前必读文件)
  - [1. Project Overview](#1-project-overview)
  - [2. Team Division](#2-team-division)
  - [3. Problem Definition](#3-problem-definition)
  - [4. Folder Structure](#4-folder-structure)
  - [5. Dataset Standardization](#5-dataset-standardization)
  - [6. Evaluation Metrics](#6-evaluation-metrics)
  - [7. Technical Routes](#7-technical-routes)
    - [7.1 Dynamic Programming (DP)](#71-dynamic-programming-dp)
    - [7.2 Graph Neural Network (GNN)](#72-graph-neural-network-gnn)
    - [7.3 Deep Learning (DL)](#73-deep-learning-dl)
  - [8. Experiments](#8-experiments)
  - [9. Documentation and Meetings](#9-documentation-and-meetings)
  - [10. Timeline](#10-timeline)
    - [Week 1](#week-1)
    - [Week 2](#week-2)
    - [Week 3](#week-3)
    - [Week 4](#week-4)

[toc]

---

## 0. Documentation and Meetings

All meeting notes and technical discussions will be stored in `docs/`.

## 📄 会前必读文件
<!-- update for sync -->
- [01｜会议规则说明](docs/3440会议规则.docx)
- [02｜Git 分支结构说明](docs/Git分支结构说明.docx)
- [03｜Git Hub Vscode 使用说明](docs/GitHub_Vscode使用说明.docx)


---

## 1. Project Overview
This project compares three different methods to solve the **0/1 Knapsack Problem**:

- Dynamic Programming (DP) — optimal baseline  
- Graph Neural Network (GNN) — graph-based learning method  
- Deep Learning (DL) — item-wise prediction model (e.g., MLP / Transformer)

**Deliverables:**

- Source code for DP, GNN, and DL  
- Unified dataset and evaluation metrics  
- Experiment results and comparison  
- Final report (paper)

---

## 2. Team Division

| Member   | Responsibility                    |
|----------|-----------------------------------|
| Member A | DP coding + experiments           |
| Member B | DP coding + experiments           |
| Garry    | GNN model + experiments           |
| Member C | DL model (MLP/Transformer)        |
| Member D | DL model (sequence / PointerNet)  |

(可以之后把名字改成真实英文名或学号。)

---

## 3. Problem Definition

We solve the **0/1 Knapsack Problem**:

Given:

- `weights = [w1, w2, ..., wn]`
- `values  = [v1, v2, ..., vn]`
- `capacity = C`

Maximize:
\[
\sum_{i=1}^{n} v[i] \cdot x[i]
\]

Subject to:
\[
\sum_{i=1}^{n} w[i] \cdot x[i] \le C,\quad x[i] \in \{0,1\}
\]

All three methods (DP / GNN / DL) must solve **exactly the same formulation**.

---

## 4. Folder Structure

```text
project_root/
│
├── dp/
│   ├── dp_solver.py          # or .java
│   ├── dp_experiment.ipynb   # optional
│
├── gnn/
│   ├── gnn_model.py
│   ├── gnn_train.py
│
├── dl/
│   ├── dl_model.py
│   ├── dl_train.py
│
├── dataset/
│   ├── knapsack_50.csv
│   ├── knapsack_100.csv
│   ├── knapsack_200.csv
│
├── results/
│   ├── dp_results.csv
│   ├── gnn_results.csv
│   ├── dl_results.csv
│
├── docs/
│   ├── meeting_01.md
│   ├── technical_notes.md
│
└── README.md
```

---

## 5. Dataset Standardization

We use the public 0/1 knapsack instances from:

- https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/

**Rules:**

- All methods (DP / GNN / DL) must use the **same dataset files**.
- Convert or store them to a unified CSV format, for example:

```text
item_id,weight,value
1,2,3
2,4,6
3,5,7
...
```

Example Python loading snippet (optional):

```python
import pandas as pd

df = pd.read_csv("dataset/knapsack_50.csv")
weights = df["weight"].tolist()
values = df["value"].tolist()
capacity = 100  # set according to the instance definition
```

---

## 6. Evaluation Metrics

All methods must output results in a comparable way.

For each instance, each method should output:

- `solution_vector`: a list of 0/1 (length = n), e.g. `[1,0,1,0,...]`
- `total_value`: sum of values of selected items
- `total_weight`: sum of weights of selected items
- `feasible`: whether `total_weight <= capacity`
- `runtime`: wall-clock time (in seconds or milliseconds)

These metrics will be saved (for example) in `results/*.csv` and used for experiment tables in the paper.

---

## 7. Technical Routes

### 7.1 Dynamic Programming (DP)

- State definition:  
  `dp[i][w]` = maximum value using the first `i` items with capacity `w`.

- Transition:

  - Not taking item `i`:  
    `dp[i][w] = dp[i-1][w]`
  - Taking item `i` (if `w >= weight[i]`):  
    `dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])`

- After filling the DP table, use backtracking to recover a **0/1 solution vector**.

- Output format:

```text
method=DP,
solution=[1,0,1,0,...],
total_value=XX,
total_weight=YY,
feasible=true/false,
runtime=...
```

---

### 7.2 Graph Neural Network (GNN)

- Build a graph where each item is a node. For simplicity, start with a **fully connected graph**.
- Node features example:

```text
[weight, value, value/weight]
```

- Use a small GNN (e.g., 2-layer GCN or GraphSAGE):
  - Message passing over nodes
  - Non-linear transformations (ReLU + Linear)
  - Final layer outputs a score for each item

- Convert score to decision:
  - Apply sigmoid
  - Threshold (score > 0.5 → 1)

- Training:
  - Use DP optimal solution as label
  - Loss: binary cross-entropy

- Output: same structure as DP

---

### 7.3 Deep Learning (DL)

- Input: item feature matrix `(n_items, feature_dim)`  
  Example: `[weight, value, value/weight]`

- Model: MLP or Transformer
- Output: probability per item
- Decision: threshold → 0/1
- Loss: binary cross-entropy
- Output: same structure as DP and GNN

---

## 8. Experiments

We compare methods on different problem sizes (e.g., `n = 50, 100, 200`).

**Metrics:**

- `total_value`  
- `feasible`  
- `runtime`  
- `approx_ratio = total_value / dp_optimal_value`

Example results format:

```text
method,instance,n,total_value,total_weight,feasible,runtime,approx_ratio
DP,knapsack_50,50,1234,98,True,0.001,1.0
GNN,knapsack_50,50,1200,100,True,0.01,0.97
DL,knapsack_50,50,1180,95,True,0.008,0.95
```

---

## 9. Documentation and Meetings

All meeting notes and technical discussions will be stored in `docs/`.

Example:

- `docs/meeting_01.md`  
- `docs/technical_notes.md`

---

## 10. Timeline

### Week 1
- Learn basics of DP / GNN / DL
- Finalize README
- Download dataset

### Week 2
- Implement DP baseline
- Implement GNN & DL skeletons

### Week 3
- Run experiments
- Save results in `/results`

### Week 4
- Analyze results & write paper
- Prepare presentation
```}
