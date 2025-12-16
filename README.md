# Knapsack Problem: DP vs GNN vs Reinforcement Learning (DQN)

This repository presents a systematic comparison of three representative approaches for solving the **0/1 Knapsack Problem**:

- **Dynamic Programming (DP)** – exact algorithm and ground-truth oracle  
- **Graph Neural Network (GNN)** – supervised learning with structural reasoning  
- **Reinforcement Learning (DQN)** – sequential decision-making via Deep Q-Network  

The project was developed as part of **CPS 3440 (Algorithms)** and focuses on **fair comparison, reproducibility, and practical insights**, rather than solely pursuing best performance.

---

## 📌 Problem Overview

The **0/1 Knapsack Problem** is a classic NP-hard combinatorial optimization problem:

> Given a set of items, each with a weight and value, select a subset of items such that the total weight does not exceed a capacity constraint while maximizing total value.

This project explores how **different computational paradigms**—exact algorithms, supervised learning, and reinforcement learning—perform on the same knapsack instances.

---

## 🧠 Methods Implemented

### 1. Dynamic Programming (DP)
- Classical exact solver for 0/1 knapsack
- Time complexity: \(O(nW)\)
- Used as a **ground-truth oracle**
- Provides optimal values and item selections for evaluation and supervision

📂 Location:
```
Dynamic_Programming/
tools/dp_baseline_eval.py
```

---

### 2. Graph Neural Network (GNN)
- Supervised learning approach using **DP-optimal solutions as labels**
- Each knapsack instance is modeled as a graph:
  - Nodes = items
  - Node features = weight, value, value density, normalized features
- Uses message passing to capture global interactions
- Inference uses greedy decoding to ensure feasibility

📂 Location:
```
Graph_Neural_Network/
├── run_train.py
├── evaluate_gnn.py
└── gnn.pt
```

---

### 3. Reinforcement Learning (Deep Q-Network, DQN)
- Knapsack formulated as a **sequential decision-making problem**
- At each step, the agent decides to **select or skip** the current item
- Uses:
  - Experience replay
  - Target network
  - ε-greedy exploration
- Tabular Q-learning was initially explored but found impractical due to large state space and lack of generalization
- Final approach uses an MLP-based DQN

📂 Location:
```
Reinforcement_Learning/dqn_knapsack_project/
├── train_dqn.py
├── evaluate_dqn.py
├── model.py
└── train_meta.json
```

⚠️ **Note on training budget**  
Due to hardware constraints, DQN was trained for **50k steps** (reduced from the originally planned 200k steps).  
This choice was made to maintain a **reasonable and fair comparison** with GNN training cost under the same environment.

---

## 📊 Dataset

- 1000 medium-difficulty 0/1 knapsack instances
- Stored as `.npz` files
- Each instance contains:
  - `weights`: 1D array
  - `values`: 1D array
  - `capacity`: scalar
- Dataset is **read-only** and shared by all methods

📂 Location:
```
dataset/knapsack01_medium/
```

---

## 🧪 Experimental Pipeline

All methods follow a **unified evaluation protocol**:

1. Load the same dataset
2. DP computes optimal solutions (oracle)
3. GNN and DQN are evaluated on the same instances
4. Results are merged and summarized
5. Metrics are computed consistently across methods

📂 Result files:
```
results/
├── DP/dp_results.csv
├── GNN/gnn_eval_results.csv
├── DQN/eval_results.csv
└── compare/
    ├── merged_results.csv
    └── summary.json
```

A utility script is provided to merge results:
```
tools/merge_results.py
```

---

## 📈 Evaluation Metrics

- **Average Runtime (ms)**
- **Optimality Gap** (relative to DP)
- **Accuracy Rate**
- **Stability Metrics**:
  - Mean ± Standard Deviation
  - Median Gap
  - Interquartile Range (IQR)
  - 95th Percentile Gap (P95)

These metrics provide insight into **both average performance and robustness**.

---

## 🔍 Key Findings

- **DP** provides exact optimal solutions and serves as a reliable benchmark, but scales poorly with capacity.
- **GNN** achieves the best balance between speed, accuracy, and stability, closely approximating DP with fast inference.
- **DQN** shows higher variance and lower solution quality under the current setup, highlighting the challenges of sequential RL for combinatorial optimization.

---

## 📁 Repository Structure

```
knapsack-DP-GNN-DL/
├── Dynamic_Programming/
├── Graph_Neural_Network/
├── Reinforcement_Learning/
├── dataset/
├── results/
├── tools/
├── docs/
├── README.md
└── LICENSE
```

---

## 📎 Notes

- This project emphasizes **methodological clarity and fair comparison**, not leaderboard performance.
- All reported results correspond exactly to the experiments conducted and documented in the final report.
- The repository is intended for **educational and research purposes**.

---

## 👤 Authors

Course Project for **CPS 3440 – Algorithms**  
Wenzhou-Kean University  

Team Leader: Guanlin Li 1308245 (W07)  
Contributors: 
Chunguang Lu 1365419 (W08)
Xiaoqian Zhang 1365436 (W08)
Mingshi Cai 1365432 (W08)
Yiyue Yin 1235600 (W07)

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.