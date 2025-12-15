from typing import Tuple

import torch

from .dp import solve_knapsack_dp
from .graph_builder import build_knapsack_graph
from .io_excel import load_excel_knapsack_instances


@torch.no_grad()
def evaluate_gnn_vs_dp_on_excel(model, excel_path: str, threshold: float = 0.5) -> Tuple[int, int, int]:
    """Compare GNN predictions with DP on the Excel instances.

    Returns:
        (num_instances, matches_value, violations_capacity)
    """
    device = next(model.parameters()).device
    model.eval()

    instances = load_excel_knapsack_instances(excel_path)

    matches = 0
    violations = 0

    for idx, inst in enumerate(instances, 1):
        weights = inst["weights"]
        values = inst["values"]
        capacity = inst["capacity"]

        dp_solution = solve_knapsack_dp(weights, values, capacity)
        dp_value = sum(v * s for v, s in zip(values, dp_solution))
        dp_weight = sum(w * s for w, s in zip(weights, dp_solution))

        graph = build_knapsack_graph(weights, values, capacity, dp_solution).to(device)

        logits = model(graph)
        probs = torch.sigmoid(logits)
        gnn_solution = (probs >= threshold).cpu().long().tolist()

        gnn_value = sum(v * s for v, s in zip(values, gnn_solution))
        gnn_weight = sum(w * s for w, s in zip(weights, gnn_solution))
        is_feasible = gnn_weight <= capacity

        print(f"Instance {idx}:")
        print(f"  DP_value={dp_value}, GNN_value={gnn_value}")
        print(f"  capacity={capacity}, total_weight_DP={dp_weight}, total_weight_GNN={gnn_weight}")
        print(f"  is_feasible_GNN={is_feasible}")

        if gnn_value == dp_value:
            matches += 1
        if not is_feasible:
            violations += 1

    print("\nSummary:")
    print(f"  Instances matching DP value: {matches}/{len(instances)}")
    print(f"  Instances violating capacity: {violations}/{len(instances)}")

    return len(instances), matches, violations
