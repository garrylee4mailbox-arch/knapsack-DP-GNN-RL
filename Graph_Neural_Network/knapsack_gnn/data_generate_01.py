import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .dp import solve_knapsack_dp


DEFAULT_SEED = 2025
DEFAULT_NUM_INSTANCES = 1000
DEFAULT_N_RANGE = (80, 200)
DEFAULT_CAPACITY_RANGE = (200, 800)
DEFAULT_WEIGHT_RANGE = (1, 100)
DEFAULT_VALUE_MULT_RANGE = (0.8, 1.3)
DEFAULT_MAX_STATES = 5_000_000
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "dataset" / "knapsack01_medium"


def set_seeds(seed: int) -> np.random.Generator:
    """Seed both numpy and random, returning a numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sample_instance_sizes(
    rng: np.random.Generator,
    n_range: Tuple[int, int],
    cap_range: Tuple[int, int],
    max_states: int,
    max_attempts: int = 1000,
) -> Tuple[int, int]:
    """Sample (n_items, capacity) while keeping DP state count feasible."""
    for _ in range(max_attempts):
        n_items = int(rng.integers(n_range[0], n_range[1] + 1))
        capacity = int(rng.integers(cap_range[0], cap_range[1] + 1))
        if n_items * capacity <= max_states:
            return n_items, capacity
    raise ValueError(
        f"Failed to sample feasible n_items/capacity after {max_attempts} attempts "
        f"with max_states={max_states}"
    )


def generate_weights_values(
    rng: np.random.Generator, n_items: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate weights and correlated values."""
    weights = rng.integers(
        DEFAULT_WEIGHT_RANGE[0],
        DEFAULT_WEIGHT_RANGE[1] + 1,
        size=n_items,
        dtype=np.int32,
    )
    multipliers = rng.uniform(
        DEFAULT_VALUE_MULT_RANGE[0], DEFAULT_VALUE_MULT_RANGE[1], size=n_items
    )
    values = np.maximum((weights * multipliers).astype(np.int32), 1)
    return weights, values


def solve_instance(weights: np.ndarray, values: np.ndarray, capacity: int) -> Dict[str, np.ndarray]:
    """Solve a single instance using DP and return solution artifacts."""
    solution_list = solve_knapsack_dp(weights.tolist(), values.tolist(), capacity)
    solution = np.asarray(solution_list, dtype=np.int8)
    total_weight = int(np.sum(weights * solution))
    if total_weight > capacity:
        raise ValueError("DP solution exceeds capacity")
    dp_value = int(np.sum(values * solution))
    return {
        "solution": solution,
        "dp_value": np.int32(dp_value),
        "total_weight": total_weight,
    }


def save_instance(
    out_dir: Path,
    idx: int,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
    solution: np.ndarray,
    dp_value: np.int32,
) -> None:
    """Persist a single instance to compressed NPZ."""
    out_path = out_dir / f"instance_{idx:04d}.npz"
    np.savez_compressed(
        out_path,
        weights=weights.astype(np.int32),
        values=values.astype(np.int32),
        capacity=np.int32(capacity),
        solution=solution.astype(np.int8),
        dp_value=dp_value,
    )


def write_meta(out_dir: Path, meta: Dict) -> None:
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate_dataset(
    out_dir: Path,
    num_instances: int,
    n_range: Tuple[int, int],
    cap_range: Tuple[int, int],
    seed: int,
    max_states: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = set_seeds(seed)

    start_time = time.perf_counter()
    for idx in range(num_instances):
        n_items, capacity = sample_instance_sizes(rng, n_range, cap_range, max_states)
        weights, values = generate_weights_values(rng, n_items)
        solved = solve_instance(weights, values, capacity)

        save_instance(
            out_dir=out_dir,
            idx=idx,
            weights=weights,
            values=values,
            capacity=capacity,
            solution=solved["solution"],
            dp_value=solved["dp_value"],
        )

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else float("inf")
            remaining = (num_instances - idx - 1) / rate if rate > 0 else float("inf")
            print(
                f"[{idx + 1}/{num_instances}] elapsed: {elapsed:.1f}s | "
                f"ETA: {remaining:.1f}s | n={n_items}, cap={capacity}"
            )

    total_time = time.perf_counter() - start_time
    meta = {
        "seed": seed,
        "num_instances": num_instances,
        "n_range": n_range,
        "capacity_range": cap_range,
        "weight_range": DEFAULT_WEIGHT_RANGE,
        "value_multiplier_range": DEFAULT_VALUE_MULT_RANGE,
        "max_states": max_states,
        "generator": "knapsack_gnn.data_generate_01",
        "total_time_sec": round(total_time, 3),
    }
    write_meta(out_dir, meta)
    print(f"Finished generating {num_instances} instances in {total_time:.1f}s")
    print(f"Data saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate medium-size 0/1 knapsack dataset with DP labels."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated NPZ files and meta.json",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=DEFAULT_NUM_INSTANCES,
        help="Number of instances to generate",
    )
    parser.add_argument(
        "--n_min",
        type=int,
        default=DEFAULT_N_RANGE[0],
        help="Minimum number of items",
    )
    parser.add_argument(
        "--n_max",
        type=int,
        default=DEFAULT_N_RANGE[1],
        help="Maximum number of items",
    )
    parser.add_argument(
        "--cap_min",
        type=int,
        default=DEFAULT_CAPACITY_RANGE[0],
        help="Minimum knapsack capacity",
    )
    parser.add_argument(
        "--cap_max",
        type=int,
        default=DEFAULT_CAPACITY_RANGE[1],
        help="Maximum knapsack capacity",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for both numpy and random",
    )
    parser.add_argument(
        "--max_states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help="Maximum n_items * capacity allowed for DP feasibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_range = (args.n_min, args.n_max)
    cap_range = (args.cap_min, args.cap_max)
    if n_range[0] <= 0 or cap_range[0] <= 0:
        raise ValueError("n_min, cap_min must be positive")
    if n_range[0] > n_range[1]:
        raise ValueError("n_min cannot exceed n_max")
    if cap_range[0] > cap_range[1]:
        raise ValueError("cap_min cannot exceed cap_max")

    generate_dataset(
        out_dir=args.out_dir,
        num_instances=args.num_instances,
        n_range=n_range,
        cap_range=cap_range,
        seed=args.seed,
        max_states=args.max_states,
    )


if __name__ == "__main__":
    main()
