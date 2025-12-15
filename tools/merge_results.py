# Merge DP, GNN, and DQN per-instance results into a comparison table.
# Run directly (no CLI args). Paths are resolved relative to the repo root.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def mark(msg: str):
    print(f"[MERGE] {msg}", flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_results(path: Path, method: str) -> Dict[str, Dict[str, Any]]:
    """Load a CSV into a dict keyed by instance_file."""
    if not path.exists():
        raise FileNotFoundError(f"{method} results file not found: {path}")
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("instance_file")
            if not key:
                continue
            rows[key] = row
    return rows


def parse_float(row: Dict[str, Any], keys) -> float | None:
    for k in keys:
        if k in row and row[k] not in (None, "", "nan"):
            try:
                return float(row[k])
            except ValueError:
                continue
    return None


def parse_int(row: Dict[str, Any], keys) -> int | None:
    for k in keys:
        if k in row and row[k] not in (None, "", "nan"):
            try:
                return int(float(row[k]))
            except ValueError:
                continue
    return None


def compute_gap(dp_val: float | None, other_val: float | None) -> float | None:
    if dp_val is None or other_val is None or dp_val == 0:
        return None
    return (dp_val - other_val) / dp_val


def main():
    root = repo_root()
    dp_csv = root / "results" / "DP" / "dp_results.csv"
    gnn_csv = root / "results" / "GNN" / "gnn_eval_results.csv"
    dqn_csv = root / "results" / "DQN" / "eval_results.csv"
    out_dir = root / "results" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = out_dir / "merged_results.csv"
    summary_json = out_dir / "summary.json"

    mark(f"DP   file: {dp_csv}")
    mark(f"GNN  file: {gnn_csv}")
    mark(f"DQN  file: {dqn_csv}")
    dp_rows = load_results(dp_csv, "DP")
    gnn_rows = load_results(gnn_csv, "GNN")
    dqn_rows = load_results(dqn_csv, "DQN")

    common_keys = sorted(set(dp_rows) & set(gnn_rows) & set(dqn_rows))
    if not common_keys:
        raise RuntimeError("No overlapping instance_file keys across DP/GNN/DQN.")
    mark(f"Found {len(common_keys)} overlapping instances")

    merged_records = []
    stats = {
        "gnn": {"feasible": 0, "count": 0, "time_ms": []},
        "dqn": {"feasible": 0, "count": 0, "time_ms": []},
        "dp": {"feasible": 0, "count": 0, "time_ms": []},
        "gaps": {"gnn": [], "dqn": []},
    }

    for key in common_keys:
        dp = dp_rows[key]
        gnn = gnn_rows[key]
        dqn = dqn_rows[key]

        n_items = parse_int(dp, ["n_items"]) or parse_int(gnn, ["n_items"]) or parse_int(dqn, ["n_items"])
        capacity = parse_float(dp, ["capacity"]) or parse_float(gnn, ["capacity"]) or parse_float(dqn, ["capacity"])

        dp_value = parse_float(dp, ["total_value", "dp_value", "value"])
        dp_weight = parse_float(dp, ["total_weight", "dp_weight", "weight"])
        dp_feasible = int(float(dp.get("feasible", 1))) if "feasible" in dp else 1
        dp_time_ms = parse_float(dp, ["inference_time_ms", "time_ms", "solve_time_ms"])

        gnn_value = parse_float(gnn, ["total_value", "gnn_value", "value"])
        gnn_weight = parse_float(gnn, ["total_weight", "gnn_weight", "weight"])
        gnn_feasible = int(float(gnn.get("feasible", 0)))
        gnn_time_ms = parse_float(gnn, ["inference_time_ms", "time_ms", "solve_time_ms"])

        dqn_value = parse_float(dqn, ["total_value_selected", "total_value", "value"])
        dqn_weight = parse_float(dqn, ["total_weight_selected", "total_weight", "weight"])
        dqn_feasible = int(float(dqn.get("feasible", 0)))
        dqn_time_ms = parse_float(dqn, ["inference_time_ms", "time_ms", "solve_time_ms"])

        gap_gnn = compute_gap(dp_value, gnn_value)
        gap_dqn = compute_gap(dp_value, dqn_value)

        stats["gnn"]["feasible"] += 1 if gnn_feasible else 0
        stats["dqn"]["feasible"] += 1 if dqn_feasible else 0
        stats["dp"]["feasible"] += 1 if dp_feasible else 0
        stats["gnn"]["count"] += 1
        stats["dqn"]["count"] += 1
        stats["dp"]["count"] += 1
        if gnn_time_ms is not None:
            stats["gnn"]["time_ms"].append(gnn_time_ms)
        if dqn_time_ms is not None:
            stats["dqn"]["time_ms"].append(dqn_time_ms)
        if dp_time_ms is not None:
            stats["dp"]["time_ms"].append(dp_time_ms)
        if gap_gnn is not None:
            stats["gaps"]["gnn"].append(gap_gnn)
        if gap_dqn is not None:
            stats["gaps"]["dqn"].append(gap_dqn)

        merged_records.append(
            {
                "instance_file": key,
                "n_items": n_items,
                "capacity": capacity,
                "dp_value": dp_value,
                "dp_weight": dp_weight,
                "dp_feasible": dp_feasible,
                "dp_time_ms": dp_time_ms,
                "gnn_value": gnn_value,
                "gnn_weight": gnn_weight,
                "gnn_feasible": gnn_feasible,
                "gnn_time_ms": gnn_time_ms,
                "gap_gnn": gap_gnn,
                "dqn_value": dqn_value,
                "dqn_weight": dqn_weight,
                "dqn_feasible": dqn_feasible,
                "dqn_time_ms": dqn_time_ms,
                "gap_dqn": gap_dqn,
                "dp_selected_items": dp.get("selected_items") if dp else None,
                "gnn_selected_items": gnn.get("selected_items") if gnn else None,
                "dqn_selected_items": dqn.get("selected_items") if dqn else None,
            }
        )

    fieldnames = [
        "instance_file",
        "n_items",
        "capacity",
        "dp_value",
        "dp_weight",
        "dp_feasible",
        "dp_time_ms",
        "gnn_value",
        "gnn_weight",
        "gnn_feasible",
        "gnn_time_ms",
        "gap_gnn",
        "dqn_value",
        "dqn_weight",
        "dqn_feasible",
        "dqn_time_ms",
        "gap_dqn",
        "dp_selected_items",
        "gnn_selected_items",
        "dqn_selected_items",
    ]

    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_records)
    mark(f"Wrote merged CSV to {merged_csv}")

    summary = {
        "counts": {k: v["count"] for k, v in stats.items() if k != "gaps"},
        "feasible_rate": {
            "dp": stats["dp"]["feasible"] / stats["dp"]["count"] if stats["dp"]["count"] else None,
            "gnn": stats["gnn"]["feasible"] / stats["gnn"]["count"] if stats["gnn"]["count"] else None,
            "dqn": stats["dqn"]["feasible"] / stats["dqn"]["count"] if stats["dqn"]["count"] else None,
        },
        "avg_time_ms": {
            "dp": float(sum(stats["dp"]["time_ms"]) / len(stats["dp"]["time_ms"])) if stats["dp"]["time_ms"] else None,
            "gnn": float(sum(stats["gnn"]["time_ms"]) / len(stats["gnn"]["time_ms"])) if stats["gnn"]["time_ms"] else None,
            "dqn": float(sum(stats["dqn"]["time_ms"]) / len(stats["dqn"]["time_ms"])) if stats["dqn"]["time_ms"] else None,
        },
        "avg_gap": {
            "gnn": float(sum(stats["gaps"]["gnn"]) / len(stats["gaps"]["gnn"])) if stats["gaps"]["gnn"] else None,
            "dqn": float(sum(stats["gaps"]["dqn"]) / len(stats["gaps"]["dqn"])) if stats["gaps"]["dqn"] else None,
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    mark(f"Wrote summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
