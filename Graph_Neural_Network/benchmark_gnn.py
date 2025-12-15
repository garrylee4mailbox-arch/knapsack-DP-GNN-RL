import csv
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _greedy_feasible_decode(probs: torch.Tensor, weights: torch.Tensor, capacity: float) -> torch.Tensor:
    """Greedy decode: pick items by descending prob while staying within capacity."""
    idx = torch.argsort(probs, descending=True)
    x_hat = torch.zeros_like(probs)
    total_w = 0.0
    for i in idx:
        w_i = weights[i].item()
        if total_w + w_i <= capacity:
            x_hat[i] = 1.0
            total_w += w_i
    return x_hat


def _ensure_attrs(batch):
    for attr in ("wts", "vals", "cap"):
        if not hasattr(batch, attr):
            raise AttributeError(f"Expected batch to have attribute '{attr}' for benchmarking.")


@torch.no_grad()
def run_gnn_benchmark(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    out_dir: str,
    n_instances: int = 100,
    seed: int = 2025,
):
    """
    Run GNN inference benchmark on the first n_instances from loader.

    Records per-instance metrics and aggregate statistics to disk.
    """
    _set_seed(seed)
    model.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    details_latest_path = out_path / "gnn_benchmark_details_latest.csv"
    summary_latest_path = out_path / "gnn_benchmark_summary_latest.json"
    details_all_path = out_path / "gnn_benchmark_details_all_runs.csv"
    summary_all_path = out_path / "gnn_benchmark_summary_all_runs.csv"

    records: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    seen = 0
    checked_attrs = False

    for batch in loader:
        if not checked_attrs:
            _ensure_attrs(batch)
            checked_attrs = True
        batch = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1

        fwd_start = time.perf_counter()
        logits = model(batch)
        probs = torch.sigmoid(logits)
        fwd_time = time.perf_counter() - fwd_start

        for g in range(num_graphs):
            if seen >= n_instances:
                break
            mask = batch_vec == g
            if mask.sum() == 0:
                continue

            dec_start = time.perf_counter()
            p_g = probs[mask].detach().cpu()
            w_g = batch.wts[mask].detach().cpu()
            v_g = batch.vals[mask].detach().cpu()
            cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())

            x_hat = _greedy_feasible_decode(p_g, w_g, cap_g)
            dec_time = time.perf_counter() - dec_start

            gnn_weight = float((x_hat * w_g).sum().item())
            gnn_value = float((x_hat * v_g).sum().item())
            feasible = gnn_weight <= cap_g + 1e-6
            fill_ratio = gnn_weight / cap_g if cap_g > 0 else 0.0
            selected_k = int(x_hat.sum().item())

            per_graph_time = (fwd_time / num_graphs) + dec_time

            records.append(
                {
                    "case_id": seen + 1,
                    "capacity": float(cap_g),
                    "gnn_value": gnn_value,
                    "gnn_weight": gnn_weight,
                    "feasible": feasible,
                    "fill_ratio": fill_ratio,
                    "selected_k": selected_k,
                    "solve_time_sec": per_graph_time,
                }
            )
            seen += 1
        if seen >= n_instances:
            break

    total_time = time.perf_counter() - total_start
    if seen == 0:
        raise RuntimeError("No instances processed in benchmark.")

    feasible_rate = sum(1 for r in records if r["feasible"]) / seen
    gnn_values = np.array([r["gnn_value"] for r in records], dtype=float)
    fill_ratios = np.array([r["fill_ratio"] for r in records], dtype=float)
    solve_times = np.array([r["solve_time_sec"] for r in records], dtype=float)
    avg_selected_k = float(np.mean([r["selected_k"] for r in records]))

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "device": str(device),
        "n_instances": seen,
        "seed": seed,
        "metrics": {
            "feasible_rate": feasible_rate,
            "avg_gnn_value": float(np.mean(gnn_values)),
            "std_gnn_value": float(np.std(gnn_values)),
            "avg_fill_ratio": float(np.mean(fill_ratios)),
            "std_fill_ratio": float(np.std(fill_ratios)),
            "avg_selected_k": avg_selected_k,
            "avg_solve_time_sec": float(np.mean(solve_times)),
            "p95_solve_time_sec": float(np.percentile(solve_times, 95)),
            "total_time_sec": total_time,
        },
    }

    details_fieldnames = [
        "run_id",
        "timestamp",
        "case_id",
        "capacity",
        "gnn_value",
        "gnn_weight",
        "feasible",
        "fill_ratio",
        "selected_k",
        "solve_time_sec",
    ]

    # Write latest details (overwrite). If locked, warn and skip.
    def _write_latest_details(path: Path) -> Optional[Path]:
        try:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=details_fieldnames)
                writer.writeheader()
                for row in records:
                    writer.writerow({"run_id": run_id, "timestamp": timestamp, **row})
            return path
        except PermissionError:
            print(f"Warning: Could not overwrite details latest at {path} (file locked). Skipping latest update.")
            return None

    # Write latest summary JSON. If locked, warn and skip.
    def _write_latest_summary(path: Path) -> Optional[Path]:
        try:
            with path.open("w") as f:
                json.dump(summary, f, indent=2)
            return path
        except PermissionError:
            print(f"Warning: Could not overwrite summary latest at {path} (file locked). Skipping latest update.")
            return None

    latest_details_saved = _write_latest_details(details_latest_path)
    latest_summary_saved = _write_latest_summary(summary_latest_path)

    # Append to cumulative details
    details_exists = details_all_path.exists()
    with details_all_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=details_fieldnames,
        )
        if not details_exists:
            writer.writeheader()
        for row in records:
            writer.writerow({"run_id": run_id, "timestamp": timestamp, **row})

    # Append to cumulative summary CSV
    summary_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "device": summary["device"],
        "n_instances": summary["n_instances"],
        "seed": summary["seed"],
        **summary["metrics"],
    }
    summary_fieldnames = list(summary_row.keys())
    summary_exists = summary_all_path.exists()
    with summary_all_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        if not summary_exists:
            writer.writeheader()
        writer.writerow(summary_row)

    m = summary["metrics"]
    print(
        "GNN benchmark | "
        f"N={seen} | feasible_rate={m['feasible_rate']:.3f} | "
        f"avg_value={m['avg_gnn_value']:.2f} | avg_fill={m['avg_fill_ratio']:.3f} | "
        f"avg_time={m['avg_solve_time_sec']:.4f} | p95_time={m['p95_solve_time_sec']:.4f} | "
        f"total_time={m['total_time_sec']:.2f}"
    )

    print("Benchmark files saved:")
    print(f"- details_latest: {latest_details_saved if latest_details_saved else 'skipped (locked)'}")
    print(f"- summary_latest: {latest_summary_saved if latest_summary_saved else 'skipped (locked)'}")
    print(f"- details_all_runs: {details_all_path}")
    print(f"- summary_all_runs: {summary_all_path}")

    return {
        "details_latest": str(latest_details_saved) if latest_details_saved else None,
        "summary_latest": str(latest_summary_saved) if latest_summary_saved else None,
        "details_all_runs": str(details_all_path),
        "summary_all_runs": str(summary_all_path),
    }
