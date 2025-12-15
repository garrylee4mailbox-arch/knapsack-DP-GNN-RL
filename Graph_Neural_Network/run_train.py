import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from benchmark_gnn import run_gnn_benchmark
from knapsack_gnn.dataset import GeneratedKnapsack01Dataset, KnapsackDataset
from knapsack_gnn.model import KnapsackGNN
from knapsack_gnn.train_eval import train_one_epoch, evaluate_node_accuracy
from knapsack_gnn.eval_compare import evaluate_gnn_vs_dp_on_excel


def greedy_feasible_decode(probs: torch.Tensor, weights: torch.Tensor, capacity: float) -> torch.Tensor:
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


@torch.no_grad()
def evaluate_instance_level(model, loader, num_cases: int = 10, device=None):
    model.eval()
    feasible_count = 0
    value_ratios = []
    optimal_matches = 0
    seen = 0

    for batch in loader:
        batch = batch.to(device) if device is not None else batch
        logits = model(batch)
        probs = torch.sigmoid(logits)

        if hasattr(batch, "batch"):
            num_graphs = batch.num_graphs
            batch_vec = batch.batch
        else:
            num_graphs = 1
            batch_vec = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)

        for g in range(num_graphs):
            if seen >= num_cases:
                break
            mask = batch_vec == g
            if mask.sum() == 0:
                continue
            p_g = probs[mask].detach().cpu()
            w_g = batch.wts[mask].detach().cpu()
            v_g = batch.vals[mask].detach().cpu()
            cap_g = batch.cap[g].item() if batch.cap.dim() > 0 else float(batch.cap.item())
            dp_sol = batch.y[mask].view(-1).detach().cpu()

            x_hat = greedy_feasible_decode(p_g, w_g, cap_g)

            gnn_weight = float((x_hat * w_g).sum().item())
            gnn_value = float((x_hat * v_g).sum().item())
            feasible = gnn_weight <= cap_g + 1e-6
            dp_weight = float((dp_sol * w_g).sum().item())
            dp_value = float((dp_sol * v_g).sum().item())
            ratio = gnn_value / dp_value if dp_value > 0 else 0.0

            feasible_count += 1 if feasible else 0
            value_ratios.append(ratio)
            optimal_matches += 1 if abs(gnn_value - dp_value) < 1e-6 else 0

            print(
                f"case {seen+1:03d} | cap={cap_g:.1f} | "
                f"DP: value={dp_value:.1f}, weight={dp_weight:.1f} | "
                f"GNN: value={gnn_value:.1f}, weight={gnn_weight:.1f}, feasible={feasible} | "
                f"ratio={ratio:.3f} | selected_k={int(x_hat.sum().item())}"
            )
            seen += 1
        if seen >= num_cases:
            break

    total = max(seen, 1)
    feasible_rate = feasible_count / total
    avg_value_ratio = sum(value_ratios) / total
    optimal_match_rate = optimal_matches / total
    print(
        f"Eval summary over {total} cases | "
        f"feasible_rate={feasible_rate:.3f} | "
        f"avg_value_ratio={avg_value_ratio:.3f} | "
        f"optimal_match_rate={optimal_match_rate:.3f}"
    )


def _default_generated_dir() -> str:
    # repository root -> .. -> 3440_Project/dataset/knapsack01_medium
    base = Path(__file__).resolve().parents[2]
    return str(base / "dataset" / "knapsack01_medium")


def _default_results_dir() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "results")


def _split_with_minimum(dataset, train_ratio: float, val_ratio: float):
    """Split dataset indices with at least one val and one test item when possible."""
    n = len(dataset)
    if n == 0:
        return [], [], []

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n >= 2:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        n_train = n - n_val - n_test
        # If counts over-allocated, trim val/test while keeping them >=1
        while n_train < 0:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
            n_train = n - n_val - n_test
    # If any remainder, give it to train
    if n_train + n_val + n_test < n:
        n_train += n - (n_train + n_val + n_test)

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))
    return train_indices, val_indices, test_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Knapsack GNN.")
    parser.add_argument(
        "--dataset_source",
        choices=["excel", "generated"],
        default="generated",
        help="Use Excel dataset or generated NPZ dataset.",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=_default_generated_dir(),
        help="Directory containing generated NPZ files (relative to Graph_Neural_Network).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="k for kNN graph when using the generated dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (generated dataset).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (generated dataset).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for DataLoaders.",
    )
    parser.add_argument(
        "--do_gnn_benchmark",
        action="store_true",
        help="Run GNN inference benchmark on the test split (generated mode only).",
    )
    parser.add_argument(
        "--benchmark_n",
        type=int,
        default=100,
        help="Number of instances to benchmark.",
    )
    parser.add_argument(
        "--benchmark_out_dir",
        type=str,
        default=_default_results_dir(),
        help="Output directory for benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for benchmarking.",
    )
    args = parser.parse_args()

    # If no CLI args were provided, override with requested defaults
    if len(sys.argv) == 1:
        args.dataset_source = "generated"
        args.generated_dir = r"C:\Users\GuanlinLi\Desktop\WKU\25 FAWZ\CPS 3440\3440_Project\dataset\knapsack01_medium"
        args.do_gnn_benchmark = True
        args.benchmark_n = 100
        args.benchmark_out_dir = _default_results_dir()

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    excel_path = "./dataset/data.xlsx"
    if args.dataset_source == "excel":
        dataset = KnapsackDataset(excel_path=excel_path)
        train_idx, val_idx, test_idx = _split_with_minimum(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
    else:
        dataset = GeneratedKnapsack01Dataset(
            root_dir=args.generated_dir,
            k=args.k,
        )
        train_idx, val_idx, test_idx = _split_with_minimum(
            dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)

    print(
        "Dataset source:",
        args.dataset_source,
        "| generated_dir:",
        args.generated_dir,
    )
    print(
        "Dataset size:",
        len(dataset),
        "| train:",
        len(train_set),
        "| val:",
        len(val_set),
        "| test:",
        len(test_set),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 3
    model = KnapsackGNN(in_dim=in_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = args.epochs
    train_start = perf_counter()
    for epoch in range(1, num_epochs + 1):
        epoch_start = perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_node_accuracy(model, val_loader, device)
        epoch_time_sec = perf_counter() - epoch_start
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"epoch_time_sec: {epoch_time_sec:.2f}"
        )
    total_train_time_sec = perf_counter() - train_start
    print(f"Total training time (sec): {total_train_time_sec:.2f}")

    # Save trained model for downstream evaluation/comparison (DP/DQN/GNN)
    gnn_save_dir = Path(__file__).resolve().parent.parent / "results" / "GNN"
    gnn_save_dir.mkdir(parents=True, exist_ok=True)
    gnn_save_path = gnn_save_dir / "gnn.pt"
    torch.save(model.state_dict(), gnn_save_path)
    print(f"[GNN-TRAIN] Saved model to {gnn_save_path.resolve()}")

    # Post-training comparison on the same Excel file
    if args.dataset_source == "excel":
        evaluate_gnn_vs_dp_on_excel(model, excel_path=excel_path)
    else:
        print("Skipping DP vs GNN Excel evaluation for generated dataset mode.")
        evaluate_instance_level(model, test_loader, num_cases=10, device=device)
        if args.do_gnn_benchmark:
            benchmark_out_path = Path(args.benchmark_out_dir)
            benchmark_out_path.mkdir(parents=True, exist_ok=True)
            saved_paths = run_gnn_benchmark(
                model,
                test_loader,
                device,
                args.benchmark_out_dir,
                n_instances=args.benchmark_n,
                seed=args.seed,
            )
            if saved_paths:
                print("Benchmark outputs:")
                for key, path in saved_paths.items():
                    print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
