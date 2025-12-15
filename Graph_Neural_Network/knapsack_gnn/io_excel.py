from pathlib import Path
from typing import List, Dict

import pandas as pd


def resolve_excel_path(path: str, default_rel: str = "dataset/data.xlsx") -> Path:
    """Resolve an excel path.

    - If `path` is relative, it is resolved against the project root (three levels up from this file).
    - If it does not exist, fall back to `default_rel` under the project root.
    """
    # __file__ -> .../Graph_Neural_Network/knapsack_gnn/io_excel.py
    # Project root is one level above Graph_Neural_Network so we need parents[2].
    project_root = Path(__file__).resolve().parents[2]
    default_path = (project_root / default_rel).resolve()

    user_path = Path(path)
    if not user_path.is_absolute():
        user_path = (project_root / user_path).resolve()

    excel_path = user_path if user_path.is_file() else default_path
    if not excel_path.is_file():
        raise FileNotFoundError(f"Knapsack Excel file not found at: {excel_path.resolve()}")
    return excel_path


def load_excel_knapsack_instances(path: str) -> List[Dict]:
    """Load up to 8 knapsack instances from an Excel file.

    Expected columns per instance i:
        weight{i}, value{i}, cap{i}

    Notes:
        - weights/values are read from non-NaN pairs.
        - capacity is read from the first non-NaN value in cap{i}.
          (This supports the common format where cap{i} appears only in the first row.)
    """
    excel_path = resolve_excel_path(path)
    df = pd.read_excel(excel_path)

    instances: List[Dict] = []
    for i in range(1, 9):
        w_col, v_col, c_col = f"weight{i}", f"value{i}", f"cap{i}"
        if w_col not in df.columns or v_col not in df.columns or c_col not in df.columns:
            continue

        pair_df = df[[w_col, v_col]].dropna(how="any")
        if pair_df.empty:
            continue

        cap_series = df[c_col].dropna()
        if cap_series.empty:
            continue

        weights = pair_df[w_col].astype(int).tolist()
        values = pair_df[v_col].astype(int).tolist()
        capacity = int(cap_series.iloc[0])

        instances.append({"weights": weights, "values": values, "capacity": capacity})

    return instances
