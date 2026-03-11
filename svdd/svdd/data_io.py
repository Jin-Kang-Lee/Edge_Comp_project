from pathlib import Path
from typing import Optional

import numpy as np


def load_csv_numeric(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    try:
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {path}") from exc

    if data.size == 0:
        raise ValueError(f"CSV appears empty: {path}")

    if np.isnan(data).any():
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=1)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or Inf after loading.")

    return data.astype(np.float32)


def validate_feature_dim(data: np.ndarray, expected_dim: Optional[int] = None) -> None:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {data.shape}")
    if data.shape[1] <= 0:
        raise ValueError("Input data must contain at least one feature column.")
    if expected_dim is not None and data.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} input features, got {data.shape[1]}")
