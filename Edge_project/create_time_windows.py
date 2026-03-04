#!/usr/bin/env python3
"""Create train/test forecasting windows from normalized rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "duty_cycle",
    "env_mean",
    "env_std",
    "env_min",
    "env_max",
    "acc_mean_abs",
    "acc_std_abs",
    "acc_min_abs",
    "acc_max_abs",
    "other_mean",
    "other_std",
    "other_min",
    "other_max",
]


def build_windows(features: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window_size, len(features)):
        xs.append(features[i - window_size : i])
        ys.append(features[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"

    src = data_dir / "mock_normal_norm.csv"
    out_train = data_dir / "mock_windows_train.npz"
    out_test = data_dir / "mock_windows_test.npz"

    window_size = 10

    df = pd.read_csv(src)
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    X, Y = build_windows(features, window_size=window_size)

    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    np.savez_compressed(out_train, X=X_train, Y=Y_train, window_size=window_size)
    np.savez_compressed(out_test, X=X_test, Y=Y_test, window_size=window_size)

    print(f"Saved {out_train}: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Saved {out_test}: X={X_test.shape}, Y={Y_test.shape}")


if __name__ == "__main__":
    main()
