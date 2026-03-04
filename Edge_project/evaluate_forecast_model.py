#!/usr/bin/env python3
"""Evaluate forecast anomaly detection on mixed mock dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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
MASK_COLUMNS = ["m0", "m1", "m2", "m3"]
SLOT_FEATURES = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}


def normalize_with_stats(df: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mask = df[MASK_COLUMNS].to_numpy(dtype=np.int32)
    out = feats.copy()

    for slot, idxs in SLOT_FEATURES.items():
        active = mask[:, slot] == 1
        for idx in idxs:
            out[active, idx] = (feats[active, idx] - means[idx]) / stds[idx]
            out[~active, idx] = 0.0
    return out


def build_windows(features: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size : i])
        Y.append(features[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"
    model_dir = root / "model_output"

    mixed_df = pd.read_csv(data_dir / "mock_mixed.csv")

    with (data_dir / "norm_stats.json").open("r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    with (data_dir / "mock_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (model_dir / "thresholds.json").open("r", encoding="utf-8") as f:
        th = json.load(f)

    means = np.array(norm_stats["means"], dtype=np.float32)
    stds = np.array(norm_stats["stds"], dtype=np.float32)
    threshold = float(th["mse_p99"])

    window_size = 10
    features_norm = normalize_with_stats(mixed_df, means, stds)
    X, Y = build_windows(features_norm, window_size)

    model = tf.keras.models.load_model(model_dir / "forecast_model.keras")
    pred = model.predict(X.reshape(len(X), -1), verbose=0)
    mse = np.mean((Y - pred) ** 2, axis=1)

    y_pred = (mse > threshold).astype(np.int32)

    labels = np.array(meta["mixed_labels"], dtype=np.int32)
    y_true = labels[window_size:]  # aligned with Y target indices

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = safe_div(tp + tn, len(y_true))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    print("Evaluation on mock_mixed.csv")
    print(f"Threshold (mse_p99): {threshold:.6f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")


if __name__ == "__main__":
    main()
