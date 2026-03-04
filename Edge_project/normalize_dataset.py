#!/usr/bin/env python3
"""Mask-aware z-score normalization for contract_v1 datasets."""

from __future__ import annotations

import json
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
MASK_COLUMNS = ["m0", "m1", "m2", "m3"]
SLOT_FEATURES = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"

    src = data_dir / "mock_normal.csv"
    out_csv = data_dir / "mock_normal_norm.csv"
    stats_json = data_dir / "norm_stats.json"

    df = pd.read_csv(src)
    feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mask = df[MASK_COLUMNS].to_numpy(dtype=np.int32)

    means = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
    stds = np.ones(len(FEATURE_COLUMNS), dtype=np.float32)

    norm_feats = feats.copy()

    for slot, idxs in SLOT_FEATURES.items():
        active_rows = mask[:, slot] == 1
        for idx in idxs:
            if np.any(active_rows):
                vals = feats[active_rows, idx]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
            else:
                mean = 0.0
                std = 1.0

            if std < 1e-6:
                std = 1.0

            means[idx] = mean
            stds[idx] = std

            norm_feats[active_rows, idx] = (feats[active_rows, idx] - mean) / std
            norm_feats[~active_rows, idx] = 0.0  # inactive values stay exactly padded

    out_df = pd.DataFrame(norm_feats, columns=FEATURE_COLUMNS)
    out_df[MASK_COLUMNS] = mask
    out_df.to_csv(out_csv, index=False)

    stats = {
        "feature_columns": FEATURE_COLUMNS,
        "mask_columns": MASK_COLUMNS,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "slot_features": {str(k): v for k, v in SLOT_FEATURES.items()},
        "source": "mock_normal.csv",
    }
    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {out_csv}")
    print(f"Saved {stats_json}")


if __name__ == "__main__":
    main()
