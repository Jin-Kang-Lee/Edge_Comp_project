#!/usr/bin/env python3
"""Strict validator for contract_v1 datasets."""

from __future__ import annotations

import argparse
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


def validate_dataframe(df: pd.DataFrame, source: str) -> None:
    if df.shape[1] != 17:
        raise ValueError(f"{source}: expected exactly 17 columns, got {df.shape[1]}")

    missing = [c for c in FEATURE_COLUMNS + MASK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")

    if df[FEATURE_COLUMNS + MASK_COLUMNS].isna().any().any():
        raise ValueError(f"{source}: NaN values detected")

    mask = df[MASK_COLUMNS].to_numpy(dtype=np.int32)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{source}: mask columns must contain only 0 or 1")

    feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    for slot, idxs in SLOT_FEATURES.items():
        inactive_rows = mask[:, slot] == 0
        if inactive_rows.any():
            inactive_values = feats[np.ix_(inactive_rows, idxs)]
            if not np.allclose(inactive_values, 0.0, atol=0.0):
                raise ValueError(
                    f"{source}: inactive slot {slot} has non-zero padded values"
                )

    print(f"{source}: OK ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate contract_v1 CSV datasets")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["data_mock/mock_normal.csv", "data_mock/mock_mixed.csv"],
        help="CSV files to validate",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for rel in args.files:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        validate_dataframe(df, source=str(path))


if __name__ == "__main__":
    main()
