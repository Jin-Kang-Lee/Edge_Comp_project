#!/usr/bin/env python3
"""
Validate contract_v1 datasets (schema + mask padding + basic sanity checks).

Inputs (relative to project root):
  - data_artifacts/mock_normal.csv
  - data_artifacts/mock_mixed.csv
  - data_artifacts/mock_normal_norm.csv (optional)
  - data_artifacts/mock_mixed_norm.csv (optional)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("validate_dataset")


FEATURE_COLUMNS = [
    "duty_cycle",
    "env_mean",
    "env_std",
    "env_min",
    "env_max",
    "vib_mean",
    "vib_std",
    "vib_min",
    "vib_max",
    "other_mean",
    "other_std",
    "other_min",
    "other_max",
]

MASK_COLUMNS = ["m0", "m1", "m2", "m3"]

SLOT_FEATURES = {
    "m0": ["duty_cycle"],
    "m1": ["env_mean", "env_std", "env_min", "env_max"],
    "m2": ["vib_mean", "vib_std", "vib_min", "vib_max"],
    "m3": ["other_mean", "other_std", "other_min", "other_max"],
}


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    expected = FEATURE_COLUMNS + MASK_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    extra = [c for c in df.columns if c not in expected]
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    if df.shape[1] != 17:
        errors.append(f"Expected 17 columns, got {df.shape[1]}")
    return errors


def validate_masks(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    for m in MASK_COLUMNS:
        if not set(df[m].dropna().unique()).issubset({0, 1}):
            errors.append(f"Mask {m} contains values outside {{0,1}}.")
    return errors


def validate_padding(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    for mask, cols in SLOT_FEATURES.items():
        inactive = df[mask] == 0
        if inactive.any():
            non_zero = (df.loc[inactive, cols].abs().sum(axis=1) > 0)
            if non_zero.any():
                count = int(non_zero.sum())
                errors.append(f"Padding violation: {count} rows have non-zero features when {mask}=0.")
    return errors


def validate_basic_sanity(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if np.isnan(df[FEATURE_COLUMNS].to_numpy()).any():
        errors.append("NaN detected in feature columns.")
    if np.isinf(df[FEATURE_COLUMNS].to_numpy()).any():
        errors.append("Inf detected in feature columns.")

    # Logical consistency: min <= mean <= max for scalar slots where active.
    for prefix, mask in [("env", "m1"), ("vib", "m2"), ("other", "m3")]:
        active = df[mask] == 1
        if active.any():
            mean = df.loc[active, f"{prefix}_mean"]
            min_v = df.loc[active, f"{prefix}_min"]
            max_v = df.loc[active, f"{prefix}_max"]
            if (min_v > mean).any():
                errors.append(f"{prefix}: found min > mean in active rows.")
            if (max_v < mean).any():
                errors.append(f"{prefix}: found max < mean in active rows.")
    return errors


def run_checks(path: Path) -> int:
    df = load_csv(path)
    errors: List[str] = []
    errors.extend(validate_schema(df))
    errors.extend(validate_masks(df))
    errors.extend(validate_padding(df))
    errors.extend(validate_basic_sanity(df))

    if errors:
        LOGGER.error("Validation failed for %s", path)
        for e in errors:
            LOGGER.error("  - %s", e)
        return 1

    LOGGER.info("Validation OK: %s", path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate contract_v1 CSV datasets.")
    parser.add_argument("--normal", type=str, default="data_artifacts/mock_normal.csv")
    parser.add_argument("--mixed", type=str, default="data_artifacts/mock_mixed.csv")
    parser.add_argument("--normal-norm", type=str, default="data_artifacts/mock_normal_norm.csv")
    parser.add_argument("--mixed-norm", type=str, default="data_artifacts/mock_mixed_norm.csv")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    paths = [Path(args.normal), Path(args.mixed), Path(args.normal_norm), Path(args.mixed_norm)]
    exit_code = 0
    for path in paths:
        if path.exists():
            exit_code |= run_checks(path)
        else:
            LOGGER.warning("Skipping missing file: %s", path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
