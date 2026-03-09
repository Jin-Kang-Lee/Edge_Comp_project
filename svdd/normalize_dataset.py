#!/usr/bin/env python3
"""
Mask-aware Z-score normalization for contract_v1 datasets.

Inputs (relative to project root):
  - data_artifacts/mock_normal.csv
  - data_artifacts/mock_mixed.csv

Outputs:
  - data_artifacts/mock_normal_norm.csv
  - data_artifacts/mock_mixed_norm.csv
  - data_artifacts/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("normalize_dataset")


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
    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS + MASK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def compute_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    for mask_col, feat_cols in SLOT_FEATURES.items():
        active = df[mask_col] == 1
        if active.any():
            for col in feat_cols:
                vals = df.loc[active, col].astype(np.float32)
                mean = float(vals.mean())
                std = float(vals.std(ddof=0))
                if std == 0.0:
                    std = 1.0
                stats[col] = {"mean": mean, "std": std}
        else:
            for col in feat_cols:
                stats[col] = {"mean": 0.0, "std": 1.0}

    return stats


def apply_normalization(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    norm_df = df.copy()

    for mask_col, feat_cols in SLOT_FEATURES.items():
        active = norm_df[mask_col] == 1
        for col in feat_cols:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            # Normalize only when mask is active; else keep zeros.
            norm_df.loc[active, col] = (norm_df.loc[active, col] - mean) / std
            norm_df.loc[~active, col] = 0.0

    return norm_df


def save_stats(path: Path, stats: Dict[str, Dict[str, float]]) -> None:
    payload = {
        "contract": "contract_v1",
        "features": FEATURE_COLUMNS,
        "masks": MASK_COLUMNS,
        "stats": stats,
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize datasets with mask-aware Z-score scaling.")
    parser.add_argument("--normal-in", type=str, default="data_artifacts/mock_normal.csv")
    parser.add_argument("--mixed-in", type=str, default="data_artifacts/mock_mixed.csv")
    parser.add_argument("--normal-out", type=str, default="data_artifacts/mock_normal_norm.csv")
    parser.add_argument("--mixed-out", type=str, default="data_artifacts/mock_mixed_norm.csv")
    parser.add_argument("--stats-out", type=str, default="data_artifacts/norm_stats.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        normal_path = Path(args.normal_in)
        mixed_path = Path(args.mixed_in)

        LOGGER.info("Loading normal dataset: %s", normal_path)
        normal_df = load_csv(normal_path)
        validate_schema(normal_df)

        LOGGER.info("Computing stats from normal data...")
        stats = compute_stats(normal_df)

        LOGGER.info("Normalizing normal dataset...")
        normal_norm = apply_normalization(normal_df, stats)

        LOGGER.info("Loading mixed dataset: %s", mixed_path)
        mixed_df = load_csv(mixed_path)
        validate_schema(mixed_df)

        LOGGER.info("Normalizing mixed dataset...")
        mixed_norm = apply_normalization(mixed_df, stats)

        out_normal = Path(args.normal_out)
        out_mixed = Path(args.mixed_out)
        out_stats = Path(args.stats_out)

        out_normal.parent.mkdir(parents=True, exist_ok=True)
        out_mixed.parent.mkdir(parents=True, exist_ok=True)
        out_stats.parent.mkdir(parents=True, exist_ok=True)

        normal_norm.to_csv(out_normal, index=False)
        mixed_norm.to_csv(out_mixed, index=False)
        save_stats(out_stats, stats)

        LOGGER.info("Saved normalized normal dataset to %s", out_normal)
        LOGGER.info("Saved normalized mixed dataset to %s", out_mixed)
        LOGGER.info("Saved normalization stats to %s", out_stats)
        return 0
    except Exception as exc:
        LOGGER.exception("Normalization failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
