#!/usr/bin/env python3
"""
Merge the real AM2302, GY511, and LD2410 CSVs into one SVDD-ready feature table.

The output schema is:
  temp_mean,temp_std,temp_min,temp_max,
  hum_mean,hum_std,hum_min,hum_max,
  accel_mean,accel_std,accel_min,accel_max,
  dist_mean,dist_std,dist_min,dist_max

Optionally filters obvious AM2302 dropout rows where all temperature and
humidity summary values are zero.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger("prepare_real_dataset")


OUTPUT_COLUMNS = [
    "temp_mean",
    "temp_std",
    "temp_min",
    "temp_max",
    "hum_mean",
    "hum_std",
    "hum_min",
    "hum_max",
    "accel_mean",
    "accel_std",
    "accel_min",
    "accel_max",
    "dist_mean",
    "dist_std",
    "dist_min",
    "dist_max",
]


AM2302_COLUMNS = [
    "temp_mean",
    "temp_std",
    "temp_min",
    "temp_max",
    "hum_mean",
    "hum_std",
    "hum_min",
    "hum_max",
]


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_csv(path: Path, expected_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[list(expected_columns)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a merged real-sensor dataset for SVDD.")
    parser.add_argument("--am2302", type=str, default="AM2302.csv")
    parser.add_argument("--gy511", type=str, default="GY511.csv")
    parser.add_argument("--ld2410", type=str, default="LD2410.csv")
    parser.add_argument("--output", type=str, default="svdd/data_artifacts/real_merged.csv")
    parser.add_argument(
        "--keep-timestamp",
        action="store_true",
        help="Keep the timestamp column in a separate output for inspection.",
    )
    parser.add_argument(
        "--timestamp-output",
        type=str,
        default="svdd/data_artifacts/real_merged_with_timestamp.csv",
        help="Where to save the merged table with timestamps when --keep-timestamp is set.",
    )
    parser.add_argument(
        "--drop-am2302-zero-rows",
        action="store_true",
        help="Drop rows where all AM2302 temperature and humidity summary values are zero.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def filter_am2302_dropouts(df: pd.DataFrame) -> pd.DataFrame:
    zero_mask = (df.loc[:, AM2302_COLUMNS].abs().sum(axis=1) == 0.0)
    return df.loc[~zero_mask].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        am = load_csv(
            Path(args.am2302),
            [
                "timestamp",
                "temp_mean",
                "temp_std",
                "temp_min",
                "temp_max",
                "hum_mean",
                "hum_std",
                "hum_min",
                "hum_max",
            ],
        )
        gy = load_csv(
            Path(args.gy511),
            ["timestamp", "accel_mean", "accel_std", "accel_min", "accel_max"],
        )
        ld = load_csv(
            Path(args.ld2410),
            ["timestamp", "dist_mean", "dist_std", "dist_min", "dist_max"],
        )

        merged = am.merge(gy, on="timestamp", how="inner").merge(ld, on="timestamp", how="inner")
        if merged.empty:
            raise ValueError("No overlapping timestamps found across the three CSVs.")

        merged = merged.sort_values("timestamp").reset_index(drop=True)

        if args.drop_am2302_zero_rows:
            before = merged.shape[0]
            merged = filter_am2302_dropouts(merged)
            LOGGER.info("Dropped %d AM2302 zero rows", before - merged.shape[0])

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[:, OUTPUT_COLUMNS].to_csv(out_path, index=False)
        LOGGER.info("Saved %d merged rows to %s", merged.shape[0], out_path)

        if args.keep_timestamp:
            ts_path = Path(args.timestamp_output)
            ts_path.parent.mkdir(parents=True, exist_ok=True)
            merged.loc[:, ["timestamp", *OUTPUT_COLUMNS]].to_csv(ts_path, index=False)
            LOGGER.info("Saved timestamp-aligned view to %s", ts_path)

        return 0
    except Exception as exc:
        LOGGER.exception("Failed to prepare real dataset: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
