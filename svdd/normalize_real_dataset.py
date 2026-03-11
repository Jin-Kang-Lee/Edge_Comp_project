#!/usr/bin/env python3
"""
Normalize a real merged SVDD dataset using Z-score statistics from the training split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("normalize_real_dataset")


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize merged real SVDD datasets.")
    parser.add_argument("--train-in", type=str, required=True, help="Training CSV used to compute stats.")
    parser.add_argument("--eval-in", type=str, default=None, help="Optional evaluation CSV normalized with train stats.")
    parser.add_argument("--train-out", type=str, required=True)
    parser.add_argument("--eval-out", type=str, default=None)
    parser.add_argument("--stats-out", type=str, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_numeric_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    return df


def normalize(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    return ((df - means) / stds).astype("float32")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        train_in = Path(args.train_in)
        train_out = Path(args.train_out)
        stats_out = Path(args.stats_out)

        train_df = load_numeric_frame(train_in)
        means = train_df.mean(axis=0)
        stds = train_df.std(axis=0, ddof=0).replace(0.0, 1.0)

        train_norm = normalize(train_df, means, stds)
        train_out.parent.mkdir(parents=True, exist_ok=True)
        train_norm.to_csv(train_out, index=False)

        payload = {
            "columns": list(train_df.columns),
            "stats": {
                column: {"mean": float(means[column]), "std": float(stds[column])}
                for column in train_df.columns
            },
        }
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Saved normalized train dataset to %s", train_out)
        LOGGER.info("Saved normalization stats to %s", stats_out)

        if args.eval_in:
            if not args.eval_out:
                raise ValueError("--eval-out is required when --eval-in is provided.")
            eval_in = Path(args.eval_in)
            eval_out = Path(args.eval_out)
            eval_df = load_numeric_frame(eval_in)
            if list(eval_df.columns) != list(train_df.columns):
                raise ValueError("Evaluation columns do not match training columns.")
            eval_norm = normalize(eval_df, means, stds)
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_norm.to_csv(eval_out, index=False)
            LOGGER.info("Saved normalized eval dataset to %s", eval_out)

        return 0
    except Exception as exc:
        LOGGER.exception("Normalization failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
