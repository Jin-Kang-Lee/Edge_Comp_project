#!/usr/bin/env python3
"""
Split a timestamp-aligned real dataset into train/eval CSVs and emit a label template.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd


LOGGER = logging.getLogger("split_real_dataset")


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split real SVDD data into train/eval sets.")
    parser.add_argument(
        "--input",
        type=str,
        default="svdd/data_artifacts/real_merged_with_timestamp.csv",
        help="Timestamp-aligned merged CSV.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of rows assigned to the train split.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["chronological", "alternating"],
        default="chronological",
        help="Chronological keeps early rows for train. Alternating interleaves rows.",
    )
    parser.add_argument("--train-out", type=str, default="svdd/data_artifacts/real_train.csv")
    parser.add_argument("--eval-out", type=str, default="svdd/data_artifacts/real_eval.csv")
    parser.add_argument(
        "--train-ts-out",
        type=str,
        default="svdd/data_artifacts/real_train_with_timestamp.csv",
    )
    parser.add_argument(
        "--eval-ts-out",
        type=str,
        default="svdd/data_artifacts/real_eval_with_timestamp.csv",
    )
    parser.add_argument(
        "--eval-labels-out",
        type=str,
        default="svdd/data_artifacts/real_eval_labels.csv",
        help="CSV template with row_index,timestamp,label for manual labeling.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Expected a timestamp column in the input CSV.")
    if df.empty:
        raise ValueError("Input CSV is empty.")
    return df


def split_frame(df: pd.DataFrame, train_ratio: float, split_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if split_mode == "chronological":
        cutoff = max(1, min(len(df) - 1, int(len(df) * train_ratio)))
        return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()

    eval_every = max(2, round(1.0 / max(1e-6, 1.0 - train_ratio)))
    train_mask = (df.index % eval_every) != 0
    train = df.loc[train_mask].copy()
    eval_df = df.loc[~train_mask].copy()
    if train.empty or eval_df.empty:
        raise ValueError("Alternating split produced an empty train or eval set.")
    return train, eval_df


def save_split(feature_df: pd.DataFrame, ts_df: pd.DataFrame, feature_path: Path, ts_path: Path) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_path, index=False)
    ts_df.to_csv(ts_path, index=False)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        df = load_frame(Path(args.input))
        train_df, eval_df = split_frame(df, args.train_ratio, args.split_mode)

        feature_columns = [column for column in df.columns if column != "timestamp"]
        save_split(
            train_df.loc[:, feature_columns],
            train_df,
            Path(args.train_out),
            Path(args.train_ts_out),
        )
        save_split(
            eval_df.loc[:, feature_columns],
            eval_df,
            Path(args.eval_out),
            Path(args.eval_ts_out),
        )

        labels_df = eval_df.loc[:, ["timestamp"]].reset_index(drop=True)
        labels_df.insert(0, "row_index", labels_df.index.astype("int32"))
        labels_df["label"] = 0
        labels_path = Path(args.eval_labels_out)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(labels_path, index=False)

        LOGGER.info("Saved train split: %d rows", train_df.shape[0])
        LOGGER.info("Saved eval split: %d rows", eval_df.shape[0])
        LOGGER.info("Saved label template to %s", labels_path)
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to split real dataset: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
