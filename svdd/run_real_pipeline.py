#!/usr/bin/env python3
"""
Run the end-to-end real-data SVDD pipeline:
1. Merge the three real sensor CSVs
2. Split merged data into train/eval
3. Normalize using train statistics
4. Train the SVDD encoder
5. Evaluate and export predictions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "data_artifacts"


def run_step(args: List[str]) -> None:
    cmd = [sys.executable, *args]
    print(f"[run_real_pipeline] {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-data SVDD pipeline.")
    parser.add_argument("--am2302", type=str, default="AM2302.csv")
    parser.add_argument("--gy511", type=str, default="GY511.csv")
    parser.add_argument("--ld2410", type=str, default="LD2410.csv")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--drop-am2302-zero-rows", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["chronological", "alternating"],
        default="chronological",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional labeled eval CSV. If omitted, the generated template path is used.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse artifacts already present in the model output directory.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_dir = artifacts_dir / "real_model_output"

    merged_csv = artifacts_dir / "real_merged.csv"
    merged_ts_csv = artifacts_dir / "real_merged_with_timestamp.csv"
    train_csv = artifacts_dir / "real_train.csv"
    eval_csv = artifacts_dir / "real_eval.csv"
    train_norm_csv = artifacts_dir / "real_train_norm.csv"
    eval_norm_csv = artifacts_dir / "real_eval_norm.csv"
    norm_stats_json = artifacts_dir / "real_norm_stats.json"
    eval_labels_csv = Path(args.labels_csv) if args.labels_csv else artifacts_dir / "real_eval_labels.csv"
    eval_predictions_csv = model_dir / "eval_predictions.csv"

    try:
        prepare_cmd = [
            str(SCRIPT_DIR / "prepare_real_dataset.py"),
            "--am2302",
            args.am2302,
            "--gy511",
            args.gy511,
            "--ld2410",
            args.ld2410,
            "--output",
            str(merged_csv),
            "--keep-timestamp",
            "--timestamp-output",
            str(merged_ts_csv),
        ]
        if args.drop_am2302_zero_rows:
            prepare_cmd.append("--drop-am2302-zero-rows")
        run_step(prepare_cmd)

        split_cmd = [
            str(SCRIPT_DIR / "split_real_dataset.py"),
            "--input",
            str(merged_ts_csv),
            "--train-ratio",
            str(args.train_ratio),
            "--split-mode",
            args.split_mode,
            "--train-out",
            str(train_csv),
            "--eval-out",
            str(eval_csv),
            "--train-ts-out",
            str(artifacts_dir / "real_train_with_timestamp.csv"),
            "--eval-ts-out",
            str(artifacts_dir / "real_eval_with_timestamp.csv"),
            "--eval-labels-out",
            str(artifacts_dir / "real_eval_labels.csv"),
        ]
        run_step(split_cmd)

        normalize_cmd = [
            str(SCRIPT_DIR / "normalize_real_dataset.py"),
            "--train-in",
            str(train_csv),
            "--eval-in",
            str(eval_csv),
            "--train-out",
            str(train_norm_csv),
            "--eval-out",
            str(eval_norm_csv),
            "--stats-out",
            str(norm_stats_json),
        ]
        run_step(normalize_cmd)

        if not args.skip_train:
            train_cmd = [
                str(SCRIPT_DIR / "train_svdd.py"),
                "--input",
                str(train_norm_csv),
                "--output-dir",
                str(model_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
            ]
            run_step(train_cmd)

        if not args.skip_eval:
            eval_cmd = [
                str(SCRIPT_DIR / "evaluate_svdd.py"),
                "--mixed-norm",
                str(eval_norm_csv),
                "--model",
                str(model_dir / "svdd_encoder.keras"),
                "--config",
                str(model_dir / "svdd_config.json"),
                "--predictions-out",
                str(eval_predictions_csv),
            ]
            if eval_labels_csv.exists():
                eval_cmd.extend(["--labels-csv", str(eval_labels_csv)])
            run_step(eval_cmd)

        print("[run_real_pipeline] Pipeline complete.")
        print(f"[run_real_pipeline] Artifacts: {artifacts_dir}")
        print(f"[run_real_pipeline] Model output: {model_dir}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[run_real_pipeline] Step failed with exit code {exc.returncode}")
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
