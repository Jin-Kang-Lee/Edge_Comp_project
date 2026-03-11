#!/usr/bin/env python3
"""
Train a Deep SVDD encoder on normalized "normal" data.

Requirements:
  - TensorFlow / Keras 3.x
  - NumPy

Input:
  - CSV with normalized features (default: data_artifacts/mock_normal_norm.csv)

Output:
  - Keras encoder model
  - svdd_config.json with center 'c' and P99 threshold
  - train_report.json with loss history and distance stats
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from svdd.artifacts import save_config, save_model, save_report
from svdd.constants import INPUT_NORMAL_CSV, OUTPUT_MODEL_DIR
from svdd.data_io import load_csv_numeric, validate_feature_dim
from svdd.logging_utils import setup_logging
from svdd.model import build_encoder
from svdd.objective import compute_center, compute_distances, svdd_loss


LOGGER = logging.getLogger("train_svdd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deep SVDD encoder.")
    parser.add_argument("--input", type=str, default=str(INPUT_NORMAL_CSV), help="Path to normalized normal CSV.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_MODEL_DIR), help="Directory for model artifacts.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

        input_path = Path(args.input)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading data: %s", input_path)
        data = load_csv_numeric(input_path)
        LOGGER.info("Loaded data shape: %s", data.shape)
        validate_feature_dim(data)

        input_dim = int(data.shape[1])
        model = build_encoder(input_dim=input_dim)

        # Warmup: compute center 'c' from a single forward pass on 10% of data.
        batch_size = max(1, int(args.batch_size))
        warmup_count = max(1, int(0.10 * data.shape[0]))
        warmup_batch = data[:warmup_count]
        if warmup_batch.shape[0] == 0:
            raise ValueError("Not enough data for warmup pass.")

        center = compute_center(model, warmup_batch)
        center_tf = tf.constant(center, dtype=tf.float32)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=svdd_loss(center_tf),
        )

        # Dummy labels are unused by the loss.
        dummy_y = np.zeros((data.shape[0], 1), dtype=np.float32)
        history = model.fit(
            data,
            dummy_y,
            epochs=args.epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
        )

        LOGGER.info("Training complete.")

        # Compute distances and P99 threshold.
        dists = compute_distances(model, data, center, batch_size=batch_size)
        p99 = float(np.percentile(dists, 99.0))

        # Save artifacts.
        model_path = save_model(model, out_dir)
        config_path = save_config(out_dir, center, p99, input_dim)
        report_path = save_report(out_dir, history.history.get("loss", []), dists)

        LOGGER.info("Saved model to %s", model_path)
        LOGGER.info("Saved config to %s", config_path)
        LOGGER.info("Saved report to %s", report_path)

        return 0
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
