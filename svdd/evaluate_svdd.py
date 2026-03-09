#!/usr/bin/env python3
"""
Evaluate Deep SVDD on mixed normalized data.

Inputs (relative to project root):
  - data_artifacts/mock_mixed_norm.csv
  - data_artifacts/mock_meta.json
  - data_artifacts/model_output/svdd_encoder.keras
  - data_artifacts/model_output/svdd_config.json

Outputs:
  - Prints metrics to stdout
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from svdd.constants import OUTPUT_MODEL_DIR
from svdd.data_io import load_csv_numeric, validate_feature_dim
from svdd.logging_utils import setup_logging
from svdd.objective import compute_distances


LOGGER = logging.getLogger("evaluate_svdd")


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text())


def load_meta(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Meta not found: {path}")
    return json.loads(path.read_text())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Deep SVDD on mixed dataset.")
    parser.add_argument("--mixed-norm", type=str, default="data_artifacts/mock_mixed_norm.csv")
    parser.add_argument("--meta", type=str, default="data_artifacts/mock_meta.json")
    parser.add_argument(
        "--model",
        type=str,
        default=str(OUTPUT_MODEL_DIR / "svdd_encoder.keras"),
        help="Path to trained encoder (.keras).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(OUTPUT_MODEL_DIR / "svdd_config.json"),
        help="Path to SVDD config (center + threshold).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        mixed_path = Path(args.mixed_norm)
        meta_path = Path(args.meta)
        model_path = Path(args.model)
        config_path = Path(args.config)

        LOGGER.info("Loading mixed normalized data: %s", mixed_path)
        mixed = load_csv_numeric(mixed_path)
        validate_feature_dim(mixed)

        LOGGER.info("Loading model: %s", model_path)
        model = tf.keras.models.load_model(model_path, compile=False)

        LOGGER.info("Loading config: %s", config_path)
        config = load_config(config_path)
        center = np.array(config["center_c"], dtype=np.float32)
        threshold = float(config["p99_threshold"])

        LOGGER.info("Computing distances...")
        dists = compute_distances(model, mixed, center, batch_size=args.batch_size)
        preds = (dists > threshold).astype(np.int32)

        LOGGER.info("Loading meta: %s", meta_path)
        meta = load_meta(meta_path)
        anomaly_indices = set(meta.get("mixed_anomaly_indices", []))
        y_true = np.zeros(mixed.shape[0], dtype=np.int32)
        if anomaly_indices:
            idx = np.array(sorted(anomaly_indices), dtype=np.int32)
            idx = idx[idx < mixed.shape[0]]
            y_true[idx] = 1
        else:
            LOGGER.warning("No anomaly indices found in meta; metrics may be meaningless.")

        metrics = compute_metrics(y_true, preds)

        LOGGER.info("Evaluation results (threshold=%.6f):", threshold)
        for key in ["accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]:
            LOGGER.info("  %s: %s", key, metrics[key])

        return 0
    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
