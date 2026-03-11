import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf


def save_model(model: tf.keras.Model, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "svdd_encoder.keras"
    model.save(model_path)
    return model_path


def save_config(out_dir: Path, center: np.ndarray, p99: float, input_dim: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "center_c": center.tolist(),
        "p99_threshold": float(p99),
        "input_dim": int(input_dim),
        "embedding_dim": int(center.shape[0]),
    }
    config_path = out_dir / "svdd_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def save_report(out_dir: Path, loss_history: List[float], dists: np.ndarray) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "loss_history": loss_history,
        "distance_stats": {
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "mean": float(np.mean(dists)),
            "p50": float(np.percentile(dists, 50.0)),
            "p90": float(np.percentile(dists, 90.0)),
            "p95": float(np.percentile(dists, 95.0)),
            "p99": float(np.percentile(dists, 99.0)),
        },
    }
    report_path = out_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path
