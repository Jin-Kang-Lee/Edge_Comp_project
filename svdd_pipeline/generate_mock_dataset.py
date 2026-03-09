#!/usr/bin/env python3
"""
Generate synthetic datasets following contract_v1 for Edge_DeepSVDD.

Outputs (relative to project root):
  - data_artifacts/mock_normal.csv (5000 rows)
  - data_artifacts/mock_mixed.csv  (1000 rows, 10% anomalies)
  - data_artifacts/mock_meta.json (metadata + anomaly indices)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("generate_mock_dataset")


COLUMNS = [
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
    "m0",
    "m1",
    "m2",
    "m3",
]


SLOT_RANGES = {
    "duty_cycle": {"mean": 45.0, "std": 3.0, "min": 0.0, "max": 100.0},
    "env": {"mean": 25.0, "std": 0.8, "range": 2.0},
    "vib": {"mean": 0.02, "std": 0.01, "range": 0.05},
    "other": {"mean": 10.0, "std": 1.0, "range": 3.0},
}


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_sensor_config(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("sensor_config must have 4 comma-separated values (e.g., 1,1,1,0).")
    config = []
    for p in parts:
        if p not in {"0", "1"}:
            raise ValueError("sensor_config values must be 0 or 1.")
        config.append(int(p))
    return config


def ensure_min_max(mean: np.ndarray, base_range: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    span = np.abs(base_range + rng.normal(0.0, base_range * 0.1, size=mean.shape))
    min_v = mean - span
    max_v = mean + span
    return min_v, max_v


def generate_normal_rows(n: int, sensor_config: List[int], rng: np.random.Generator) -> np.ndarray:
    data = np.zeros((n, 17), dtype=np.float32)

    # Slot 0: duty_cycle
    if sensor_config[0] == 1:
        duty = rng.normal(SLOT_RANGES["duty_cycle"]["mean"], SLOT_RANGES["duty_cycle"]["std"], size=n)
        duty = np.clip(duty, SLOT_RANGES["duty_cycle"]["min"], SLOT_RANGES["duty_cycle"]["max"])
        data[:, 0] = duty

    # Slot 1: environmental (mean, std, min, max)
    if sensor_config[1] == 1:
        mean = rng.normal(SLOT_RANGES["env"]["mean"], SLOT_RANGES["env"]["std"], size=n)
        std = np.abs(rng.normal(SLOT_RANGES["env"]["std"], SLOT_RANGES["env"]["std"] * 0.1, size=n))
        std = np.maximum(std, 1e-3)
        min_v, max_v = ensure_min_max(mean, SLOT_RANGES["env"]["range"], rng)
        data[:, 1:5] = np.stack([mean, std, min_v, max_v], axis=1)

    # Slot 2: vibration/inertial
    if sensor_config[2] == 1:
        mean = rng.normal(SLOT_RANGES["vib"]["mean"], SLOT_RANGES["vib"]["std"], size=n)
        std = np.abs(rng.normal(SLOT_RANGES["vib"]["std"], SLOT_RANGES["vib"]["std"] * 0.1, size=n))
        std = np.maximum(std, 1e-4)
        min_v, max_v = ensure_min_max(mean, SLOT_RANGES["vib"]["range"], rng)
        data[:, 5:9] = np.stack([mean, std, min_v, max_v], axis=1)

    # Slot 3: other scalar
    if sensor_config[3] == 1:
        mean = rng.normal(SLOT_RANGES["other"]["mean"], SLOT_RANGES["other"]["std"], size=n)
        std = np.abs(rng.normal(SLOT_RANGES["other"]["std"], SLOT_RANGES["other"]["std"] * 0.1, size=n))
        std = np.maximum(std, 1e-3)
        min_v, max_v = ensure_min_max(mean, SLOT_RANGES["other"]["range"], rng)
        data[:, 9:13] = np.stack([mean, std, min_v, max_v], axis=1)

    # Masks (m0..m3)
    data[:, 13:17] = np.array(sensor_config, dtype=np.float32)
    return data


def apply_anomaly(
    row: np.ndarray,
    sensor_config: List[int],
    rng: np.random.Generator,
) -> None:
    anomaly_type = rng.choice(["spike", "stuck", "out_of_bounds"])

    # Slot 0: duty_cycle
    if sensor_config[0] == 1:
        if anomaly_type == "spike":
            row[0] = 150.0 + rng.normal(0.0, 5.0)
        elif anomaly_type == "stuck":
            row[0] = 0.0
        else:
            row[0] = -20.0 + rng.normal(0.0, 2.0)

    # Slot 1: environmental
    if sensor_config[1] == 1:
        if anomaly_type == "spike":
            mean = 80.0 + rng.normal(0.0, 5.0)
            std = 5.0
        elif anomaly_type == "stuck":
            mean = 25.0
            std = 0.0
        else:
            mean = -10.0 + rng.normal(0.0, 2.0)
            std = 1.5
        min_v, max_v = ensure_min_max(np.array([mean]), 5.0, rng)
        row[1:5] = [mean, std, min_v[0], max_v[0]]

    # Slot 2: vibration/inertial
    if sensor_config[2] == 1:
        if anomaly_type == "spike":
            mean = 0.5 + rng.normal(0.0, 0.05)
            std = 0.2
        elif anomaly_type == "stuck":
            mean = 0.02
            std = 0.0
        else:
            mean = -0.2 + rng.normal(0.0, 0.02)
            std = 0.05
        min_v, max_v = ensure_min_max(np.array([mean]), 0.2, rng)
        row[5:9] = [mean, std, min_v[0], max_v[0]]

    # Slot 3: other scalar
    if sensor_config[3] == 1:
        if anomaly_type == "spike":
            mean = 50.0 + rng.normal(0.0, 2.0)
            std = 8.0
        elif anomaly_type == "stuck":
            mean = 10.0
            std = 0.0
        else:
            mean = -5.0 + rng.normal(0.0, 1.0)
            std = 2.0
        min_v, max_v = ensure_min_max(np.array([mean]), 6.0, rng)
        row[9:13] = [mean, std, min_v[0], max_v[0]]

    # Enforce padding rule for inactive slots.
    if sensor_config[0] == 0:
        row[0] = 0.0
    if sensor_config[1] == 0:
        row[1:5] = 0.0
    if sensor_config[2] == 0:
        row[5:9] = 0.0
    if sensor_config[3] == 0:
        row[9:13] = 0.0

    row[13:17] = np.array(sensor_config, dtype=np.float32)


def generate_mixed_rows(
    n: int, anomaly_frac: float, sensor_config: List[int], rng: np.random.Generator
) -> Tuple[np.ndarray, List[int]]:
    data = generate_normal_rows(n, sensor_config, rng)
    anomaly_count = int(round(n * anomaly_frac))
    anomaly_indices = rng.choice(n, size=anomaly_count, replace=False).tolist()
    for idx in anomaly_indices:
        apply_anomaly(data[idx], sensor_config, rng)
    return data, anomaly_indices


def save_outputs(out_dir: Path, normal: np.ndarray, mixed: np.ndarray, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    normal_df = pd.DataFrame(normal, columns=COLUMNS)
    mixed_df = pd.DataFrame(mixed, columns=COLUMNS)

    normal_df.to_csv(out_dir / "mock_normal.csv", index=False)
    mixed_df.to_csv(out_dir / "mock_mixed.csv", index=False)

    (out_dir / "mock_meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock datasets for SVDD.")
    parser.add_argument("--rows-normal", type=int, default=5000, help="Number of normal rows.")
    parser.add_argument("--rows-mixed", type=int, default=1000, help="Number of mixed rows.")
    parser.add_argument("--anomaly-frac", type=float, default=0.10, help="Fraction of anomalies in mixed set.")
    parser.add_argument("--sensor-config", type=str, default="1,1,1,1", help="Active slots, e.g. 1,1,1,0")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default="data_artifacts", help="Output directory.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        rng = np.random.default_rng(args.seed)
        sensor_config = parse_sensor_config(args.sensor_config)

        LOGGER.info("Generating normal dataset...")
        normal = generate_normal_rows(args.rows_normal, sensor_config, rng)

        LOGGER.info("Generating mixed dataset with anomalies...")
        mixed, anomaly_indices = generate_mixed_rows(
            args.rows_mixed, args.anomaly_frac, sensor_config, rng
        )

        meta = {
            "contract": "contract_v1",
            "sensor_config": sensor_config,
            "columns": COLUMNS,
            "units": {
                "duty_cycle": "percent",
                "env": "C",
                "vib": "g",
                "other": "unit",
            },
            "mixed_anomaly_fraction": args.anomaly_frac,
            "mixed_anomaly_indices": anomaly_indices,
        }

        out_dir = Path(args.out_dir)
        save_outputs(out_dir, normal, mixed, meta)
        LOGGER.info("Saved outputs to %s", out_dir)
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to generate datasets: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
