#!/usr/bin/env python3
"""Generate synthetic contract_v1 datasets for pipeline smoke testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Contract v1 feature layout (13 features + 4 mask columns)
FEATURE_COLUMNS = [
    "duty_cycle",
    "env_mean",
    "env_std",
    "env_min",
    "env_max",
    "acc_mean_abs",
    "acc_std_abs",
    "acc_min_abs",
    "acc_max_abs",
    "other_mean",
    "other_std",
    "other_min",
    "other_max",
]
MASK_COLUMNS = ["m0", "m1", "m2", "m3"]
ALL_COLUMNS = FEATURE_COLUMNS + MASK_COLUMNS

# Slot -> feature indices in FEATURE_COLUMNS
SLOT_FEATURES: Dict[int, List[int]] = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}


def _sample_normal_features(rng: np.random.Generator) -> np.ndarray:
    """Sample one normal row of 13 features before mask padding."""
    duty = np.clip(rng.normal(0.35, 0.15), 0.0, 1.0)

    env_mean = rng.normal(24.0, 1.0)
    env_std = np.clip(rng.normal(1.2, 0.25), 0.05, 5.0)
    env_min = env_mean - abs(rng.normal(2.0, 0.5))
    env_max = env_mean + abs(rng.normal(2.3, 0.6))

    acc_mean = np.clip(rng.normal(0.40, 0.12), 0.0, 3.0)
    acc_std = np.clip(rng.normal(0.08, 0.03), 0.005, 1.0)
    acc_min = max(0.0, acc_mean - abs(rng.normal(0.11, 0.04)))
    acc_max = acc_mean + abs(rng.normal(0.13, 0.05))

    other_mean = rng.normal(100.0, 8.0)
    other_std = np.clip(rng.normal(4.0, 1.0), 0.1, 30.0)
    other_min = other_mean - abs(rng.normal(7.0, 2.0))
    other_max = other_mean + abs(rng.normal(7.5, 2.5))

    return np.array(
        [
            duty,
            env_mean,
            env_std,
            env_min,
            env_max,
            acc_mean,
            acc_std,
            acc_min,
            acc_max,
            other_mean,
            other_std,
            other_min,
            other_max,
        ],
        dtype=np.float32,
    )


def _sample_mask(rng: np.random.Generator) -> np.ndarray:
    """Create a realistic slot mask; at least one slot active."""
    probs = np.array([0.95, 0.90, 0.85, 0.90])
    mask = (rng.random(4) < probs).astype(np.int32)
    if not mask.any():
        mask[rng.integers(0, 4)] = 1
    return mask


def _apply_mask_padding(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pad inactive slot features with exact zeros."""
    padded = features.copy()
    for slot, idxs in SLOT_FEATURES.items():
        if mask[slot] == 0:
            padded[idxs] = 0.0
    return padded


def _build_normal_rows(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.zeros((n_rows, 17), dtype=np.float32)
    for i in range(n_rows):
        feat = _sample_normal_features(rng)
        mask = _sample_mask(rng)
        feat = _apply_mask_padding(feat, mask)
        rows[i, :13] = feat
        rows[i, 13:] = mask
    return rows


def _inject_spike(row_features: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a spike anomaly into one active slot feature."""
    active_slots = np.flatnonzero(mask)
    if active_slots.size == 0:
        return row_features
    slot = int(rng.choice(active_slots))
    idx = int(rng.choice(SLOT_FEATURES[slot]))
    scale = rng.uniform(4.0, 8.0)
    sign = rng.choice([-1.0, 1.0])
    row_features[idx] += sign * scale * max(1.0, abs(float(row_features[idx])))
    return row_features


def _inject_stuck_block(data: np.ndarray, start: int, length: int, slot: int) -> None:
    """Force selected slot features to remain constant for a block."""
    end = min(start + length, data.shape[0])
    idxs = SLOT_FEATURES[slot]
    reference = data[start, idxs].copy()
    for i in range(start, end):
        if data[i, 13 + slot] == 1:
            data[i, idxs] = reference


def _build_mixed_rows(n_rows: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, dict]:
    data = _build_normal_rows(n_rows, rng)
    labels = np.zeros(n_rows, dtype=np.int32)

    # Random spikes (~8% rows)
    spike_count = int(0.08 * n_rows)
    spike_idx = rng.choice(np.arange(10, n_rows), size=spike_count, replace=False)
    for idx in spike_idx:
        data[idx, :13] = _inject_spike(data[idx, :13], data[idx, 13:].astype(np.int32), rng)
        labels[idx] = 1

    # Stuck sensor blocks (~10 blocks)
    blocks = []
    for _ in range(10):
        start = int(rng.integers(20, n_rows - 20))
        length = int(rng.integers(6, 16))
        slot = int(rng.integers(0, 4))
        _inject_stuck_block(data, start, length, slot)
        labels[start : min(start + length, n_rows)] = 1
        blocks.append({"start": start, "length": length, "slot": slot})

    meta = {
        "mixed_labels": labels.tolist(),
        "spike_indices": sorted([int(i) for i in spike_idx]),
        "stuck_blocks": blocks,
    }
    return data, labels, meta


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    normal = _build_normal_rows(n_rows=5000, rng=rng)
    mixed, mixed_labels, mixed_meta = _build_mixed_rows(n_rows=2000, rng=rng)

    normal_df = pd.DataFrame(normal, columns=ALL_COLUMNS)
    mixed_df = pd.DataFrame(mixed, columns=ALL_COLUMNS)

    normal_path = data_dir / "mock_normal.csv"
    mixed_path = data_dir / "mock_mixed.csv"
    meta_path = data_dir / "mock_meta.json"

    normal_df.to_csv(normal_path, index=False)
    mixed_df.to_csv(mixed_path, index=False)

    meta = {
        "contract": "contract_v1",
        "feature_columns": FEATURE_COLUMNS,
        "mask_columns": MASK_COLUMNS,
        "slot_features": {str(k): v for k, v in SLOT_FEATURES.items()},
        "mock_normal_rows": int(normal.shape[0]),
        "mock_mixed_rows": int(mixed.shape[0]),
        "mock_mixed_anomaly_count": int(mixed_labels.sum()),
        **mixed_meta,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {normal_path}")
    print(f"Saved {mixed_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
