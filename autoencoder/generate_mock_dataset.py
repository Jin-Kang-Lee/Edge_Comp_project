import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Contract v1 (Fixed mapping)
# -----------------------------
# Slot 0 (Motion/Presence, binary duty_cycle): f0
# Slot 1 (Environmental scalar: mean,std,min,max): f1..f4
# Slot 2 (Accel magnitude: mean,std,min,max): f5..f8
# Slot 3 (Other scalar: mean,std,min,max): f9..f12
SLOT_TO_FEATURE_IDXS: Dict[int, List[int]] = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}

FEATURE_COLS = [f"f{i}" for i in range(13)]
MASK_COLS = [f"m{i}" for i in range(4)]
ALL_COLS = FEATURE_COLS + MASK_COLS


@dataclass
class Config:
    n_normal: int
    n_anom: int
    seed: int


def _clip_minmax_consistency(mean: float, std: float, mn: float, mx: float) -> Tuple[float, float, float, float]:
    # Ensure min <= mean <= max and std >= 0; also enforce a small margin
    if mn > mx:
        mn, mx = mx, mn
    mean = float(np.clip(mean, mn, mx))
    std = float(max(std, 0.0))
    if mx - mn < 1e-6:
        mx = mn + 1e-3
    return mean, std, mn, mx


def gen_slot0_motion(rng: np.random.Generator) -> float:
    """
    Motion duty_cycle in [0,1].
    Normal: mostly near 0 with occasional small activity.
    """
    # 85% near 0, 15% small activity
    if rng.random() < 0.85:
        dc = rng.beta(1.0, 25.0)  # heavily skewed to 0
    else:
        dc = rng.beta(2.0, 8.0)   # small activity
    return float(np.clip(dc, 0.0, 1.0))


def gen_slot1_env(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """
    Environmental scalar window stats (mean,std,min,max).
    Normal: around 25 with mild variation.
    """
    mean = rng.normal(25.0, 0.6)
    std = abs(rng.normal(0.15, 0.05))
    # min/max around mean +/- k*std
    mn = mean - rng.uniform(1.0, 2.5) * max(std, 0.05)
    mx = mean + rng.uniform(1.0, 2.5) * max(std, 0.05)
    return _clip_minmax_consistency(mean, std, mn, mx)


def gen_slot2_accmag(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """
    Accelerometer magnitude window stats (mean,std,min,max).
    Normal: small vibrations around ~0.5 (arbitrary units).
    """
    mean = rng.normal(0.50, 0.05)
    std = abs(rng.normal(0.05, 0.02))
    mn = mean - rng.uniform(1.0, 3.0) * max(std, 0.01)
    mx = mean + rng.uniform(1.0, 3.0) * max(std, 0.01)
    # Acc magnitude shouldn't be negative
    mn = max(mn, 0.0)
    return _clip_minmax_consistency(mean, std, mn, mx)


def gen_slot3_other(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """
    Other scalar window stats (mean,std,min,max).
    Normal: stable around 100 with mild noise (e.g., distance mm, light lux, etc.)
    """
    mean = rng.normal(100.0, 5.0)
    std = abs(rng.normal(2.0, 0.7))
    mn = mean - rng.uniform(1.0, 2.5) * max(std, 0.5)
    mx = mean + rng.uniform(1.0, 2.5) * max(std, 0.5)
    return _clip_minmax_consistency(mean, std, mn, mx)


def make_normal_row(rng: np.random.Generator, mask: List[int]) -> List[float]:
    feats = [0.0] * 13

    # Slot 0
    if mask[0] == 1:
        feats[0] = gen_slot0_motion(rng)

    # Slot 1
    if mask[1] == 1:
        feats[1:5] = list(gen_slot1_env(rng))

    # Slot 2
    if mask[2] == 1:
        feats[5:9] = list(gen_slot2_accmag(rng))

    # Slot 3
    if mask[3] == 1:
        feats[9:13] = list(gen_slot3_other(rng))

    return feats


def inject_anomaly(rng: np.random.Generator, feats: List[float], mask: List[int]) -> Tuple[List[float], str]:
    """
    Inject one of several anomaly types on active slots.
    Returns (new_feats, anomaly_type)
    """
    feats = feats.copy()

    possible = []
    if mask[0] == 1:
        possible.append("motion_stuck_high")
    if mask[1] == 1:
        possible.append("env_jump")
    if mask[2] == 1:
        possible.append("acc_spike")
    if mask[3] == 1:
        possible.append("other_outlier")

    if not possible:
        return feats, "none"  # should not happen if at least one slot active

    kind = rng.choice(possible)

    if kind == "motion_stuck_high":
        feats[0] = float(rng.uniform(0.9, 1.0))

    elif kind == "env_jump":
        # Blow up mean/max dramatically
        mean, std, mn, mx = feats[1:5]
        jump = rng.uniform(5.0, 12.0)
        mean = mean + jump
        mx = mx + jump
        # optionally increase std too
        std = std * rng.uniform(2.0, 5.0)
        mn = mn + jump * rng.uniform(0.3, 0.7)
        feats[1:5] = list(_clip_minmax_consistency(mean, std, mn, mx))

    elif kind == "acc_spike":
        mean, std, mn, mx = feats[5:9]
        spike = rng.uniform(0.5, 2.0)  # big bump
        mx = mx + spike
        mean = mean + spike * rng.uniform(0.2, 0.6)
        std = std * rng.uniform(2.0, 6.0)
        mn = max(mn - spike * rng.uniform(0.0, 0.3), 0.0)
        feats[5:9] = list(_clip_minmax_consistency(mean, std, mn, mx))

    elif kind == "other_outlier":
        mean, std, mn, mx = feats[9:13]
        # Extreme shift or collapse
        if rng.random() < 0.5:
            shift = rng.uniform(40.0, 120.0)
            mean += shift
            mx += shift
            mn += shift
        else:
            # sensor stuck (std ~ 0, min=max=mean)
            mean = mean + rng.uniform(-30.0, 30.0)
            std = 0.0
            mn = mean
            mx = mean
        feats[9:13] = list(_clip_minmax_consistency(mean, std, mn, mx))

    return feats, kind


def gen_dataset(cfg: Config, masks: List[List[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    # Normal-only
    rows_n = []
    for _ in range(cfg.n_normal):
        mask = masks[rng.integers(0, len(masks))].copy()
        feats = make_normal_row(rng, mask)
        rows_n.append(feats + mask)

    df_normal = pd.DataFrame(rows_n, columns=ALL_COLS)

    # Mixed (normal + anomalies)
    rows_m = []
    labels = []
    for _ in range(cfg.n_normal):
        mask = masks[rng.integers(0, len(masks))].copy()
        feats = make_normal_row(rng, mask)
        rows_m.append(feats + mask)
        labels.append("normal")

    for _ in range(cfg.n_anom):
        mask = masks[rng.integers(0, len(masks))].copy()
        feats = make_normal_row(rng, mask)
        feats2, kind = inject_anomaly(rng, feats, mask)
        rows_m.append(feats2 + mask)
        labels.append(kind)

    df_mixed = pd.DataFrame(rows_m, columns=ALL_COLS)
    df_mixed.insert(0, "label", labels)

    return df_normal, df_mixed


def write_meta(out_dir: Path, masks: List[List[int]]):
    meta = {
        "contract_version": "v1",
        "window_seconds": 1.0,
        "padding_value": 0,
        "features": {
            "n_features": 13,
            "feature_cols": FEATURE_COLS,
            "slot_to_feature_idxs": SLOT_TO_FEATURE_IDXS,
            "slot_definitions": {
                "0": "Motion/Presence (binary duty_cycle)",
                "1": "Environmental Scalar (mean,std,min,max)",
                "2": "Inertial/Vibration (accel magnitude mean,std,min,max)",
                "3": "Other Scalar (mean,std,min,max)",
            },
        },
        "mask": {
            "n_slots": 4,
            "mask_cols": MASK_COLS,
            "meaning": "m=1 if slot active else 0",
            "example_masks_used": masks,
        },
    }
    (out_dir / "mock_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_mock", help="output directory")
    ap.add_argument("--n_normal", type=int, default=5000)
    ap.add_argument("--n_anom", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--masks",
        default="1111,0111,0011,1100",
        help="Comma-separated slot masks, e.g. 1111,0111",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for s in args.masks.split(","):
        s = s.strip()
        if len(s) != 4 or any(c not in "01" for c in s):
            raise ValueError(f"Invalid mask '{s}'. Use 4 chars of 0/1 like 1110.")
        masks.append([int(c) for c in s])

    cfg = Config(n_normal=args.n_normal, n_anom=args.n_anom, seed=args.seed)

    df_normal, df_mixed = gen_dataset(cfg, masks)

    normal_path = out_dir / "mock_normal.csv"
    mixed_path = out_dir / "mock_mixed.csv"

    df_normal.to_csv(normal_path, index=False)
    df_mixed.to_csv(mixed_path, index=False)

    write_meta(out_dir, masks)

    print("Wrote:")
    print(f"  {normal_path}  (rows={len(df_normal)})")
    print(f"  {mixed_path}   (rows={len(df_mixed)})")
    print(f"  {out_dir / 'mock_meta.json'}")


if __name__ == "__main__":
    main()