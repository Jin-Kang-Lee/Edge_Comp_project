#This is for the Mask-aware z-score
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLOT_TO_IDXS = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}

FEATURE_COLS = [f"f{i}" for i in range(13)]
MASK_COLS = [f"m{i}" for i in range(4)]
ALL_COLS = FEATURE_COLS + MASK_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_stats", required=True)
    ap.add_argument("--std_floor", type=float, default=1e-6, help="avoid division by tiny std")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)[ALL_COLS].copy()

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    M = df[MASK_COLS].to_numpy(dtype=np.int32)

    means = np.zeros(13, dtype=np.float32)
    stds = np.ones(13, dtype=np.float32)

    # Compute per-feature mean/std, but only on rows where the slot is active
    for slot, idxs in SLOT_TO_IDXS.items():
        active_rows = M[:, slot] == 1
        if not np.any(active_rows):
            # slot never active in this dataset; keep mean=0,std=1 (features remain 0 anyway)
            continue

        slot_data = X[active_rows][:, idxs]  # (n_active, slot_dim)
        mu = slot_data.mean(axis=0)
        sd = slot_data.std(axis=0)
        sd = np.where(sd < args.std_floor, 1.0, sd)

        means[idxs] = mu
        stds[idxs] = sd

        # Apply normalization only on active rows
        X[active_rows][:, idxs] = (X[active_rows][:, idxs] - mu) / sd

        # Ensure inactive rows stay exactly 0
        inactive_rows = ~active_rows
        X[inactive_rows][:, idxs] = 0.0

    out_df = pd.DataFrame(X, columns=FEATURE_COLS)
    for i, mc in enumerate(MASK_COLS):
        out_df[mc] = M[:, i]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    stats = {
        "contract_version": "v1",
        "feature_cols": FEATURE_COLS,
        "mask_cols": MASK_COLS,
        "slot_to_feature_idxs": SLOT_TO_IDXS,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "std_floor": args.std_floor,
    }
    Path(args.out_stats).write_text(json.dumps(stats, indent=2))

    print(f"Wrote normalized CSV: {args.out_csv}")
    print(f"Wrote stats JSON:     {args.out_stats}")


if __name__ == "__main__":
    main()