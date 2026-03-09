import argparse
import sys
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


def die(msg: str) -> None:
    print(f"[VALIDATION ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-8, help="tolerance for padded zeros")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # ---- Schema check
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        die(f"Missing columns: {missing}. Expected {ALL_COLS}")

    df = df[ALL_COLS].copy()

    # ---- Numeric + finite check
    for c in ALL_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            die(f"Column '{c}' is not numeric.")
        if not np.isfinite(df[c].to_numpy()).all():
            die(f"Column '{c}' contains NaN/inf.")

    # ---- Mask must be 0/1 integers
    for mc in MASK_COLS:
        vals = df[mc].unique()
        if not set(vals).issubset({0, 1}):
            die(f"Mask column '{mc}' has values {sorted(vals)}; must be only 0/1.")

    # ---- Padding must be zero when slot inactive
    tol = args.tolerance
    for slot, idxs in SLOT_TO_IDXS.items():
        mcol = f"m{slot}"
        fcols = [f"f{i}" for i in idxs]
        inactive = df[mcol] == 0
        if inactive.any():
            max_abs = df.loc[inactive, fcols].abs().to_numpy().max()
            if max_abs > tol:
                die(
                    f"Slot {slot} inactive but has non-zero padded features. "
                    f"Max abs={max_abs} on cols {fcols}. "
                    f"Fix data generation to zero-pad inactive slots."
                )

    # ---- Range sanity
    # duty_cycle in [0,1] where active
    active0 = df["m0"] == 1
    if active0.any():
        v = df.loc[active0, "f0"]
        if (v < -tol).any() or (v > 1 + tol).any():
            die("Slot0 duty_cycle f0 out of [0,1] range.")

    # std features non-negative
    for std_col in ["f2", "f6", "f10"]:
        if (df[std_col] < -tol).any():
            die(f"Std column {std_col} has negative values.")

    # min <= mean <= max for slot1,2,3
    groups = {
        "slot1": ("m1", "f1", "f3", "f4"),  # mean, min, max
        "slot2": ("m2", "f5", "f7", "f8"),
        "slot3": ("m3", "f9", "f11", "f12"),
    }
    for name, (mcol, mean_c, min_c, max_c) in groups.items():
        active = df[mcol] == 1
        if not active.any():
            continue
        mean = df.loc[active, mean_c]
        mn = df.loc[active, min_c]
        mx = df.loc[active, max_c]
        if (mn > mx + tol).any():
            die(f"{name}: found min > max in active rows.")
        if ((mean < mn - tol) | (mean > mx + tol)).any():
            die(f"{name}: found mean outside [min,max] in active rows.")

    print(f"[OK] Validation passed for {args.csv}. Rows={len(df)} Columns={len(df.columns)}")


if __name__ == "__main__":
    main()