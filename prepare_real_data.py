import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

def _clip_minmax_consistency(mean: float, std: float, mn: float, mx: float) -> Tuple[float, float, float, float]:
    if mn > mx:
        mn, mx = mx, mn
    mean = float(np.clip(mean, mn, mx))
    std = float(max(std, 0.0))
    if mx - mn < 1e-6:
        mx = mn + 1e-3
    return mean, std, mn, mx

def inject_anomaly(rng: np.random.Generator, feats: List[float], mask: List[int]) -> Tuple[List[float], str]:
    feats = feats.copy()
    possible = []
    if mask[0] == 1: possible.append("temp_jump")
    if mask[1] == 1: possible.append("hum_jump")
    if mask[2] == 1: possible.append("acc_spike")
    if mask[3] == 1: possible.append("dist_outlier")

    if not possible:
        return feats, "none"

    kind = rng.choice(possible)

    if kind == "temp_jump":
        # Severe: +20 to +40 degrees above normal (~31°C baseline)
        mean, std, mn, mx = feats[0:4]
        jump = rng.uniform(20.0, 40.0)
        mean += jump
        mx += jump
        std *= rng.uniform(3.0, 6.0)
        mn += jump * rng.uniform(0.5, 0.8)
        feats[0:4] = list(_clip_minmax_consistency(mean, std, mn, mx))

    elif kind == "hum_jump":
        # Severe: -60 to -30 drop (humidity baseline ~85%)
        mean, std, mn, mx = feats[4:8]
        jump = rng.uniform(-60.0, -30.0)
        mean += jump
        mn += jump
        mx += jump * rng.uniform(0.5, 0.8)
        std *= rng.uniform(3.0, 6.0)
        feats[4:8] = list(_clip_minmax_consistency(mean, std, mn, mx))

    elif kind == "acc_spike":
        # Severe: acceleration 5x to 10x higher than normal (~1.0 baseline)
        mean, std, mn, mx = feats[8:12]
        spike = rng.uniform(5.0, 10.0)
        mx += spike
        mean += spike * rng.uniform(0.5, 0.9)
        std *= rng.uniform(4.0, 8.0)
        mn = max(mn, 0.0)
        feats[8:12] = list(_clip_minmax_consistency(mean, std, mn, mx))

    elif kind == "dist_outlier":
        # Severe: distance jumps by +150 to +300 from baseline
        mean, std, mn, mx = feats[12:16]
        if rng.random() < 0.5:
            shift = rng.uniform(150.0, 300.0)
            mean += shift
            mx += shift
            mn += shift
        else:
            # stuck sensor: std collapses to 0
            mean = mean + rng.uniform(100.0, 200.0)
            std = 0.0
            mn = mx = mean
        feats[12:16] = list(_clip_minmax_consistency(mean, std, mn, mx))

    return feats, kind

def main():
    out_dir = Path("data_real_prepared")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CSVs
    print("Loading real CSVs...")
    df_am = pd.read_csv("data_real/AM2302.csv")
    df_gy = pd.read_csv("data_real/GY511.csv")
    df_ld = pd.read_csv("data_real/LD2410.csv")

    # 2. Merge on timestamp
    df = df_am.merge(df_gy, on="timestamp", how="outer").merge(df_ld, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 3. Create masks based on presence of values (not NaN)
    # AM2302 Temp
    m0 = (~df["temp_mean"].isna()).astype(int)
    # AM2302 Hum
    m1 = (~df["hum_mean"].isna()).astype(int)
    # GY511
    m2 = (~df["accel_mean"].isna()).astype(int)
    # LD2410
    m3 = (~df["dist_mean"].isna()).astype(int)

    # 4. Fill NaNs with 0 value
    df = df.fillna(0.0)

    # Map to f0..f15
    f_cols = [
        "temp_mean", "temp_std", "temp_min", "temp_max",
        "hum_mean", "hum_std", "hum_min", "hum_max",
        "accel_mean", "accel_std", "accel_min", "accel_max",
        "dist_mean", "dist_std", "dist_min", "dist_max"
    ]
    
    # Ensure dataframe has all these
    feats_array = df[f_cols].to_numpy()
    masks_array = np.column_stack([m0, m1, m2, m3])

    feature_cols = [f"f{i}" for i in range(16)]
    mask_cols = [f"m{i}" for i in range(4)]
    all_cols = feature_cols + mask_cols

    # Create df_normal
    normal_data = np.hstack([feats_array, masks_array])
    df_normal = pd.DataFrame(normal_data, columns=all_cols)

    # Convert mask cols to integer
    for mc in mask_cols:
        df_normal[mc] = df_normal[mc].astype(int)

    # Create mixed (all normal + inserted anomalies)
    rng = np.random.default_rng(42)
    n_anom = 200 # number of anomalies to inject
    rows_m = []
    labels = []

    # Add normal rows
    for i in range(len(df_normal)):
        rows_m.append(df_normal.iloc[i].tolist())
        labels.append("normal")

    # Generate anomaly rows by assigning random ones from normal data
    valid_indices = df_normal.index[df_normal[mask_cols].sum(axis=1) > 0].tolist()
    if valid_indices:
        for _ in range(n_anom):
            idx = rng.choice(valid_indices)
            row = df_normal.iloc[idx].copy()
            feats = row[feature_cols].tolist()
            mask = row[mask_cols].astype(int).tolist()
            feats2, kind = inject_anomaly(rng, feats, mask)
            rows_m.append(feats2 + mask)
            labels.append(kind)
    else:
        print("Warning: no active sensors to inject anomalies into.")

    df_mixed = pd.DataFrame(rows_m, columns=all_cols)
    df_mixed.insert(0, "label", labels)
    for mc in mask_cols:
        df_mixed[mc] = df_mixed[mc].astype(int)

    # Save CSVs
    normal_path = out_dir / "real_normal.csv"
    mixed_path = out_dir / "real_mixed.csv"
    
    df_normal.to_csv(normal_path, index=False)
    df_mixed.to_csv(mixed_path, index=False)

    # Save meta
    meta = {
        "contract_version": "v2",
        "window_seconds": 1.0,
        "padding_value": 0,
        "features": {
            "n_features": 16,
            "feature_cols": feature_cols,
            "slot_to_feature_idxs": {
                "0": [0, 1, 2, 3],
                "1": [4, 5, 6, 7],
                "2": [8, 9, 10, 11],
                "3": [12, 13, 14, 15]
            },
            "slot_definitions": {
                "0": "AM2302 Temp (mean,std,min,max)",
                "1": "AM2302 Hum (mean,std,min,max)",
                "2": "GY511 Accel (mean,std,min,max)",
                "3": "LD2410 Dist (mean,std,min,max)"
            }
        },
        "mask": {
            "n_slots": 4,
            "mask_cols": mask_cols,
            "meaning": "m=1 if slot active else 0"
        }
    }
    (out_dir / "real_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Created {out_dir}")
    print(f"Saved {normal_path} ({len(df_normal)} rows)")
    print(f"Saved {mixed_path} ({len(df_mixed)} rows)")
    print(f"Saved {out_dir / 'real_meta.json'}")

if __name__ == "__main__":
    main()
