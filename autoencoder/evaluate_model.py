import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Contract v1 mapping
SLOT_TO_IDXS = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
}
FEATURE_COLS = [f"f{i}" for i in range(13)]
MASK_COLS = [f"m{i}" for i in range(4)]
ALL_COLS = FEATURE_COLS + MASK_COLS


def mask_pattern(m_row: np.ndarray) -> str:
    return "".join(str(int(v)) for v in m_row.tolist())


def masked_mse_batch(x: np.ndarray, xhat: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Compute mask-aware MSE for each row."""
    N = x.shape[0]
    mse = np.zeros((N,), dtype=np.float32)

    slot_feat_masks = []
    for slot in range(4):
        mask = np.zeros((13,), dtype=bool)
        mask[SLOT_TO_IDXS[slot]] = True
        slot_feat_masks.append(mask)

    for i in range(N):
        active = m[i].astype(bool)
        feat_mask = np.zeros((13,), dtype=bool)
        for slot in range(4):
            if active[slot]:
                feat_mask |= slot_feat_masks[slot]

        if not np.any(feat_mask):
            mse[i] = 0.0
            continue

        diff = xhat[i, feat_mask] - x[i, feat_mask]
        mse[i] = float(np.mean(diff * diff))

    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data_mock/mock_mixed.csv", help="mixed dataset to test on")
    ap.add_argument("--stats_json", default="data_mock/norm_stats.json", help="normalization stats")
    ap.add_argument("--model_path", default="data_mock/model_output/autoencoder.keras", help="trained model")
    ap.add_argument("--thresholds_json", default="data_mock/model_output/thresholds.json", help="thresholds")
    args = ap.parse_args()

    print("Loading data and model...")
    df = pd.read_csv(args.test_csv)
    
    # Extract labels and true binary anomaly status
    y_true_labels = df["label"].values
    y_true_binary = (y_true_labels != "normal").astype(int)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    M = df[MASK_COLS].to_numpy(dtype=np.int32)

    # 1. Apply Normalization using saved stats
    stats = json.loads(Path(args.stats_json).read_text())
    means = np.array(stats["means"], dtype=np.float32)
    stds = np.array(stats["stds"], dtype=np.float32)

    for slot, idxs in SLOT_TO_IDXS.items():
        active_rows = M[:, slot] == 1
        if not np.any(active_rows):
            continue
        X[active_rows][:, idxs] = (X[active_rows][:, idxs] - means[idxs]) / stds[idxs]
        
        # Ensure inactive are 0
        inactive_rows = ~active_rows
        X[inactive_rows][:, idxs] = 0.0

    # 2. Run Inference
    model = tf.keras.models.load_model(args.model_path)
    X_hat = model.predict(X, batch_size=256, verbose=0)
    
    # 3. Calculate MSE
    mse = masked_mse_batch(X, X_hat, M)

    # 4. Apply Thresholds to classify
    thresh_data = json.loads(Path(args.thresholds_json).read_text())["thresholds_by_mask"]
    
    y_pred_binary = np.zeros(len(X), dtype=int)
    patterns = np.array([mask_pattern(M[i]) for i in range(len(M))])
    
    for p in np.unique(patterns):
        if p not in thresh_data:
            print(f"Warning: Mask {p} not found in thresholds! Skipping.")
            continue
            
        thr = thresh_data[p]
        mask_idx = (patterns == p)
        # Anomaly if mse > threshold
        y_pred_binary[mask_idx] = (mse[mask_idx] > thr).astype(int)

    # 5. Report Results
    print("\n--- Evaluation Results ---")
    print(classification_report(y_true_binary, y_pred_binary, target_names=["Normal (0)", "Anomaly (1)"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print(f"                Predicted Normal  | Predicted Anomaly")
    print(f"Actual Normal   | {cm[0][0]:<15} | {cm[0][1]}")
    print(f"Actual Anomaly  | {cm[1][0]:<15} | {cm[1][1]}")
    
    # Let's see how well it caught different types of anomalies
    print("\nDetection rates by anomaly type:")
    for label_type in np.unique(y_true_labels):
        if label_type == "normal":
            continue
        idx = (y_true_labels == label_type)
        acc = np.mean(y_pred_binary[idx] == 1) * 100
        print(f"  {label_type}: {acc:.1f}% caught ({np.sum(y_pred_binary[idx])}/{np.sum(idx)})")

if __name__ == "__main__":
    main()
