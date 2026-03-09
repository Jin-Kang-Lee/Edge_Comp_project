import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class TqdmEpochCallback(tf.keras.callbacks.Callback):
    """Displays a clean tqdm progress bar — one bar per epoch."""

    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]
        self.pbar = tqdm(
            total=self.epochs,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            colour="cyan",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        self.pbar.set_postfix(
            loss=f"{loss:.6f}",
            val_loss=f"{val_loss:.6f}",
        )
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()
        print("\n[Training complete]")


# Contract v1 mapping (same as your validator/normalizer)
SLOT_TO_IDXS = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
}

FEATURE_COLS = [f"f{i}" for i in range(16)]
MASK_COLS = [f"m{i}" for i in range(4)]
ALL_COLS = FEATURE_COLS + MASK_COLS


def build_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    MLP autoencoder for 16-feature real sensor data.
    - Encoder: 64 -> 32 -> bottleneck(8)
    - Decoder: 8  -> 32 -> 64 -> output
    Bottleneck of 8 is the Goldilocks zone: big enough to faithfully
    learn the 4-slot sensor patterns, tight enough that anomalous
    inputs cannot be reconstructed accurately.
    """
    inp = tf.keras.Input(shape=(input_dim,), name="x")
    x = tf.keras.layers.Dense(64, activation="relu")(inp)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    z = tf.keras.layers.Dense(8, activation="relu", name="latent")(x)  # bottleneck
    x = tf.keras.layers.Dense(32, activation="relu")(z)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(input_dim, activation=None, name="x_hat")(x)

    model = tf.keras.Model(inp, out, name="ae_mlp_v2")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def mask_pattern(m_row: np.ndarray) -> str:
    # m_row is shape (4,), values 0/1
    return "".join(str(int(v)) for v in m_row.tolist())


def masked_mse_batch(x: np.ndarray, xhat: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Compute mask-aware MSE for each row.
    - x, xhat: (N, 16)
    - m: (N, 4)
    Only include features from active slots per row.
    Returns: (N,) mse values
    """
    N = x.shape[0]
    mse = np.zeros((N,), dtype=np.float32)

    # Precompute per-slot feature indices as boolean masks (length 16)
    slot_feat_masks = []
    for slot in range(4):
        mask = np.zeros((16,), dtype=bool)
        mask[SLOT_TO_IDXS[slot]] = True
        slot_feat_masks.append(mask)

    for i in range(N):
        active = m[i].astype(bool)
        feat_mask = np.zeros((16,), dtype=bool)
        for slot in range(4):
            if active[slot]:
                feat_mask |= slot_feat_masks[slot]

        # Safety: if no slots active, mse stays 0
        if not np.any(feat_mask):
            mse[i] = 0.0
            continue

        diff = xhat[i, feat_mask] - x[i, feat_mask]
        mse[i] = float(np.mean(diff * diff))

    return mse


def compute_thresholds_by_mask(mse: np.ndarray, M: np.ndarray, percentile: float) -> Dict[str, float]:
    thresholds = {}
    patterns = np.array([mask_pattern(M[i]) for i in range(len(M))])

    for p in np.unique(patterns):
        vals = mse[patterns == p]
        thr = float(np.percentile(vals, percentile))
        thresholds[p] = thr

    return thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="normalized training csv (normal-only)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)  # bumped from 50
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threshold_percentile", type=float, default=95.0)  # lowered from 99.5
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)[ALL_COLS]
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    M = df[MASK_COLS].to_numpy(dtype=np.int32)

    # Train/val split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X = X[idx]
    M = M[idx]

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]

    model = build_autoencoder(input_dim=16)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        TqdmEpochCallback(),
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=0,  # suppress per-batch output; tqdm callback handles display
    )

    # Reconstruction + mask-aware MSE
    X_hat = model.predict(X, batch_size=args.batch, verbose=0)
    mse = masked_mse_batch(X, X_hat, M)

    thresholds = compute_thresholds_by_mask(mse, M, args.threshold_percentile)

    # Save model
    model_path = out_dir / "autoencoder.keras"
    model.save(model_path)

    # Save thresholds + report
    (out_dir / "thresholds.json").write_text(json.dumps({
        "threshold_percentile": args.threshold_percentile,
        "thresholds_by_mask": thresholds
    }, indent=2))

    report = {
        "rows": int(len(X)),
        "train_loss_last": float(history.history["loss"][-1]),
        "val_loss_last": float(history.history["val_loss"][-1]),
        "mse_mean": float(np.mean(mse)),
        "mse_p95": float(np.percentile(mse, 95)),
        "mse_p99": float(np.percentile(mse, 99)),
        "mse_p995": float(np.percentile(mse, 99.5)),
        "mse_max": float(np.max(mse)),
        "unique_masks": sorted(list(thresholds.keys())),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))

    print("Saved:")
    print(f"  {model_path}")
    print(f"  {out_dir / 'thresholds.json'}")
    print(f"  {out_dir / 'train_report.json'}")
    print("Thresholds by mask:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()