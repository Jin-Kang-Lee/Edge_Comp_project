import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402

from pipeline import (
    RAW_GROUPS,
    MASK_NAMES,
    ContractSchema,
    add_masks,
    apply_scaler,
    build_contract_v1,
    compute_scaler,
    inject_anomalies,
    make_windows,
    make_windows_multi,
    merge_sensors,
    save_json,
    select_features_by_variance,
    split_time_series,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def configure_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        return "GPU"
    return "CPU"


def parse_anom_type_weights(raw: str) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(value)
        except ValueError:
            continue
    return weights or None


def build_mlp_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_linear_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(n_features)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def build_cnn_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_mlp_multi_model(window: int, n_features: int, horizon: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon * n_features)(x)
    outputs = tf.keras.layers.Reshape((horizon, n_features))(outputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_linear_multi_model(window: int, n_features: int, horizon: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(horizon * n_features)(x)
    outputs = tf.keras.layers.Reshape((horizon, n_features))(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model


def build_cnn_multi_model(window: int, n_features: int, horizon: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(horizon * n_features)(x)
    outputs = tf.keras.layers.Reshape((horizon, n_features))(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_linear_ridge(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    X_flat = X.reshape(len(X), -1).astype(np.float64)
    y = y.astype(np.float64)
    if y.ndim == 3:
        y = y.reshape(len(y), -1)
    ones = np.ones((X_flat.shape[0], 1), dtype=np.float64)
    X_aug = np.concatenate([X_flat, ones], axis=1)
    reg = np.eye(X_aug.shape[1], dtype=np.float64) * np.sqrt(l2)
    reg[-1, -1] = 0.0
    A = np.concatenate([X_aug, reg], axis=0)
    B = np.concatenate([y, np.zeros((reg.shape[0], y.shape[1]), dtype=np.float64)], axis=0)
    w_aug, *_ = np.linalg.lstsq(A, B, rcond=None)
    w = w_aug[:-1].astype(np.float32)
    b = w_aug[-1].astype(np.float32)
    return w, b


def mse_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.square(y_true - y_pred), axis=1)


def masked_mse_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    err = np.square(y_true - y_pred) * mask
    denom = np.maximum(mask.sum(axis=1), 1e-6)
    return err.sum(axis=1) / denom


def masked_mse_scores_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    horizon: int,
) -> np.ndarray:
    # y_* shape: [N, horizon, features], mask shape: [N, horizon, features]
    err = np.square(y_true - y_pred) * mask
    denom = np.maximum(mask.sum(axis=(1, 2)), 1e-6)
    return err.sum(axis=(1, 2)) / denom


def compute_error_scale(
    y_true: np.ndarray, y_pred: np.ndarray, method: str
) -> np.ndarray:
    resid = y_true - y_pred
    if resid.ndim == 3:
        axis = (0, 1)
    else:
        axis = 0
    if method == "std":
        scale = np.std(resid, axis=axis)
    else:
        scale = np.median(np.abs(resid), axis=axis)
    scale = np.asarray(scale, dtype=np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale = np.maximum(scale, 1e-6)
    return scale


def compute_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    agg: str = "mean",
    topk: int = 3,
    error_scale: np.ndarray | None = None,
) -> np.ndarray:
    err = y_true - y_pred
    if error_scale is not None:
        err = err / error_scale
    err = np.square(err)
    mask_flat = mask
    if mask is not None:
        err = err * mask
    if err.ndim == 3:
        err = err.reshape(len(err), -1)
        if mask_flat is not None:
            mask_flat = mask_flat.reshape(len(mask_flat), -1)
    if agg == "mean":
        if mask_flat is not None:
            denom = np.maximum(mask_flat.sum(axis=1), 1e-6)
            return err.sum(axis=1) / denom
        return err.mean(axis=1)
    if mask_flat is not None:
        err = err.copy()
        err[mask_flat <= 0] = -np.inf
    if agg == "max":
        scores = np.max(err, axis=1)
    else:
        k = max(1, min(int(topk), err.shape[1]))
        part = np.partition(err, -k, axis=1)[:, -k:]
        scores = np.mean(part, axis=1)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    window = min(window, len(scores))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(scores, kernel, mode="same")


def apply_mask_penalty(
    scores: np.ndarray, mask: np.ndarray, weight: float
) -> np.ndarray:
    if weight <= 0.0:
        return scores
    if mask.ndim == 2:
        penalty = (1.0 - mask).mean(axis=1)
    else:
        penalty = (1.0 - mask).mean(axis=(1, 2))
    return scores + weight * penalty


def split_fraction(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = max(0.0, min(1.0, frac))
    if len(df) == 0:
        return df, df
    split_idx = int(len(df) * frac)
    split_idx = max(1, min(len(df) - 1, split_idx))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(
        drop=True
    )


def load_real_events(path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    df = pd.read_csv(path)
    start_col = None
    end_col = None
    for s_col, e_col in (
        ("start_ts", "end_ts"),
        ("start", "end"),
        ("start_time", "end_time"),
    ):
        if s_col in df.columns and e_col in df.columns:
            start_col = s_col
            end_col = e_col
            break
    if start_col is None or end_col is None:
        raise ValueError(
            "Real anomaly file must contain start/end columns (start_ts/end_ts or start/end)."
        )
    starts = pd.to_datetime(df[start_col], errors="coerce")
    ends = pd.to_datetime(df[end_col], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        raise ValueError("Invalid timestamps in real anomaly file.")
    events = list(zip(starts, ends))
    return events


def label_real_anomalies(df: pd.DataFrame, events: list[tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    labels = np.zeros(len(df), dtype=np.int8)
    for start, end in events:
        if end < start:
            start, end = end, start
        mask = (ts >= start) & (ts <= end)
        labels[mask.to_numpy()] = 1
    return labels


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.start_time = None

    def on_epoch_begin(self, epoch, logs=None):  # type: ignore[override]
        if self.enabled:
            self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        if not self.enabled:
            return
        elapsed = None
        if self.start_time is not None:
            elapsed = float(tf.timestamp() - self.start_time)
        loss = float(logs.get("loss")) if logs and logs.get("loss") is not None else None
        val_loss = (
            float(logs.get("val_loss"))
            if logs and logs.get("val_loss") is not None
            else None
        )
        loss_str = f"{loss:.4f}" if loss is not None else "n/a"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        elapsed_str = f"{elapsed:.3f}s" if elapsed is not None else "n/a"
        print(
            f"Epoch {epoch + 1} done | loss={loss_str} val_loss={val_loss_str} time={elapsed_str}",
            flush=True,
        )


def calibrate_threshold_fbeta(
    y_true: np.ndarray,
    scores: np.ndarray,
    beta: float,
    min_recall: float,
    min_anom_run: int = 1,
) -> float:
    if len(scores) == 0:
        return float("nan")
    thresholds = np.unique(scores)
    best_score = -1.0
    best_thr = float(thresholds[0])
    beta2 = beta * beta
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        y_pred = filter_short_runs(y_pred, min_anom_run)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall:
            continue
        if prec == 0 and rec == 0:
            fbeta = 0.0
        else:
            fbeta = (1.0 + beta2) * prec * rec / (beta2 * prec + rec)
        if fbeta > best_score:
            best_score = fbeta
            best_thr = float(thr)

    if best_score < 0:
        # Fallback: maximize F-beta without recall constraint.
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            y_pred = filter_short_runs(y_pred, min_anom_run)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if prec == 0 and rec == 0:
                fbeta = 0.0
            else:
                fbeta = (1.0 + beta2) * prec * rec / (beta2 * prec + rec)
            if fbeta > best_score:
                best_score = fbeta
                best_thr = float(thr)

    return best_thr


def calibrate_threshold_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_recall: float,
    min_anom_run: int = 1,
) -> float:
    if len(scores) == 0:
        return float("nan")
    thresholds = np.unique(scores)
    best_score = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        y_pred = filter_short_runs(y_pred, min_anom_run)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec < min_recall:
            continue
        acc = accuracy_score(y_true, y_pred)
        if acc > best_score:
            best_score = acc
            best_thr = float(thr)
    if best_score < 0:
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            y_pred = filter_short_runs(y_pred, min_anom_run)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_score:
                best_score = acc
                best_thr = float(thr)
    return best_thr


def calibrate_threshold_constraints(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_precision: float,
    min_recall: float,
    min_anom_run: int = 1,
) -> float:
    if len(scores) == 0:
        return float("nan")
    thresholds = np.unique(scores)
    best_score = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        y_pred = filter_short_runs(y_pred, min_anom_run)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if prec < min_precision or rec < min_recall:
            continue
        acc = accuracy_score(y_true, y_pred)
        if acc > best_score:
            best_score = acc
            best_thr = float(thr)
    if best_score < 0:
        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)
            y_pred = filter_short_runs(y_pred, min_anom_run)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_score:
                best_score = acc
                best_thr = float(thr)
    return best_thr


def filter_short_runs(y_pred: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1 or len(y_pred) == 0:
        return y_pred
    y_pred = y_pred.copy()
    idx = 0
    n = len(y_pred)
    while idx < n:
        if y_pred[idx] == 0:
            idx += 1
            continue
        start = idx
        while idx < n and y_pred[idx] == 1:
            idx += 1
        if idx - start < min_run:
            y_pred[start:idx] = 0
    return y_pred


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, thr: float, min_anom_run: int = 1
) -> dict:
    y_pred = (scores >= thr).astype(int)
    y_pred = filter_short_runs(y_pred, min_anom_run)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def quantize_model(model: tf.keras.Model, rep_data: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_dataset():
        for i in range(min(len(rep_data), 200)):
            yield [rep_data[i : i + 1]]

    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasting anomaly detector.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR / "model_output")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-frac", type=float, default=0.05)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--fit-verbose", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--model-type", choices=["linear", "mlp", "cnn"], default="linear"
    )
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--score-smooth", type=int, default=1)
    parser.add_argument(
        "--score-agg",
        choices=["mean", "max", "topk"],
        default="mean",
        help="Aggregate per-feature errors into a scalar score.",
    )
    parser.add_argument(
        "--score-topk",
        type=int,
        default=3,
        help="Top-k features to average when score-agg=topk.",
    )
    parser.add_argument(
        "--mask-penalty",
        type=float,
        default=0.0,
        help="Penalty added to scores when masks are dropped (useful for dropout anomalies).",
    )
    parser.add_argument("--anom-min-len", type=int, default=10)
    parser.add_argument("--anom-max-len", type=int, default=18)
    parser.add_argument("--anom-severity", type=float, default=3.5)
    parser.add_argument(
        "--anom-fixed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fixed anomaly magnitudes for more consistent injections.",
    )
    parser.add_argument(
        "--anom-global",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply anomalies across all features for stronger signals.",
    )
    parser.add_argument(
        "--anom-clip-nonneg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip non-negative features back to >=0 after anomaly injection.",
    )
    parser.add_argument(
        "--anom-easy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stronger, more separable anomaly magnitudes.",
    )
    parser.add_argument(
        "--anom-type-weights",
        type=str,
        default="",
        help="Comma-separated weights per type, e.g. dropout=0.7,spike=0.1.",
    )
    parser.add_argument(
        "--anom-types",
        type=str,
        default="spike,drop,noise,drift,stuck,dropout",
        help="Comma-separated anomaly types to inject.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Multi-step forecast horizon. Use >1 for multi-horizon scoring.",
    )
    parser.add_argument(
        "--use-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mask-aware scoring for anomaly MSE.",
    )
    parser.add_argument(
        "--anom-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance anomaly types during injection.",
    )
    parser.add_argument(
        "--anom-context",
        type=int,
        default=32,
        help="Context window for local anomaly statistics.",
    )
    parser.add_argument(
        "--anom-correlate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject correlated temp/hum anomalies when possible.",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument(
        "--calib-metric",
        choices=["fbeta", "accuracy", "constraints"],
        default="fbeta",
        help="Metric to optimize when calibrating the anomaly threshold.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="Minimum precision constraint when calibrating thresholds.",
    )
    parser.add_argument(
        "--min-anom-run",
        type=int,
        default=1,
        help="Minimum consecutive positive predictions to keep as anomalies.",
    )
    parser.add_argument(
        "--error-norm",
        choices=["none", "mad", "std"],
        default="none",
        help="Normalize per-feature prediction errors using train residual stats.",
    )
    parser.add_argument(
        "--drop-features",
        type=str,
        default="accel_mean,accel_std,accel_max",
        help="Comma-separated feature names to drop before selection.",
    )
    parser.add_argument("--train-pctl", type=float, default=99.5)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no GPU is detected.",
    )
    parser.add_argument(
        "--threshold-cap",
        choices=["none", "train_pctl"],
        default="train_pctl",
        help="Optionally cap the calibrated threshold using train-score percentile.",
    )
    parser.add_argument(
        "--threshold-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to the calibrated threshold.",
    )
    parser.add_argument(
        "--real-anoms",
        type=Path,
        default=None,
        help="CSV with real anomaly time ranges (start_ts/end_ts or start/end).",
    )
    parser.add_argument(
        "--drop-real-anoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop real anomaly rows from training split.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    device = configure_device()
    if args.require_gpu and device != "GPU":
        raise SystemExit(
            "GPU required but none detected. Install tensorflow-macos and "
            "tensorflow-metal on Apple Silicon, or run on a machine with a visible GPU."
        )
    print(f"Training device: {device}", flush=True)

    print("Preparing data...", flush=True)
    merged = merge_sensors(args.data_dir)
    merged = add_masks(merged)

    raw_features = [c for cols in RAW_GROUPS.values() for c in cols]
    drop_list = [s.strip() for s in args.drop_features.split(",") if s.strip()]
    real_labels = None
    real_col = "_real_anom"
    if args.real_anoms is not None:
        events = load_real_events(args.real_anoms)
        real_labels = label_real_anomalies(merged, events)
        merged = merged.copy()
        merged[real_col] = real_labels

    train_raw_df, val_raw_df, test_raw_df = split_time_series(
        merged, args.train_frac, args.val_frac
    )
    real_test_labels = None
    if real_labels is not None:
        if args.drop_real_anoms:
            train_raw_df = train_raw_df[train_raw_df[real_col] == 0].reset_index(drop=True)
        real_test_labels = test_raw_df[real_col].to_numpy(dtype=np.int8)

    train_feature_df = train_raw_df.drop(columns=[real_col], errors="ignore")
    val_feature_df = val_raw_df.drop(columns=[real_col], errors="ignore")
    test_feature_df = test_raw_df.drop(columns=[real_col], errors="ignore")

    variances = train_feature_df[raw_features].var().fillna(0.0).to_dict()
    if drop_list:
        unknown = sorted(set(drop_list) - set(raw_features))
        if unknown:
            raise ValueError(f"Unknown drop features: {unknown}")
        selected = [f for f in raw_features if f not in drop_list]
        if len(selected) != 13:
            raise ValueError(
                f"Selected {len(selected)} features after drop, expected 13."
            )
        dropped = drop_list
    else:
        selected, dropped, _ = select_features_by_variance(
            train_feature_df, raw_features, 13
        )

    train_df = build_contract_v1(train_feature_df, selected)
    val_df = build_contract_v1(val_feature_df, selected)
    test_df = build_contract_v1(test_feature_df, selected)
    calib_df, val_eval_df = split_fraction(val_df, args.calib_frac)

    calib_len = len(calib_df)
    val_len = len(val_eval_df)
    if calib_len == 0 or val_len == 0:
        raise SystemExit("Calibration/validation split is empty; adjust --calib-frac.")
    calib_df_mod, calib_labels = inject_anomalies(
        calib_df,
        selected,
        seed=args.seed + 1,
        target_frac=args.anomaly_frac,
        min_index=args.window,
        min_len=args.anom_min_len,
        max_len=args.anom_max_len,
        severity=args.anom_severity,
        fixed_scale=args.anom_fixed,
        force_all_features=args.anom_global,
        clip_nonneg=args.anom_clip_nonneg,
        easy_mode=args.anom_easy,
        balance_types=args.anom_balanced,
        context_window=args.anom_context,
        correlate_temp_hum=args.anom_correlate,
        anomaly_types=[t.strip() for t in args.anom_types.split(",") if t.strip()],
        anomaly_type_weights=parse_anom_type_weights(args.anom_type_weights),
    )
    val_df_mod, val_labels = inject_anomalies(
        val_eval_df,
        selected,
        seed=args.seed + 2,
        target_frac=args.anomaly_frac,
        min_index=args.window,
        min_len=args.anom_min_len,
        max_len=args.anom_max_len,
        severity=args.anom_severity,
        fixed_scale=args.anom_fixed,
        force_all_features=args.anom_global,
        clip_nonneg=args.anom_clip_nonneg,
        easy_mode=args.anom_easy,
        balance_types=args.anom_balanced,
        context_window=args.anom_context,
        correlate_temp_hum=args.anom_correlate,
        anomaly_types=[t.strip() for t in args.anom_types.split(",") if t.strip()],
        anomaly_type_weights=parse_anom_type_weights(args.anom_type_weights),
    )
    if real_test_labels is not None and real_test_labels.sum() > 0:
        print("Using real anomaly labels for test split.", flush=True)
        test_df_mod = test_df.copy()
        test_labels = real_test_labels
    else:
        test_df_mod, test_labels = inject_anomalies(
            test_df,
            selected,
            seed=args.seed + 3,
            target_frac=args.anomaly_frac,
            min_index=args.window,
            min_len=args.anom_min_len,
            max_len=args.anom_max_len,
            severity=args.anom_severity,
            fixed_scale=args.anom_fixed,
            force_all_features=args.anom_global,
            clip_nonneg=args.anom_clip_nonneg,
            easy_mode=args.anom_easy,
            balance_types=args.anom_balanced,
            context_window=args.anom_context,
            correlate_temp_hum=args.anom_correlate,
            anomaly_types=[t.strip() for t in args.anom_types.split(",") if t.strip()],
            anomaly_type_weights=parse_anom_type_weights(args.anom_type_weights),
        )

    scaler = compute_scaler(train_df, selected)
    x_train = apply_scaler(train_df, selected, scaler)
    x_calib = apply_scaler(calib_df_mod, selected, scaler)
    x_val = apply_scaler(val_df_mod, selected, scaler)
    x_test = apply_scaler(test_df_mod, selected, scaler)
    mask_cols = list(MASK_NAMES.values())
    masks_train = train_df[mask_cols].to_numpy(dtype=np.float32)
    masks_calib = calib_df_mod[mask_cols].to_numpy(dtype=np.float32)
    masks_val = val_df_mod[mask_cols].to_numpy(dtype=np.float32)
    masks_test = test_df_mod[mask_cols].to_numpy(dtype=np.float32)
    feat_to_group = {
        "dist_": "mask_dist",
        "accel_": "mask_accel",
        "temp_": "mask_temp",
        "hum_": "mask_hum",
    }
    feat_mask_names = []
    for feat in selected:
        for prefix, mname in feat_to_group.items():
            if feat.startswith(prefix):
                feat_mask_names.append(mname)
                break
        else:
            feat_mask_names.append("mask_dist")
    mask_idx = [mask_cols.index(m) for m in feat_mask_names]

    if args.horizon > 1:
        X_train, y_train = make_windows_multi(x_train, args.window, args.horizon)
        X_calib, y_calib = make_windows_multi(x_calib, args.window, args.horizon)
        X_val, y_val = make_windows_multi(x_val, args.window, args.horizon)
        X_test, y_test = make_windows_multi(x_test, args.window, args.horizon)
        _, m_train = make_windows_multi(masks_train[:, mask_idx], args.window, args.horizon)
        _, m_calib = make_windows_multi(masks_calib[:, mask_idx], args.window, args.horizon)
        _, m_val = make_windows_multi(masks_val[:, mask_idx], args.window, args.horizon)
        _, m_test = make_windows_multi(masks_test[:, mask_idx], args.window, args.horizon)
    else:
        X_train, y_train = make_windows(x_train, args.window)
        X_calib, y_calib = make_windows(x_calib, args.window)
        X_val, y_val = make_windows(x_val, args.window)
        X_test, y_test = make_windows(x_test, args.window)
        m_train = masks_train[args.window:][:, mask_idx]
        m_calib = masks_calib[args.window:][:, mask_idx]
        m_val = masks_val[args.window:][:, mask_idx]
        m_test = masks_test[args.window:][:, mask_idx]

    if len(X_train) == 0 or len(X_calib) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise SystemExit(
            "Not enough data for windowed splits. Increase data size or lower --window."
        )

    calib_labels_w = calib_labels[args.window :]
    val_labels_w = val_labels[args.window :]
    test_labels_w = test_labels[args.window :]
    if args.horizon > 1:
        max_len = min(len(calib_labels_w), len(m_calib))
        calib_labels_w = calib_labels_w[:max_len]
        m_calib = m_calib[:max_len]
        max_len = min(len(val_labels_w), len(m_val))
        val_labels_w = val_labels_w[:max_len]
        m_val = m_val[:max_len]
        max_len = min(len(test_labels_w), len(m_test))
        test_labels_w = test_labels_w[:max_len]
        m_test = m_test[:max_len]

    if args.model_type == "linear":
        print("Fitting linear ridge model...", flush=True)
        w, b = fit_linear_ridge(X_train, y_train, l2=args.ridge_l2)
        if args.horizon > 1:
            model = build_linear_multi_model(args.window, len(selected), args.horizon)
            model.layers[-2].set_weights([w, b])
        else:
            model = build_linear_model(args.window, len(selected))
            model.layers[-1].set_weights([w, b])
        model.compile(optimizer="adam", loss="mse")
    elif args.model_type == "mlp":
        if args.horizon > 1:
            model = build_mlp_multi_model(args.window, len(selected), args.horizon)
        else:
            model = build_mlp_model(args.window, len(selected))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            EpochLogger(enabled=True),
        ]

        print("Training model...", flush=True)
        model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=callbacks,
            verbose=args.fit_verbose,
        )
    else:
        if args.horizon > 1:
            model = build_cnn_multi_model(args.window, len(selected), args.horizon)
        else:
            model = build_cnn_model(args.window, len(selected))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            EpochLogger(enabled=True),
        ]

        print("Training model...", flush=True)
        model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=callbacks,
            verbose=args.fit_verbose,
        )

    print("Scoring calibration/validation/test...", flush=True)
    if args.model_type == "linear":
        train_flat = X_train.reshape(len(X_train), -1).astype(np.float64)
        calib_flat = X_calib.reshape(len(X_calib), -1).astype(np.float64)
        val_flat = X_val.reshape(len(X_val), -1).astype(np.float64)
        test_flat = X_test.reshape(len(X_test), -1).astype(np.float64)
        w64 = w.astype(np.float64)
        b64 = b.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            train_pred = train_flat @ w64 + b64
            calib_pred = calib_flat @ w64 + b64
            val_pred = val_flat @ w64 + b64
            test_pred = test_flat @ w64 + b64
    else:
        train_pred = model.predict(X_train, verbose=0)
        calib_pred = model.predict(X_calib, verbose=0)
        val_pred = model.predict(X_val, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
    print("Predictions complete.", flush=True)

    if args.horizon > 1 and args.model_type == "linear":
        train_pred = train_pred.reshape(len(X_train), args.horizon, len(selected))
        calib_pred = calib_pred.reshape(len(X_calib), args.horizon, len(selected))
        val_pred = val_pred.reshape(len(X_val), args.horizon, len(selected))
        test_pred = test_pred.reshape(len(X_test), args.horizon, len(selected))

    error_scale = None
    if args.error_norm != "none":
        error_scale = compute_error_scale(y_train, train_pred, args.error_norm)

    train_scores = compute_scores(
        y_train,
        train_pred,
        m_train if args.use_masks else None,
        agg=args.score_agg,
        topk=args.score_topk,
        error_scale=error_scale,
    )
    calib_scores = compute_scores(
        y_calib,
        calib_pred,
        m_calib if args.use_masks else None,
        agg=args.score_agg,
        topk=args.score_topk,
        error_scale=error_scale,
    )
    val_scores = compute_scores(
        y_val,
        val_pred,
        m_val if args.use_masks else None,
        agg=args.score_agg,
        topk=args.score_topk,
        error_scale=error_scale,
    )
    test_scores = compute_scores(
        y_test,
        test_pred,
        m_test if args.use_masks else None,
        agg=args.score_agg,
        topk=args.score_topk,
        error_scale=error_scale,
    )
    if args.mask_penalty > 0.0:
        train_scores = apply_mask_penalty(train_scores, m_train, args.mask_penalty)
        calib_scores = apply_mask_penalty(calib_scores, m_calib, args.mask_penalty)
        val_scores = apply_mask_penalty(val_scores, m_val, args.mask_penalty)
        test_scores = apply_mask_penalty(test_scores, m_test, args.mask_penalty)
    if args.score_smooth > 1:
        train_scores = smooth_scores(train_scores, args.score_smooth)
        calib_scores = smooth_scores(calib_scores, args.score_smooth)
        val_scores = smooth_scores(val_scores, args.score_smooth)
        test_scores = smooth_scores(test_scores, args.score_smooth)

    if args.calib_metric == "constraints":
        threshold_uncapped = calibrate_threshold_constraints(
            calib_labels_w,
            calib_scores,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            min_anom_run=args.min_anom_run,
        )
    elif args.calib_metric == "accuracy":
        threshold_uncapped = calibrate_threshold_accuracy(
            calib_labels_w,
            calib_scores,
            min_recall=args.min_recall,
            min_anom_run=args.min_anom_run,
        )
    else:
        threshold_uncapped = calibrate_threshold_fbeta(
            calib_labels_w,
            calib_scores,
            beta=args.beta,
            min_recall=args.min_recall,
            min_anom_run=args.min_anom_run,
        )
    threshold = threshold_uncapped * args.threshold_mult
    train_pctl = None
    if args.threshold_cap == "train_pctl" and args.calib_metric != "constraints":
        train_pctl = float(np.percentile(train_scores, args.train_pctl))
        threshold = min(threshold, train_pctl)
    train_labels_w = np.zeros_like(train_scores, dtype=np.int8)
    train_metrics = compute_metrics(
        train_labels_w, train_scores, threshold, args.min_anom_run
    )
    calib_metrics = compute_metrics(
        calib_labels_w, calib_scores, threshold, args.min_anom_run
    )
    val_metrics = compute_metrics(val_labels_w, val_scores, threshold, args.min_anom_run)
    test_metrics = compute_metrics(
        test_labels_w, test_scores, threshold, args.min_anom_run
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "forecast_model.keras"
    model.save(model_path, include_optimizer=False)
    h5_path = args.out_dir / "forecast_model.h5"
    weights_path = args.out_dir / "forecast.weights.h5"
    try:
        model.save(h5_path, include_optimizer=False)
    except Exception:
        pass
    try:
        model.save_weights(weights_path)
    except Exception as exc:
        print(f"Warning: failed to save weights ({exc})", flush=True)
    saved_model_dir = args.out_dir / "forecast_saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    try:
        if hasattr(model, "export"):
            model.export(str(saved_model_dir))
        else:
            tf.saved_model.save(model, saved_model_dir)
        print("Model saved.", flush=True)
    except Exception as exc:
        print(f"Warning: failed to save SavedModel ({exc})", flush=True)
        print("Continuing without SavedModel export.", flush=True)

    if args.skip_export:
        int8_path = None
    else:
        print("Exporting TFLite models...", flush=True)
        tflite_fp32 = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        (args.out_dir / "forecast_model_fp32.tflite").write_bytes(tflite_fp32)

        tflite_int8 = quantize_model(model, X_train)
        int8_path = args.out_dir / "forecast_model_int8.tflite"
        int8_path.write_bytes(tflite_int8)
        print("TFLite export complete.", flush=True)

    schema = ContractSchema(
        contract_version="v1",
        feature_names=selected,
        mask_names=list(MASK_NAMES.values()),
        raw_feature_names=raw_features,
        dropped_features=dropped,
        window_size=args.window,
    )
    save_json(args.out_dir / "contract_v1_schema.json", schema.to_dict())
    save_json(args.out_dir / "scaler.json", scaler)
    save_json(
        args.out_dir / "thresholds.json",
        {"threshold": threshold, "metric": "mse"},
    )

    metrics_payload = {
        "train": train_metrics,
        "calib": calib_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "threshold_uncapped": threshold_uncapped,
        "train_pctl": train_pctl,
        "threshold_cap": args.threshold_cap,
        "threshold_mult": args.threshold_mult,
        "calib_metric": args.calib_metric,
        "min_precision": args.min_precision,
        "min_anom_run": args.min_anom_run,
        "mask_penalty": args.mask_penalty,
        "score_agg": args.score_agg,
        "score_topk": args.score_topk,
        "error_norm": args.error_norm,
        "model_type": args.model_type,
        "beta": args.beta,
        "min_recall": args.min_recall,
        "calib_frac": args.calib_frac,
        "anomaly_injection": {
            "frac": args.anomaly_frac,
            "min_len": args.anom_min_len,
            "max_len": args.anom_max_len,
            "severity": args.anom_severity,
            "fixed_scale": args.anom_fixed,
            "global_features": args.anom_global,
            "clip_nonneg": args.anom_clip_nonneg,
            "easy_mode": args.anom_easy,
            "balanced": args.anom_balanced,
            "context_window": args.anom_context,
            "correlate_temp_hum": args.anom_correlate,
            "types": [t.strip() for t in args.anom_types.split(",") if t.strip()],
            "type_weights": parse_anom_type_weights(args.anom_type_weights),
        },
        "real_test_labels_used": bool(real_test_labels is not None and real_test_labels.sum() > 0),
    }
    save_json(args.out_dir / "metrics.json", metrics_payload)
    save_json(args.out_dir / "train_report.json", metrics_payload)

    calib_df_mod = calib_df_mod.copy()
    calib_df_mod["anomaly"] = calib_labels
    val_df_mod = val_df_mod.copy()
    val_df_mod["anomaly"] = val_labels
    test_df_mod = test_df_mod.copy()
    test_df_mod["anomaly"] = test_labels
    train_df = train_df.copy()
    train_df["anomaly"] = 0

    train_df.to_csv(args.out_dir / "train_split.csv", index=False)
    calib_df_mod.to_csv(args.out_dir / "calib_split.csv", index=False)
    val_df_mod.to_csv(args.out_dir / "val_split.csv", index=False)
    test_df_mod.to_csv(args.out_dir / "test_split.csv", index=False)
    print("Artifacts saved.", flush=True)

    print("Model type:", args.model_type)
    print("Selected features:", selected)
    print("Dropped features:", dropped)
    print(
        "Train/Calib/Val/Test rows:",
        len(train_df),
        len(calib_df),
        len(val_eval_df),
        len(test_df),
    )
    print(
        f"Threshold (calib F{args.beta} w/ min_recall={args.min_recall}):",
        threshold,
    )
    print("Threshold uncapped:", threshold_uncapped)
    if train_pctl is not None:
        print(f"Train pctl ({args.train_pctl}):", train_pctl)
    print("Train metrics:", train_metrics)
    print("Calibration metrics:", calib_metrics)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved:", model_path)
    if int8_path is not None:
        print("Saved:", int8_path, f"({int8_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
