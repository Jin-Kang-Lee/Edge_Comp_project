import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

try:
    import optuna  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna is not installed. Run: python3 -m pip install optuna"
    ) from exc

from pipeline import (
    RAW_GROUPS,
    MASK_NAMES,
    add_masks,
    apply_scaler,
    build_contract_v1,
    compute_scaler,
    inject_anomalies,
    make_windows,
    make_windows_multi,
    merge_sensors,
    select_features_by_variance,
    split_time_series,
)
from train_and_threshold import (
    build_cnn_model,
    build_cnn_multi_model,
    build_linear_model,
    build_linear_multi_model,
    build_mlp_model,
    build_mlp_multi_model,
    calibrate_threshold_accuracy,
    calibrate_threshold_constraints,
    calibrate_threshold_fbeta,
    compute_error_scale,
    parse_anom_type_weights,
    compute_scores,
    fit_linear_ridge,
    smooth_scores,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


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
) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(int)
    y_pred = filter_short_runs(y_pred, min_anom_run)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def masked_mse_scores(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    err = np.square(y_true - y_pred) * mask
    denom = np.maximum(mask.sum(axis=1), 1e-6)
    return err.sum(axis=1) / denom


def masked_mse_scores_multi(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    err = np.square(y_true - y_pred) * mask
    denom = np.maximum(mask.sum(axis=(1, 2)), 1e-6)
    return err.sum(axis=(1, 2)) / denom


def apply_mask_penalty(scores: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return scores
    if mask.ndim == 2:
        penalty = (1.0 - mask).mean(axis=1)
    else:
        penalty = (1.0 - mask).mean(axis=(1, 2))
    return scores + weight * penalty


def split_fraction(df, frac: float):
    frac = max(0.0, min(1.0, frac))
    if len(df) == 0:
        return df, df
    split_idx = int(len(df) * frac)
    split_idx = max(1, min(len(df) - 1, split_idx))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(
        drop=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for anomaly model.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR / "model_output")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--min-window", type=int, default=3)
    parser.add_argument("--max-window", type=int, default=10)
    parser.add_argument("--model-space", type=str, default="linear,mlp,cnn")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--score-smooth", type=int, default=1)
    parser.add_argument(
        "--score-agg-space",
        type=str,
        default="mean,max,topk",
        help="Comma-separated score aggregations to search.",
    )
    parser.add_argument(
        "--max-score-topk",
        type=int,
        default=5,
        help="Maximum top-k when score-agg=topk.",
    )
    parser.add_argument(
        "--max-mask-penalty",
        type=float,
        default=2.0,
        help="Max penalty weight for dropped masks (0 disables tuning).",
    )
    parser.add_argument(
        "--objective",
        choices=["fbeta", "fbeta_acc"],
        default="fbeta",
        help="Optimization objective for Optuna.",
    )
    parser.add_argument(
        "--acc-weight",
        type=float,
        default=0.3,
        help="Weight for accuracy when objective=fbeta_acc.",
    )
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
        "--min-anom-run-space",
        type=str,
        default="",
        help="Comma-separated run-lengths to tune (overrides --min-anom-run).",
    )
    parser.add_argument("--min-val-precision", type=float, default=0.0)
    parser.add_argument("--min-val-recall", type=float, default=0.0)
    parser.add_argument("--min-val-accuracy", type=float, default=0.0)
    parser.add_argument(
        "--val-constraint-mode",
        choices=["agg", "all"],
        default="agg",
        help="Apply min-val constraints to aggregated metrics or to every seed.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--error-norm",
        choices=["none", "mad", "std"],
        default="none",
        help="Normalize per-feature prediction errors using train residual stats.",
    )
    parser.add_argument(
        "--use-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mask-aware MSE scoring.",
    )
    parser.add_argument("--anom-frac", type=float, default=0.05)
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
    parser.add_argument(
        "--drop-features",
        type=str,
        default="accel_mean,accel_std,accel_max",
        help="Comma-separated feature names to drop before selection.",
    )
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--train-pctl", type=float, default=99.5)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed-gap", type=int, default=11)
    parser.add_argument(
        "--metric-agg",
        choices=["mean", "median"],
        default="median",
        help="Aggregate metric across multiple anomaly seeds.",
    )
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
        "--threshold-mult-min",
        type=float,
        default=1.0,
        help="Minimum multiplier applied to calibrated thresholds during tuning.",
    )
    parser.add_argument(
        "--threshold-mult-max",
        type=float,
        default=1.0,
        help="Maximum multiplier applied to calibrated thresholds during tuning.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise SystemExit(
            "GPU required but none detected. Install tensorflow-macos and "
            "tensorflow-metal on Apple Silicon, or run on a machine with a visible GPU."
        )

    merged = merge_sensors(args.data_dir)
    merged = add_masks(merged)

    raw_features = [c for cols in RAW_GROUPS.values() for c in cols]
    train_raw_df, val_raw_df, _holdout_raw_df = split_time_series(merged, 0.7, 0.15)
    drop_list = [s.strip() for s in args.drop_features.split(",") if s.strip()]
    if drop_list:
        unknown = sorted(set(drop_list) - set(raw_features))
        if unknown:
            raise ValueError(f"Unknown drop features: {unknown}")
        selected = [f for f in raw_features if f not in drop_list]
        if len(selected) != 13:
            raise ValueError(
                f"Selected {len(selected)} features after drop, expected 13."
            )
    else:
        selected, _, _ = select_features_by_variance(train_raw_df, raw_features, 13)
    train_df = build_contract_v1(train_raw_df, selected)
    val_df = build_contract_v1(val_raw_df, selected)

    calib_df, val_eval_df = split_fraction(val_df, args.calib_frac)
    scaler = compute_scaler(train_df, selected)
    mask_cols = list(MASK_NAMES.values())
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

    model_space = [m.strip() for m in args.model_space.split(",") if m.strip()]

    def objective(trial: optuna.Trial) -> float:
        window = trial.suggest_int("window", args.min_window, args.max_window)
        model_type = trial.suggest_categorical("model_type", model_space)
        score_smooth = trial.suggest_int("score_smooth", 1, args.score_smooth)
        score_agg_space = [
            s.strip()
            for s in args.score_agg_space.split(",")
            if s.strip()
        ]
        if not score_agg_space:
            score_agg_space = ["mean"]
        score_agg = trial.suggest_categorical("score_agg", score_agg_space)
        score_topk = 3
        if score_agg == "topk":
            score_topk = trial.suggest_int(
                "score_topk", 2, max(2, args.max_score_topk)
            )
        mask_penalty = 0.0
        if args.max_mask_penalty > 0:
            mask_penalty = trial.suggest_float(
                "mask_penalty", 0.0, args.max_mask_penalty
            )
        min_anom_run = args.min_anom_run
        if args.min_anom_run_space:
            space = [
                int(s)
                for s in args.min_anom_run_space.split(",")
                if s.strip()
            ]
            if space:
                min_anom_run = trial.suggest_categorical("min_anom_run", space)
        threshold_mult = args.threshold_mult_min
        if args.threshold_mult_max != args.threshold_mult_min:
            threshold_mult = trial.suggest_float(
                "threshold_mult",
                args.threshold_mult_min,
                args.threshold_mult_max,
            )

        if model_type == "linear":
            l2 = trial.suggest_float("ridge_l2", 1e-5, 1e-1, log=True)
        else:
            l2 = None

        calib_len = len(calib_df)
        val_len = len(val_eval_df)
        if calib_len == 0 or val_len == 0:
            return 0.0
        x_train = apply_scaler(train_df, selected, scaler)
        masks_train = train_df[mask_cols].to_numpy(dtype=np.float32)
        if args.horizon > 1:
            X_train, y_train = make_windows_multi(x_train, window, args.horizon)
            _, m_train = make_windows_multi(
                masks_train[:, mask_idx], window, args.horizon
            )
        else:
            X_train, y_train = make_windows(x_train, window)
            m_train = masks_train[window:][:, mask_idx]

        if len(X_train) == 0:
            return 0.0

        if model_type == "linear":
            w, b = fit_linear_ridge(X_train, y_train, l2=l2 or 1e-3)
            train_pred = X_train.reshape(len(X_train), -1) @ w + b
        else:
            if model_type == "mlp":
                if args.horizon > 1:
                    model = build_mlp_multi_model(window, len(selected), args.horizon)
                else:
                    model = build_mlp_model(window, len(selected))
            else:
                if args.horizon > 1:
                    model = build_cnn_multi_model(window, len(selected), args.horizon)
                else:
                    model = build_cnn_model(window, len(selected))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ]
            min_epochs = min(args.min_epochs, args.max_epochs)
            epochs = trial.suggest_int("epochs", min_epochs, args.max_epochs)
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                shuffle=True,
                callbacks=callbacks,
                verbose=0,
            )
            train_pred = model.predict(X_train, verbose=0)

        if args.horizon > 1 and model_type == "linear":
            train_pred = train_pred.reshape(len(X_train), args.horizon, len(selected))
        error_scale = None
        if args.error_norm != "none":
            error_scale = compute_error_scale(y_train, train_pred, args.error_norm)

        train_scores = compute_scores(
            y_train,
            train_pred,
            m_train if args.use_masks else None,
            agg=score_agg,
            topk=score_topk,
            error_scale=error_scale,
        )
        if mask_penalty > 0.0:
            train_scores = apply_mask_penalty(train_scores, m_train, mask_penalty)
        if score_smooth > 1:
            train_scores = smooth_scores(train_scores, score_smooth)

        train_pctl = None
        if args.threshold_cap == "train_pctl":
            train_pctl = float(np.percentile(train_scores, args.train_pctl))

        val_fbetas = []
        thresholds = []
        thresholds_uncapped = []
        calib_metrics_list = []
        val_metrics_list = []

        for seed_idx in range(args.seed_count):
            seed_base = args.seed + seed_idx * args.seed_gap
            calib_df_mod, calib_labels = inject_anomalies(
                calib_df,
                selected,
                seed=seed_base + 1,
                target_frac=args.anom_frac,
                min_index=window,
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
                seed=seed_base + 2,
                target_frac=args.anom_frac,
                min_index=window,
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

            x_calib = apply_scaler(calib_df_mod, selected, scaler)
            x_val = apply_scaler(val_df_mod, selected, scaler)
            masks_calib = calib_df_mod[mask_cols].to_numpy(dtype=np.float32)
            masks_val = val_df_mod[mask_cols].to_numpy(dtype=np.float32)

            if args.horizon > 1:
                X_calib, y_calib = make_windows_multi(x_calib, window, args.horizon)
                X_val, y_val = make_windows_multi(x_val, window, args.horizon)
                _, m_calib = make_windows_multi(
                    masks_calib[:, mask_idx], window, args.horizon
                )
                _, m_val = make_windows_multi(
                    masks_val[:, mask_idx], window, args.horizon
                )
            else:
                X_calib, y_calib = make_windows(x_calib, window)
                X_val, y_val = make_windows(x_val, window)
                m_calib = masks_calib[window:][:, mask_idx]
                m_val = masks_val[window:][:, mask_idx]

            if len(X_calib) == 0 or len(X_val) == 0:
                continue

            if args.horizon > 1:
                calib_labels_w = calib_labels[window : window + len(m_calib)]
                val_labels_w = val_labels[window : window + len(m_val)]
            else:
                calib_labels_w = calib_labels[window:]
                val_labels_w = val_labels[window:]

            if model_type == "linear":
                calib_pred = X_calib.reshape(len(X_calib), -1) @ w + b
                val_pred = X_val.reshape(len(X_val), -1) @ w + b
            else:
                calib_pred = model.predict(X_calib, verbose=0)
                val_pred = model.predict(X_val, verbose=0)

            if args.horizon > 1 and model_type == "linear":
                calib_pred = calib_pred.reshape(
                    len(X_calib), args.horizon, len(selected)
                )
                val_pred = val_pred.reshape(len(X_val), args.horizon, len(selected))
            calib_scores = compute_scores(
                y_calib,
                calib_pred,
                m_calib if args.use_masks else None,
                agg=score_agg,
                topk=score_topk,
                error_scale=error_scale,
            )
            val_scores = compute_scores(
                y_val,
                val_pred,
                m_val if args.use_masks else None,
                agg=score_agg,
                topk=score_topk,
                error_scale=error_scale,
            )

            if mask_penalty > 0.0:
                calib_scores = apply_mask_penalty(calib_scores, m_calib, mask_penalty)
                val_scores = apply_mask_penalty(val_scores, m_val, mask_penalty)
            if score_smooth > 1:
                calib_scores = smooth_scores(calib_scores, score_smooth)
                val_scores = smooth_scores(val_scores, score_smooth)

            if args.calib_metric == "constraints":
                threshold_uncapped = calibrate_threshold_constraints(
                    calib_labels_w,
                    calib_scores,
                    min_precision=args.min_precision,
                    min_recall=args.min_recall,
                    min_anom_run=min_anom_run,
                )
            elif args.calib_metric == "accuracy":
                threshold_uncapped = calibrate_threshold_accuracy(
                    calib_labels_w,
                    calib_scores,
                    min_recall=args.min_recall,
                    min_anom_run=min_anom_run,
                )
            else:
                threshold_uncapped = calibrate_threshold_fbeta(
                    calib_labels_w,
                    calib_scores,
                    beta=args.beta,
                    min_recall=args.min_recall,
                    min_anom_run=min_anom_run,
                )
            threshold = threshold_uncapped * threshold_mult
            if train_pctl is not None and args.calib_metric != "constraints":
                threshold = min(threshold, train_pctl)

            calib_metrics = compute_metrics(
                calib_labels_w, calib_scores, threshold, min_anom_run
            )
            val_metrics = compute_metrics(
                val_labels_w, val_scores, threshold, min_anom_run
            )
            val_fbeta = fbeta_score(
                val_metrics["precision"], val_metrics["recall"], args.beta
            )

            thresholds.append(float(threshold))
            thresholds_uncapped.append(float(threshold_uncapped))
            calib_metrics_list.append(calib_metrics)
            val_metrics_list.append(val_metrics)
            val_fbetas.append(float(val_fbeta))

        if not val_fbetas:
            return 0.0

        agg = np.mean if args.metric_agg == "mean" else np.median
        val_fbeta_agg = float(agg(val_fbetas))
        threshold_agg = float(agg(thresholds))
        threshold_uncapped_agg = float(agg(thresholds_uncapped))

        def agg_metrics(metrics_list):
            keys = metrics_list[0].keys()
            return {k: float(agg([m[k] for m in metrics_list])) for k in keys}

        calib_metrics_agg = agg_metrics(calib_metrics_list)
        val_metrics_agg = agg_metrics(val_metrics_list)

        trial.set_user_attr("threshold", float(threshold_agg))
        trial.set_user_attr("threshold_uncapped", float(threshold_uncapped_agg))
        trial.set_user_attr("train_pctl", train_pctl)
        trial.set_user_attr("calib_metrics", calib_metrics_agg)
        trial.set_user_attr("val_metrics", val_metrics_agg)
        trial.set_user_attr("val_fbeta", float(val_fbeta_agg))
        trial.set_user_attr("val_fbeta_list", val_fbetas)

        min_val_prec = args.min_val_precision
        min_val_rec = args.min_val_recall
        min_val_acc = args.min_val_accuracy

        def meets_min(metrics: dict) -> bool:
            return (
                metrics.get("precision", 0.0) >= min_val_prec
                and metrics.get("recall", 0.0) >= min_val_rec
                and metrics.get("accuracy", 0.0) >= min_val_acc
            )

        if args.val_constraint_mode == "all":
            if any(not meets_min(m) for m in val_metrics_list):
                return 0.0
        else:
            if not meets_min(val_metrics_agg):
                return 0.0

        if args.objective == "fbeta_acc":
            acc_weight = max(0.0, min(1.0, args.acc_weight))
            val_acc = float(val_metrics_agg.get("accuracy", 0.0))
            return (1.0 - acc_weight) * val_fbeta_agg + acc_weight * val_acc
        return val_fbeta_agg

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout or None)

    best = study.best_trial
    best_params = dict(best.params)
    if "min_anom_run" not in best_params:
        best_params["min_anom_run"] = args.min_anom_run
    if "threshold_mult" not in best_params:
        best_params["threshold_mult"] = args.threshold_mult_min
    result = {
        "best_params": best_params,
        "best_value": best.value,
        "best_threshold": best.user_attrs.get("threshold"),
        "best_threshold_uncapped": best.user_attrs.get("threshold_uncapped"),
        "best_train_pctl": best.user_attrs.get("train_pctl"),
        "threshold_cap": args.threshold_cap,
        "train_pctl": args.train_pctl,
        "calib_metrics": best.user_attrs.get("calib_metrics"),
        "val_metrics": best.user_attrs.get("val_metrics"),
        "calib_frac": args.calib_frac,
        "seed_count": args.seed_count,
        "seed_gap": args.seed_gap,
        "metric_agg": args.metric_agg,
        "beta": args.beta,
        "min_recall": args.min_recall,
        "calib_metric": args.calib_metric,
        "min_precision": args.min_precision,
        "min_val_precision": args.min_val_precision,
        "min_val_recall": args.min_val_recall,
        "min_val_accuracy": args.min_val_accuracy,
        "val_constraint_mode": args.val_constraint_mode,
        "min_anom_run": best_params.get("min_anom_run"),
        "threshold_mult_min": args.threshold_mult_min,
        "threshold_mult_max": args.threshold_mult_max,
        "objective": args.objective,
        "acc_weight": args.acc_weight,
        "score_agg": best.params.get("score_agg"),
        "score_topk": best.params.get("score_topk"),
        "error_norm": args.error_norm,
        "horizon": args.horizon,
        "use_masks": args.use_masks,
        "anom": {
            "frac": args.anom_frac,
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
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "optuna_best.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.out_dir / "optuna_study.csv").write_text(
        study.trials_dataframe().to_csv(index=False), encoding="utf-8"
    )

    print("Best params:", best.params)
    print("Best val fbeta:", best.value)
    print("Best threshold uncapped:", best.user_attrs.get("threshold_uncapped"))
    print("Best train pctl:", best.user_attrs.get("train_pctl"))
    print("Best calib metrics:", best.user_attrs.get("calib_metrics"))
    print("Best val metrics:", best.user_attrs.get("val_metrics"))


if __name__ == "__main__":
    main()
