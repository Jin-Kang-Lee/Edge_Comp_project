import argparse
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/export using Optuna best params."
    )
    parser.add_argument("--artifacts", type=Path, default=THIS_DIR / "model_output")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--rep-samples", type=int, default=50)
    parser.add_argument(
        "--drop-features",
        type=str,
        default="accel_mean,accel_std,accel_max",
        help="Comma-separated features to drop (must match training setup).",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no GPU is detected for training.",
    )
    args = parser.parse_args()

    best_path = args.artifacts / "optuna_best.json"
    if not best_path.exists():
        raise SystemExit(f"Missing {best_path}. Run optuna_tune.py first.")

    best = json.loads(best_path.read_text())
    params = best.get("best_params", {})

    window = params.get("window")
    model_type = params.get("model_type")
    score_smooth = params.get("score_smooth")
    ridge_l2 = params.get("ridge_l2")
    epochs = params.get("epochs")
    mask_penalty = params.get("mask_penalty")
    score_agg = params.get("score_agg")
    score_topk = params.get("score_topk")
    min_anom_run = params.get("min_anom_run", best.get("min_anom_run"))
    threshold_mult = params.get("threshold_mult")
    error_norm = params.get("error_norm", best.get("error_norm"))
    beta = best.get("beta")
    min_recall = best.get("min_recall")
    calib_metric = best.get("calib_metric")
    min_precision = best.get("min_precision")
    threshold_cap = best.get("threshold_cap")
    train_pctl = best.get("train_pctl")
    horizon = best.get("horizon")
    use_masks = best.get("use_masks")
    anom = best.get("anom", {}) if isinstance(best.get("anom"), dict) else {}

    if window is None or model_type is None:
        raise SystemExit("optuna_best.json is missing required params.")

    train_cmd = [
        sys.executable,
        str(THIS_DIR / "train_and_threshold.py"),
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(args.artifacts),
        "--window",
        str(window),
        "--model-type",
        str(model_type),
        "--drop-features",
        args.drop_features,
        "--skip-export",
    ]
    if args.require_gpu:
        train_cmd.append("--require-gpu")

    if score_smooth is not None:
        train_cmd += ["--score-smooth", str(score_smooth)]
    if score_agg:
        train_cmd += ["--score-agg", str(score_agg)]
    if score_topk is not None:
        train_cmd += ["--score-topk", str(score_topk)]
    if mask_penalty is not None:
        train_cmd += ["--mask-penalty", str(mask_penalty)]
    if beta is not None:
        train_cmd += ["--beta", str(beta)]
    if min_recall is not None:
        train_cmd += ["--min-recall", str(min_recall)]
    if min_precision is not None:
        train_cmd += ["--min-precision", str(min_precision)]
    if min_anom_run is not None:
        train_cmd += ["--min-anom-run", str(min_anom_run)]
    if threshold_mult is not None:
        train_cmd += ["--threshold-mult", str(threshold_mult)]
    if error_norm:
        train_cmd += ["--error-norm", str(error_norm)]
    if calib_metric:
        train_cmd += ["--calib-metric", str(calib_metric)]
    if threshold_cap:
        train_cmd += ["--threshold-cap", str(threshold_cap)]
        if threshold_cap == "train_pctl" and train_pctl is not None:
            train_cmd += ["--train-pctl", str(train_pctl)]
    if horizon is not None:
        train_cmd += ["--horizon", str(horizon)]
    if use_masks is not None:
        train_cmd.append("--use-masks" if use_masks else "--no-use-masks")
    if anom:
        if "frac" in anom:
            train_cmd += ["--anomaly-frac", str(anom["frac"])]
        if "min_len" in anom:
            train_cmd += ["--anom-min-len", str(anom["min_len"])]
        if "max_len" in anom:
            train_cmd += ["--anom-max-len", str(anom["max_len"])]
        if "severity" in anom:
            train_cmd += ["--anom-severity", str(anom["severity"])]
        if "fixed_scale" in anom:
            train_cmd.append("--anom-fixed" if anom["fixed_scale"] else "--no-anom-fixed")
        if "global_features" in anom:
            train_cmd.append(
                "--anom-global" if anom["global_features"] else "--no-anom-global"
            )
        if "clip_nonneg" in anom:
            train_cmd.append(
                "--anom-clip-nonneg"
                if anom["clip_nonneg"]
                else "--no-anom-clip-nonneg"
            )
        if "easy_mode" in anom:
            train_cmd.append(
                "--anom-easy" if anom["easy_mode"] else "--no-anom-easy"
            )
        if "type_weights" in anom and anom["type_weights"]:
            weights = ",".join(
                f"{k}={v}" for k, v in anom["type_weights"].items()
            )
            train_cmd += ["--anom-type-weights", weights]
        if "balanced" in anom:
            train_cmd.append("--anom-balanced" if anom["balanced"] else "--no-anom-balanced")
        if "context_window" in anom:
            train_cmd += ["--anom-context", str(anom["context_window"])]
        if "correlate_temp_hum" in anom:
            train_cmd.append(
                "--anom-correlate" if anom["correlate_temp_hum"] else "--no-anom-correlate"
            )
        if "types" in anom and anom["types"]:
            train_cmd += ["--anom-types", ",".join(str(t) for t in anom["types"])]
    if model_type == "linear" and ridge_l2 is not None:
        train_cmd += ["--ridge-l2", str(ridge_l2)]
    if model_type in {"mlp", "cnn"} and epochs is not None:
        train_cmd += ["--epochs", str(epochs)]

    print("Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    export_cmd = [
        sys.executable,
        str(THIS_DIR / "quantize_model.py"),
        "--artifacts",
        str(args.artifacts),
        "--rep-samples",
        str(args.rep_samples),
    ]
    print("Running:", " ".join(export_cmd))
    subprocess.run(export_cmd, check=True)


if __name__ == "__main__":
    main()
