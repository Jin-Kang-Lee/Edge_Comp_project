import argparse
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import os
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "edgeml_mpl_cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "edgeml_xdg_cache")
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

from pipeline import apply_scaler, make_windows

THIS_DIR = Path(__file__).resolve().parent


def build_mlp_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs)


def build_cnn_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs)


def build_linear_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(n_features)(x)
    return tf.keras.Model(inputs, outputs)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_type: str, window: int, n_features: int) -> tf.keras.Model:
    if model_type == "cnn":
        return build_cnn_model(window, n_features)
    if model_type == "linear":
        return build_linear_model(window, n_features)
    return build_mlp_model(window, n_features)


def load_manual_h5_weights(
    model: tf.keras.Model, h5_path: Path, window: int, n_features: int
) -> None:
    _ = model(tf.zeros((1, window, n_features), dtype=tf.float32))
    with h5py.File(h5_path, "r") as f:
        weights_root = f["model_weights"]
        for layer in model.layers:
            if not layer.weights:
                continue
            group = weights_root.get(layer.name)
            if group is None:
                continue
            if layer.name in group:
                group = group[layer.name]
            values = []
            for weight in layer.weights:
                weight_name = weight.name.split("/")[-1].split(":")[0]
                if weight_name not in group:
                    values = []
                    break
                values.append(group[weight_name][()])
            if values:
                layer.set_weights(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export forecast model FP32 and INT8 TFLite models."
    )
    parser.add_argument("--artifacts", type=Path, default=THIS_DIR / "model_output")
    parser.add_argument("--rep-samples", type=int, default=200)
    parser.add_argument(
        "--rep-mode",
        choices=["real", "range", "zeros"],
        default="zeros",
        help="Representative dataset source for INT8 calibration.",
    )
    parser.add_argument(
        "--load-source",
        choices=["auto", "keras", "h5", "weights", "manual-h5"],
        default="auto",
        help="How to load the source model before TFLite export.",
    )
    args = parser.parse_args()

    model_path = args.artifacts / "forecast_model.keras"
    h5_path = args.artifacts / "forecast_model.h5"
    weights_path = args.artifacts / "forecast.weights.h5"
    schema_path = args.artifacts / "contract_v1_schema.json"
    scaler_path = args.artifacts / "scaler.json"
    train_path = args.artifacts / "train_split.csv"
    metrics_path = args.artifacts / "metrics.json"

    if (
        not model_path.exists()
        and not h5_path.exists()
        and not weights_path.exists()
    ):
        raise FileNotFoundError(
            f"Missing model: {model_path} or {h5_path} or {weights_path}"
        )
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    schema = load_json(schema_path)
    scaler = load_json(scaler_path)
    feature_names = schema["feature_names"]
    window = int(schema.get("window_size", 16))

    train_df = pd.read_csv(train_path)
    x_train = apply_scaler(train_df, feature_names, scaler)
    X_train, _ = make_windows(x_train, window)

    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    model_type = metrics.get("model_type", "mlp")
    model = None
    if args.load_source in {"auto", "keras"} and model_path.exists():
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            model = None
    if model is None and args.load_source in {"auto", "h5"} and h5_path.exists():
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
        except Exception:
            model = None
    if model is None and args.load_source in {"auto", "weights"} and weights_path.exists():
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")
        model = build_model(model_type, window, len(feature_names))
        model.load_weights(weights_path)
    if model is None and args.load_source in {"auto", "manual-h5"} and h5_path.exists():
        model = build_model(model_type, window, len(feature_names))
        load_manual_h5_weights(model, h5_path, window, len(feature_names))

    fp32_path = args.artifacts / "forecast_model_fp32.tflite"
    int8_path = args.artifacts / "forecast_model_int8.tflite"

    if model is None:
        raise FileNotFoundError("A loadable Keras model is required for quantization.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32 = converter.convert()
    fp32_path.write_bytes(fp32)

    def rep_dataset():
        if args.rep_mode == "zeros" or len(X_train) == 0:
            count = max(1, args.rep_samples)
            zero = np.zeros((1, window, len(feature_names)), dtype=np.float32)
            for _ in range(count):
                yield [zero]
            return
        if args.rep_mode == "range":
            flat = X_train.reshape(-1)
            quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            values = [float(np.quantile(flat, q)) for q in quantiles]
            values.append(0.0)
            ordered = []
            seen = set()
            for value in values:
                rounded = round(value, 6)
                if rounded in seen:
                    continue
                seen.add(rounded)
                ordered.append(value)
            for value in ordered:
                arr = np.full((1, window, len(feature_names)), value, dtype=np.float32)
                yield [arr]
            return
        count = min(len(X_train), args.rep_samples)
        for i in range(count):
            yield [X_train[i : i + 1]]

    def configure_int8(converter):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    configure_int8(converter)
    int8 = converter.convert()
    int8_path.write_bytes(int8)

    print("Saved:", fp32_path)
    print("Saved:", int8_path, f"({int8_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
