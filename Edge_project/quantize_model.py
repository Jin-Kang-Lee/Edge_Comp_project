#!/usr/bin/env python3
"""Convert trained Keras model to fully quantized INT8 TFLite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def representative_dataset(X: np.ndarray):
    for i in range(min(300, len(X))):
        yield [X[i : i + 1].astype(np.float32)]


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"
    model_dir = root / "model_output"

    keras_model = tf.keras.models.load_model(model_dir / "forecast_model.keras")

    train_npz = np.load(data_dir / "mock_windows_train.npz")
    X_train = train_npz["X"].astype(np.float32).reshape(len(train_npz["X"]), -1)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = model_dir / "forecast_int8.tflite"
    with out_path.open("wb") as f:
        f.write(tflite_model)

    size_bytes = out_path.stat().st_size
    print(f"Saved {out_path} ({size_bytes} bytes)")
    if size_bytes > 10 * 1024:
        print("WARNING: Model exceeds ~10KB target. Consider reducing hidden units.")


if __name__ == "__main__":
    main()
