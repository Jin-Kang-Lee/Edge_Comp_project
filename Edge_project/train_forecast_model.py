#!/usr/bin/env python3
"""Train a lightweight temporal prediction MLP and derive anomaly threshold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model(input_size: int, output_size: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,), name="input"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
            tf.keras.layers.Dense(output_size, activation="linear", name="pred_next"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def main() -> None:
    tf.random.set_seed(42)
    np.random.seed(42)

    root = Path(__file__).resolve().parent
    data_dir = root / "data_mock"
    model_dir = root / "model_output"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_npz = np.load(data_dir / "mock_windows_train.npz")
    test_npz = np.load(data_dir / "mock_windows_test.npz")

    X_train = train_npz["X"].astype(np.float32)
    Y_train = train_npz["Y"].astype(np.float32)
    X_test = test_npz["X"].astype(np.float32)
    Y_test = test_npz["Y"].astype(np.float32)

    n_features = X_train.shape[2]
    input_size = X_train.shape[1] * n_features

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    model = build_model(input_size=input_size, output_size=n_features)

    history = model.fit(
        X_train_flat,
        Y_train,
        validation_data=(X_test_flat, Y_test),
        epochs=20,
        batch_size=64,
        verbose=0,
    )

    pred_train = model.predict(X_train_flat, verbose=0)
    train_mse = np.mean((Y_train - pred_train) ** 2, axis=1)
    threshold_p99 = float(np.percentile(train_mse, 99))

    model_path = model_dir / "forecast_model.keras"
    threshold_path = model_dir / "thresholds.json"
    report_path = model_dir / "train_report.json"

    model.save(model_path)

    with threshold_path.open("w", encoding="utf-8") as f:
        json.dump({"mse_p99": threshold_p99}, f, indent=2)

    report = {
        "input_size": int(input_size),
        "output_size": int(n_features),
        "train_samples": int(len(X_train_flat)),
        "test_samples": int(len(X_test_flat)),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "threshold_mse_p99": threshold_p99,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {model_path}")
    print(f"Saved {threshold_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
