"""
quantize_model.py
=================
Task 3 of the INF2009 Software-Defined Sensor pipeline.

Steps performed inside this script:
  Step 1 — Model Conversion   : Load autoencoder.keras → TFLite converter
  Step 2 — INT8 Quantization  : Apply Full Integer Quantization (weights + activations)
  Step 3 — Calibration        : Feed representative samples from normal training data
                                so the converter can calculate the correct INT8 scale/zero-point

Output:
  - autoencoder_int8.tflite   : The quantized flatbuffer model ready for TFLM on the Pico 2.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Contract v1 — must match the rest of the pipeline
FEATURE_COLS = [f"f{i}" for i in range(13)]


# ---------------------------------------------------------------------------
# Step 3: Calibration — representative dataset generator
# ---------------------------------------------------------------------------
def make_representative_dataset(csv_path: str, num_samples: int = 500):
    """
    Returns a generator function that yields representative input samples.

    The TFLite converter calls this generator to observe the range of real
    values passing through each layer. It uses those ranges to choose the
    correct INT8 scale factor and zero-point so accuracy is preserved.

    Rules:
      - Must yield a LIST containing ONE numpy array of shape (1, input_dim).
      - dtype must be float32.
      - Ideally 100-500 diverse samples from your normal training data.
    """
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    # Randomly pick num_samples rows so the calibration covers diverse inputs
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=min(num_samples, len(X)), replace=False)

    def generator():
        for i in indices:
            # Shape must be (1, 13) — single unbatched sample
            yield [np.expand_dims(X[i], axis=0)]

    return generator


def main():
    ap = argparse.ArgumentParser(description="Quantize autoencoder to INT8 TFLite flatbuffer.")
    ap.add_argument(
        "--model_path",
        default="data_mock/model_output/autoencoder.keras",
        help="Path to the trained autoencoder.keras from train_and_threshold.py",
    )
    ap.add_argument(
        "--rep_data_csv",
        default="data_mock/mock_normal_norm.csv",
        help="Normalized normal-only CSV used for INT8 calibration (same data used for training)",
    )
    ap.add_argument(
        "--out_dir",
        default="data_mock/model_output",
        help="Directory where autoencoder_int8.tflite will be saved",
    )
    ap.add_argument(
        "--num_calibration_samples",
        type=int,
        default=500,
        help="How many samples to use for INT8 calibration (100-500 is sufficient)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = out_dir / "autoencoder_int8.tflite"

    # ------------------------------------------------------------------
    # Step 1: Model Conversion — Load Keras and point the TFLite Converter
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading Keras model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    model.summary()

    print("\n[Step 1] Initializing TFLite Converter from Keras model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ------------------------------------------------------------------
    # Step 2: INT8 Quantization — configure the converter for Full Integer
    # ------------------------------------------------------------------
    print("\n[Step 2] Configuring Full INT8 Quantization...")

    # Tell the converter to apply post-training quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Restrict to ONLY INT8 ops — ensures the generated model is
    # executable purely in integer arithmetic on the Pico 2 Cortex-M33
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Force input and output tensors to INT8 as well.
    # This is important! Without this, the model still does float32 at the
    # boundary (wrapped INT8 inside) which wastes cycles on the MCU.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # ------------------------------------------------------------------
    # Step 3: Calibration — attach the representative dataset generator
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Attaching calibration data from: {args.rep_data_csv}")
    print(f"         Using {args.num_calibration_samples} representative samples...")
    converter.representative_dataset = make_representative_dataset(
        args.rep_data_csv,
        num_samples=args.num_calibration_samples,
    )

    # ------------------------------------------------------------------
    # Convert! — this runs calibration + quantization internally
    # ------------------------------------------------------------------
    print("\n[Converting] Running calibration and quantization pass...")
    tflite_model = converter.convert()
    print("[Converting] Done.")

    # ------------------------------------------------------------------
    # Save the .tflite flatbuffer
    # ------------------------------------------------------------------
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Size report
    original_size = Path(args.model_path).stat().st_size
    quantized_size = tflite_path.stat().st_size
    reduction = (1 - quantized_size / original_size) * 100

    print(f"\n{'='*50}")
    print(f"[SUCCESS] Saved: {tflite_path}")
    print(f"  Original .keras size : {original_size / 1024:.2f} KB")
    print(f"  Quantized .tflite    : {quantized_size / 1024:.2f} KB")
    print(f"  Size reduction       : {reduction:.1f}%")
    print(f"{'='*50}")

    if quantized_size <= 50 * 1024:
        print("[OK] Model fits comfortably within Pico 2 Flash/SRAM budget.")
    else:
        print("[WARNING] Model may be too large. Consider reducing layer sizes in build_autoencoder().")

    print("\nNext step: run publish_mqtt.py to push this .tflite to the Pico 2 over MQTT.")


if __name__ == "__main__":
    main()
