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
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Contract v2 — must match the rest of the pipeline
FEATURE_COLS = [f"f{i}" for i in range(16)]


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
            # Shape must be (1, 16) — single unbatched sample
            yield [np.expand_dims(X[i], axis=0)]

    return generator


def main():
    ap = argparse.ArgumentParser(description="Quantize autoencoder to INT8 TFLite flatbuffer.")
    ap.add_argument(
        "--model_path",
        default="data_real_prepared/model_output/autoencoder.keras",
        help="Path to the trained autoencoder.keras from train_and_threshold.py",
    )
    ap.add_argument(
        "--rep_data_csv",
        default="data_real_prepared/real_normal_norm.csv",
        help="IMPORTANT: Must be the NORMALIZED normal-only CSV (real_normal_norm.csv), "
             "not the raw CSV. The calibration must see Z-score scaled values "
             "matching what the model was trained on.",
    )
    ap.add_argument(
        "--out_dir",
        default="data_real_prepared/model_output",
        help="Directory where autoencoder_int8.tflite will be saved",
    )
    ap.add_argument(
        "--thresholds_json",
        default="data_real_prepared/model_output/thresholds.json",
        help="Path to thresholds.json produced by train_and_threshold.py",
    )
    ap.add_argument(
        "--norm_stats_json",
        default="data_real_prepared/norm_stats.json",
        help="Path to norm_stats.json produced by normalize_dataset.py",
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
    print(f"         Using 500 representative samples...")
    converter.representative_dataset = make_representative_dataset(
        args.rep_data_csv,
        num_samples=500,
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

    # ------------------------------------------------------------------
    # Copy deployment artifacts (thresholds + norm stats) to out_dir
    # so your friend has a single folder with everything needed for inference
    # ------------------------------------------------------------------
    thresh_dst = out_dir / "thresholds.json"
    stats_dst  = out_dir / "norm_stats.json"

    if Path(args.thresholds_json).exists():
        if Path(args.thresholds_json).resolve() != thresh_dst.resolve():
            shutil.copy2(args.thresholds_json, thresh_dst)
            print(f"\n[Artifacts] Copied thresholds  → {thresh_dst}")
        else:
            print(f"\n[Artifacts] thresholds.json already in output dir: {thresh_dst}")
    else:
        print(f"[WARNING] thresholds.json not found at {args.thresholds_json}. "
              "Your friend will need this file to run inference!")

    if Path(args.norm_stats_json).exists():
        if Path(args.norm_stats_json).resolve() != stats_dst.resolve():
            shutil.copy2(args.norm_stats_json, stats_dst)
            print(f"[Artifacts] Copied norm stats   → {stats_dst}")
        else:
            print(f"[Artifacts] norm_stats.json already in output dir: {stats_dst}")
    else:
        print(f"[WARNING] norm_stats.json not found at {args.norm_stats_json}. "
              "Your friend will need this file to pre-process raw sensor inputs!")

    # Print threshold values so your friend knows the anomaly detection cutoff
    if thresh_dst.exists():
        thresholds = json.loads(thresh_dst.read_text())["thresholds_by_mask"]
        print("\n[Thresholds] Anomaly detection cutoffs (MSE > threshold = anomaly):")
        for mask_pattern, thr in thresholds.items():
            print(f"  mask={mask_pattern} → {thr:.6f}")

    print(f"\n{'='*50}")
    print(f"[SUCCESS] Saved: {tflite_path}")
    print(f"  Original .keras size : {original_size / 1024:.2f} KB")
    print(f"  Quantized .tflite    : {quantized_size / 1024:.2f} KB")
    print(f"  Size reduction       : {reduction:.1f}%")
    print(f"{'='*50}")

    # Pico 2 has 520 KB SRAM; 200 KB is a safe conservative limit for the model
    if quantized_size <= 200 * 1024:
        print("[OK] Model fits comfortably within Pico 2 Flash/SRAM budget (< 200 KB).")
    else:
        print("[WARNING] Model may be too large for Pico 2. Consider reducing layer sizes in build_autoencoder().")

    print("\n--- Files to send ---")
    print(f"  1. {tflite_path}")
    print(f"  2. {thresh_dst}")
    print(f"  3. {stats_dst}")
    print("\nNext step: run publish_mqtt.py to push this .tflite to the Pico 2 over MQTT.")


if __name__ == "__main__":
    main()
