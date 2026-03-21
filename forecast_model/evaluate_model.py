from __future__ import annotations

import argparse
import json
from pathlib import Path

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric_block(name: str, values: dict) -> str:
    return (
        f"{name:5} "
        f"acc={values.get('accuracy', 0.0):.4f} "
        f"precision={values.get('precision', 0.0):.4f} "
        f"recall={values.get('recall', 0.0):.4f} "
        f"f1={values.get('f1', 0.0):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the saved evaluation summary for the forecast model."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parent / "model_output",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts
    metrics_path = artifacts_dir / "train_report.json"
    threshold_path = artifacts_dir / "thresholds.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing threshold file: {threshold_path}")

    metrics = load_json(metrics_path)
    threshold = load_json(threshold_path)

    print("Forecast model evaluation summary")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Model type: {metrics.get('model_type', 'unknown')}")
    print(f"Threshold: {threshold.get('threshold', 'n/a')}")
    print(f"Score aggregation: {metrics.get('score_agg', 'n/a')}")
    print(f"Error normalization: {metrics.get('error_norm', 'n/a')}")
    print()
    print(format_metric_block("train", metrics.get("train", {})))
    print(format_metric_block("calib", metrics.get("calib", {})))
    print(format_metric_block("val", metrics.get("val", {})))
    print(format_metric_block("test", metrics.get("test", {})))


if __name__ == "__main__":
    main()
