from pathlib import Path

# Project-relative paths
DATA_ARTIFACTS_DIR = Path("data_artifacts")
INPUT_NORMAL_CSV = DATA_ARTIFACTS_DIR / "mock_normal_norm.csv"
OUTPUT_MODEL_DIR = DATA_ARTIFACTS_DIR / "model_output"

# Contract: 13 features + 4 masks
FEATURE_DIM = 17
