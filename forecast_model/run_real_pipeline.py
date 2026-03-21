from __future__ import annotations

import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def run_script(script_name: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(THIS_DIR / script_name), *extra_args]
    subprocess.run(cmd, check=True)


def main() -> None:
    run_script("validate_dataset.py", ["--data-dir", str(REPO_ROOT)])
    run_script("train_and_threshold.py", ["--data-dir", str(REPO_ROOT)])
    run_script("quantize_model.py", [])


if __name__ == "__main__":
    main()
