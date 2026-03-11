# SVDD Setup

## 1. Install Python

Install Python 3.11. After installation, verify:

```powershell
python --version
```

## 2. Create a virtual environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r svdd\requirements.txt
```

## 3. Run the real-data pipeline

```powershell
python svdd\run_real_pipeline.py --drop-am2302-zero-rows
```

## 4. Add labels and rerun evaluation

Edit `svdd\data_artifacts\real_eval_labels.csv`, mark anomaly rows with `label=1`, then rerun:

```powershell
python svdd\run_real_pipeline.py --drop-am2302-zero-rows --skip-train
```
