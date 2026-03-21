# forecast_model

This is the forecast anomaly model project.

Main layout:

- `validate_dataset.py`
- `train_and_threshold.py`
- `evaluate_model.py`
- `quantize_model.py`
- `run_real_pipeline.py`
- `data_real/`
- `model_output/`

Extra helpers from your current workflow:

- `pipeline.py`
- `optuna_tune.py`
- `run_best_from_optuna.py`

Repo root carries the raw sensor CSVs used by this forecast model:

- `AM2302.csv`
- `GY511.csv`
- `LD2410.csv`
