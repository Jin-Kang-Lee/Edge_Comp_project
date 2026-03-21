import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW_GROUPS: Dict[str, List[str]] = {
    "dist": ["dist_mean", "dist_std", "dist_min", "dist_max"],
    "accel": ["accel_mean", "accel_std", "accel_min", "accel_max"],
    "temp": ["temp_mean", "temp_std", "temp_min", "temp_max"],
    "hum": ["hum_mean", "hum_std", "hum_min", "hum_max"],
}

MASK_NAMES: Dict[str, str] = {
    "dist": "mask_dist",
    "accel": "mask_accel",
    "temp": "mask_temp",
    "hum": "mask_hum",
}

RAW_FILES: Dict[str, str] = {
    "dist": "LD2410.csv",
    "accel": "GY511.csv",
    "temp_hum": "AM2302.csv",
}


@dataclass
class ContractSchema:
    contract_version: str
    feature_names: List[str]
    mask_names: List[str]
    raw_feature_names: List[str]
    dropped_features: List[str]
    window_size: int

    def to_dict(self) -> Dict:
        return {
            "contract_version": self.contract_version,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "mask_names": self.mask_names,
            "raw_feature_names": self.raw_feature_names,
            "dropped_features": self.dropped_features,
            "window_size": self.window_size,
        }


def _load_sensor_csv(path: Path, feature_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    missing = sorted(set(feature_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[["timestamp"] + feature_cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["dup_idx"] = df.groupby("timestamp").cumcount()
    return df


def merge_sensors(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    dist_df = _load_sensor_csv(data_dir / RAW_FILES["dist"], RAW_GROUPS["dist"])
    accel_df = _load_sensor_csv(data_dir / RAW_FILES["accel"], RAW_GROUPS["accel"])
    temp_hum_df = _load_sensor_csv(
        data_dir / RAW_FILES["temp_hum"], RAW_GROUPS["temp"] + RAW_GROUPS["hum"]
    )

    merged = dist_df.merge(accel_df, on=["timestamp", "dup_idx"], how="outer")
    merged = merged.merge(temp_hum_df, on=["timestamp", "dup_idx"], how="outer")

    merged = merged.sort_values(["timestamp", "dup_idx"]).reset_index(drop=True)
    merged["timestamp_str"] = merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged = merged.drop(columns=["timestamp", "dup_idx"]).rename(
        columns={"timestamp_str": "timestamp"}
    )
    return merged


def add_masks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for group, cols in RAW_GROUPS.items():
        mask_name = MASK_NAMES[group]
        out[mask_name] = (~out[cols].isna().any(axis=1)).astype(np.int8)
    all_features = [c for cols in RAW_GROUPS.values() for c in cols]
    out[all_features] = out[all_features].fillna(0.0)
    return out


def select_features_by_variance(
    df: pd.DataFrame, feature_cols: List[str], k: int
) -> Tuple[List[str], List[str], Dict[str, float]]:
    variances = df[feature_cols].var().fillna(0.0)
    sorted_features = sorted(feature_cols, key=lambda c: (-variances[c], c))
    selected = sorted_features[:k]
    dropped = sorted_features[k:]
    return selected, dropped, variances.to_dict()


def build_contract_v1(
    df: pd.DataFrame, selected_features: List[str]
) -> pd.DataFrame:
    cols = ["timestamp"] + selected_features + list(MASK_NAMES.values())
    return df[cols].copy()


def split_time_series(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows to split")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def inject_anomalies(
    df: pd.DataFrame,
    feature_names: List[str],
    seed: int,
    target_frac: float,
    min_index: int = 0,
    min_len: int = 3,
    max_len: int = 8,
    severity: float = 1.0,
    fixed_scale: bool = False,
    force_all_features: bool = False,
    clip_nonneg: bool = True,
    easy_mode: bool = False,
    balance_types: bool = True,
    context_window: int = 32,
    correlate_temp_hum: bool = True,
    anomaly_types: list[str] | None = None,
    anomaly_type_weights: dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return df.copy(), np.zeros(0, dtype=np.int8)
    # Keep anomaly lengths reasonable for short splits.
    max_len = min(max_len, max(1, n // 5))
    min_len = max(1, min(min_len, max_len))
    target = max(1, int(n * target_frac))
    labels = np.zeros(n, dtype=np.int8)
    df_mod = df.copy()

    stats_mean = df[feature_names].mean()
    stats_std = df[feature_names].std().replace(0, 1.0)
    stats_min = df[feature_names].min()
    stats_max = df[feature_names].max()
    stats_range = stats_max - stats_min
    stats_range = stats_range.mask(stats_range == 0, stats_std)

    def local_stats(feature: str, start: int) -> tuple[float, float]:
        if context_window <= 0:
            mean = float(stats_mean[feature])
            std = float(stats_std[feature])
            return mean, std if std > 0 else 1.0
        lo = max(0, start - context_window)
        hi = min(n, start + context_window + 1)
        series = df[feature].iloc[lo:hi]
        mean = float(series.mean()) if len(series) else float(stats_mean[feature])
        std = float(series.std()) if len(series) else float(stats_std[feature])
        if not np.isfinite(mean):
            mean = float(stats_mean[feature])
        if not np.isfinite(std) or std == 0.0:
            std = float(stats_std[feature])
        return mean, std if std > 0 else 1.0

    group_map = {
        "dist": [f for f in feature_names if f.startswith("dist_")],
        "accel": [f for f in feature_names if f.startswith("accel_")],
        "temp": [f for f in feature_names if f.startswith("temp_")],
        "hum": [f for f in feature_names if f.startswith("hum_")],
    }
    available_groups = [g for g, feats in group_map.items() if feats]

    if anomaly_types is None:
        anomaly_types = ["spike", "drop", "noise", "drift", "stuck", "dropout"]
    else:
        anomaly_types = [str(t).strip() for t in anomaly_types if str(t).strip()]
    if not anomaly_types:
        anomaly_types = ["spike"]
    type_counts = {t: 0 for t in anomaly_types}
    weight_list = None
    if anomaly_type_weights:
        weight_list = [max(0.0, float(anomaly_type_weights.get(t, 0.0))) for t in anomaly_types]
        total_w = float(np.sum(weight_list))
        if total_w > 0:
            weight_list = [w / total_w for w in weight_list]
        else:
            weight_list = None
    attempts = 0
    max_attempts = target * 50

    def scale(low: float, high: float) -> float:
        if fixed_scale:
            return 0.5 * (low + high)
        return float(rng.uniform(low, high))

    if easy_mode:
        spike_range = (6.0, 10.0)
        drop_range = (5.0, 9.0)
        noise_range = (4.0, 7.0)
        drift_range = (5.0, 9.0)
        stuck_range = (5.0, 9.0)
        dropout_range = (6.0, 10.0)
    else:
        spike_range = (3.0, 6.0)
        drop_range = (2.0, 4.0)
        noise_range = (2.0, 4.0)
        drift_range = (3.0, 6.0)
        stuck_range = (0.0, 0.0)
        dropout_range = (0.0, 0.0)

    def strong_delta(feature: str, std: float) -> float:
        base = max(std, float(stats_std[feature]))
        rng = float(stats_range[feature])
        return max(base * 30.0, rng * 1.5)

    while labels.sum() < target and attempts < max_attempts:
        attempts += 1
        low = 0 if min_index >= n - 1 else min_index
        start = int(rng.integers(low, max(low + 1, n - 2)))
        length = int(rng.integers(min_len, max_len + 1))
        end = min(n, start + length)
        if labels[start:end].any():
            continue

        if force_all_features:
            feats = list(feature_names)
            mask_names = list(MASK_NAMES.values())
        else:
            use_group = rng.random() < 0.7 and bool(available_groups)
            if use_group:
                group = str(rng.choice(available_groups))
                feats = list(group_map[group])
                mask_names = [MASK_NAMES.get(group)] if MASK_NAMES.get(group) else []
                if (
                    correlate_temp_hum
                    and group in {"temp", "hum"}
                    and group_map.get("temp")
                    and group_map.get("hum")
                ):
                    feats = list(group_map["temp"]) + list(group_map["hum"])
                    mask_names = [MASK_NAMES["temp"], MASK_NAMES["hum"]]
            else:
                feats = [str(rng.choice(feature_names))]
                mask_names = []

        if weight_list is not None:
            mode = str(rng.choice(anomaly_types, p=weight_list))
        elif balance_types:
            min_count = min(type_counts.values())
            choices = [t for t, c in type_counts.items() if c == min_count]
            mode = str(rng.choice(choices))
        else:
            mode = str(rng.choice(anomaly_types))
        if mode == "dropout" and not mask_names:
            mode = str(rng.choice([t for t in anomaly_types if t != "dropout"]))

        end_excl = end
        sl = slice(start, end_excl)
        span = end_excl - start
        for feat in feats:
            mean, std = local_stats(feat, start)
            col_idx = df_mod.columns.get_loc(feat)
            if easy_mode:
                delta = strong_delta(feat, std) * severity
                direction = 1 if rng.random() < 0.5 else -1
                if mode == "noise":
                    noise = rng.normal(0.0, delta, size=span)
                    df_mod.iloc[sl, col_idx] += noise
                elif mode == "drift":
                    ramp = np.linspace(0.0, delta, span) * direction
                    df_mod.iloc[sl, col_idx] += ramp
                elif mode == "stuck":
                    df_mod.iloc[sl, col_idx] = mean + direction * delta
                elif mode == "dropout":
                    df_mod.iloc[sl, col_idx] = mean - delta
                elif mode == "drop":
                    df_mod.iloc[sl, col_idx] = mean - delta
                else:
                    df_mod.iloc[sl, col_idx] += direction * delta
                continue
            if mode == "spike":
                magnitude = scale(*spike_range) * std * severity
                df_mod.iloc[sl, col_idx] += magnitude * (
                    1 if rng.random() < 0.5 else -1
                )
            elif mode == "drop":
                df_mod.iloc[sl, col_idx] = mean - scale(*drop_range) * std * severity
            elif mode == "noise":
                noise = rng.normal(
                    0.0, scale(*noise_range) * std * severity, size=span
                )
                df_mod.iloc[sl, col_idx] += noise
            elif mode == "drift":
                ramp = np.linspace(0.0, scale(*drift_range) * std * severity, span)
                if rng.random() < 0.5:
                    ramp = -ramp
                df_mod.iloc[sl, col_idx] += ramp
            elif mode == "stuck":
                if easy_mode:
                    direction = 1 if rng.random() < 0.5 else -1
                    target = mean + direction * scale(*stuck_range) * std * severity
                    df_mod.iloc[sl, col_idx] = target
                else:
                    df_mod.iloc[sl, col_idx] = df_mod.loc[start, feat]
            elif mode == "dropout":
                if easy_mode:
                    df_mod.iloc[sl, col_idx] = mean - scale(*dropout_range) * std * severity
                else:
                    df_mod.iloc[sl, col_idx] = 0.0

        if mode == "dropout" and mask_names:
            for mask_name in mask_names:
                mask_idx = df_mod.columns.get_loc(mask_name)
                df_mod.iloc[sl, mask_idx] = 0

        labels[start:end] = 1
        type_counts[mode] += 1

    if clip_nonneg:
        non_neg = [
            f
            for f in feature_names
            if f.startswith("dist_")
            or f.startswith("accel_")
            or f.startswith("temp_")
            or f.startswith("hum_")
        ]
        df_mod[non_neg] = df_mod[non_neg].clip(lower=0.0)

    return df_mod, labels


def compute_scaler(train_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict]:
    mean = train_df[feature_names].mean()
    std = train_df[feature_names].std().replace(0, 1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict()}


def apply_scaler(
    df: pd.DataFrame, feature_names: List[str], scaler: Dict[str, Dict]
) -> np.ndarray:
    mean = np.array([scaler["mean"][f] for f in feature_names], dtype=np.float32)
    std = np.array([scaler["std"][f] for f in feature_names], dtype=np.float32)
    values = df[feature_names].to_numpy(dtype=np.float32)
    return (values - mean) / std


def make_windows(
    data: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(data) <= window:
        raise ValueError("Window size must be smaller than sequence length")
    x = []
    y = []
    for i in range(window, len(data)):
        x.append(data[i - window : i])
        y.append(data[i])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_windows_multi(
    data: np.ndarray, window: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(data) <= window + horizon - 1:
        raise ValueError("Window+horizon must be smaller than sequence length")
    x = []
    y = []
    for i in range(window, len(data) - horizon + 1):
        x.append(data[i - window : i])
        y.append(data[i : i + horizon])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
