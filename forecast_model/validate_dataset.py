import argparse
from pathlib import Path

import pandas as pd

from pipeline import (
    RAW_GROUPS,
    MASK_NAMES,
    add_masks,
    build_contract_v1,
    merge_sensors,
    save_json,
    select_features_by_variance,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contract_v1 dataset.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR / "data_real")
    parser.add_argument("--feature-count", type=int, default=13)
    parser.add_argument(
        "--drop-features",
        type=str,
        default="accel_mean,accel_std,accel_max",
        help="Comma-separated feature names to drop before selection.",
    )
    args = parser.parse_args()

    merged = merge_sensors(args.data_dir)
    merged = add_masks(merged)

    raw_features = [c for cols in RAW_GROUPS.values() for c in cols]
    variances = merged[raw_features].var().fillna(0.0).to_dict()
    drop_list = [s.strip() for s in args.drop_features.split(",") if s.strip()]
    if drop_list:
        unknown = sorted(set(drop_list) - set(raw_features))
        if unknown:
            raise ValueError(f"Unknown drop features: {unknown}")
        selected = [f for f in raw_features if f not in drop_list]
        if len(selected) != args.feature_count:
            raise ValueError(
                f"Selected {len(selected)} features after drop, expected {args.feature_count}."
            )
        dropped = drop_list
    else:
        selected, dropped, _ = select_features_by_variance(
            merged, raw_features, args.feature_count
        )
    contract_df = build_contract_v1(merged, selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.out_dir / "contract_v1.csv"
    contract_df.to_csv(contract_path, index=False)

    schema = {
        "contract_version": "v1",
        "feature_count": len(selected),
        "feature_names": selected,
        "mask_names": list(MASK_NAMES.values()),
        "raw_feature_names": raw_features,
        "dropped_features": dropped,
        "variance": variances,
    }
    save_json(args.out_dir / "contract_v1_schema.json", schema)

    summary = contract_df.describe(include="all")
    summary_path = args.out_dir / "contract_v1_summary.csv"
    summary.to_csv(summary_path)

    print("contract_v1 dataset saved:", contract_path)
    print("schema saved:", args.out_dir / "contract_v1_schema.json")
    print("summary saved:", summary_path)
    print("rows:", len(contract_df))
    print("features:", selected)


if __name__ == "__main__":
    main()
