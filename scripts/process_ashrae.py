"""
Deterministic subset processor for ASHRAE Great Energy Predictor III dataset.

Reads from data/raw/:
- train.csv
- building_metadata.csv
- weather_train.csv

Generates:
- data/reference/ashrae_subset.csv
- data/reference/dataset_manifest.json

Requirements:
- Electricity meter data only (meter == 0)
- Selected deterministic building IDs
- Merged with building metadata and weather data
- Canonical target column: energy_kwh (mapped from meter_reading)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from config.settings import RAW_DATA_DIR, REFERENCE_DATA_DIR

# Deterministic building selection (Commercial Office / Education buildings)
TARGET_BUILDING_IDS = [0, 1, 2, 3, 4]


def process_ashrae_subset() -> Path:
    train_path = RAW_DATA_DIR / "train.csv"
    meta_path = RAW_DATA_DIR / "building_metadata.csv"
    weather_path = RAW_DATA_DIR / "weather_train.csv"

    missing = [str(p.name) for p in [train_path, meta_path, weather_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required ASHRAE raw files in {RAW_DATA_DIR}: {', '.join(missing)}. "
            "Please download them from Kaggle (https://www.kaggle.com/c/ashrae-energy-prediction/data)."
        )

    print("Loading building metadata...")
    meta_df = pd.read_csv(meta_path)
    selected_meta = meta_df[meta_df["building_id"].isin(TARGET_BUILDING_IDS)].copy()
    site_ids = selected_meta["site_id"].unique()

    print("Loading weather data...")
    weather_df = pd.read_csv(weather_path)
    selected_weather = weather_df[weather_df["site_id"].isin(site_ids)].copy()

    print("Loading electricity meter data (meter == 0)...")
    chunks = []
    for chunk in pd.read_csv(train_path, chunksize=100_000):
        filtered = chunk[
            (chunk["building_id"].isin(TARGET_BUILDING_IDS)) &
            (chunk["meter"] == 0)  # Electricity meter
        ]
        if not filtered.empty:
            chunks.append(filtered)

    meter_df = pd.concat(chunks, ignore_index=True)

    # Merge meter with metadata and weather
    merged = meter_df.merge(selected_meta, on="building_id", how="left")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    selected_weather["timestamp"] = pd.to_datetime(selected_weather["timestamp"])
    merged = merged.merge(selected_weather, on=["site_id", "timestamp"], how="left")

    # Map to canonical variable names
    # meter_reading -> energy_kwh
    merged["energy_kwh"] = merged["meter_reading"]
    if "square_feet" in merged.columns:
        # Convert sqft to m2
        merged["net_built_up_area_m2"] = merged["square_feet"] * 0.092903
        merged["building_area"] = merged["net_built_up_area_m2"]

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Save processed subset
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REFERENCE_DATA_DIR / "ashrae_subset.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Processed ASHRAE subset saved to {out_csv} ({len(merged)} rows)")

    # Create dataset manifest
    manifest = {
        "dataset_name": "ASHRAE Great Energy Predictor III (Deterministic Commercial Subset)",
        "source": "https://www.kaggle.com/c/ashrae-energy-prediction/data",
        "download_date": datetime.now().strftime("%Y-%m-%d"),
        "meter_type": "electricity (meter == 0)",
        "selection_rule": "meter == 0 AND building_id in [0, 1, 2, 3, 4] merged with site weather",
        "building_ids": TARGET_BUILDING_IDS,
        "rows_selected": len(merged),
        "feature_columns": [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "air_temperature",
            "dew_temperature",
            "building_area",
            "primary_use",
            "year_built",
            "floor_count",
        ],
        "target_column": "energy_kwh",
    }

    manifest_path = REFERENCE_DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Dataset manifest written to {manifest_path}")

    return out_csv


if __name__ == "__main__":
    process_ashrae_subset()
