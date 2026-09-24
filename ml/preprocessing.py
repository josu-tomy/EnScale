"""
Preprocessing and feature engineering module for EnScale ML pipelines.

Strictly follows:
- Temporal features: hour, day_of_week, day_of_month, month, is_weekend
- Environmental & building metadata features (where available):
  air_temperature, dew_temperature, building_area, primary_use, year_built, floor_count
- Chronological splitting (70% train, 15% validation, 15% final test)
- Strict leakage avoidance (no future energy measurements or future target leakage)
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates required model features:
    - hour
    - day_of_week
    - day_of_month
    - month
    - is_weekend
    - cyclical time encodings (hour_sin, hour_cos)
    - environmental/building metadata where available:
      air_temperature, dew_temperature, building_area, primary_use, year_built, floor_count
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("DataFrame must contain canonical 'timestamp' column")

    out["timestamp"] = pd.to_datetime(out["timestamp"])

    # Required temporal features
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["day_of_month"] = out["timestamp"].dt.day
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # Cyclical hour features
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    # Map temperature column aliases if present
    if "temperature_c" in out.columns and "air_temperature" not in out.columns:
        out["air_temperature"] = out["temperature_c"]

    if "net_built_up_area_m2" in out.columns and "building_area" not in out.columns:
        out["building_area"] = out["net_built_up_area_m2"]

    return out


# Backward compatibility alias
prepare_time_features = prepare_features


def chronological_time_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically without shuffling or lookahead leakage:
    - Exactly 70% train
    - Exactly 15% validation
    - Exactly 15% final test

    The final test set remains untouched until final evaluation.
    """
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-5:
        raise ValueError("train_pct + val_pct + test_pct must sum to 1.0")

    # Ensure chronological order
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_df)

    n_train = int(np.floor(n * train_pct))
    n_val = int(np.floor(n * val_pct))

    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train : n_train + n_val].copy()
    test_df = sorted_df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df
