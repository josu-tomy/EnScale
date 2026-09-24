"""
Preprocessing and feature engineering module for EnScale ML pipelines.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


def prepare_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts standard cyclical and calendar time features from the canonical timestamp column.
    """
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("DataFrame must contain canonical 'timestamp' column")

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = out["timestamp"].dt.month

    # Cyclical hour encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    return out


def split_features_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "energy_kwh",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target vector y.
    Target must be canonical energy_kwh.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    if feature_cols is None:
        candidate_cols = ["hour", "dayofweek", "is_weekend", "month", "hour_sin", "hour_cos", "temperature_c", "occupancy"]
        feature_cols = [c for c in candidate_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
