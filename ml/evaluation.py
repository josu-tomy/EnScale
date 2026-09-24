"""
Evaluation metrics for energy forecast models in EnScale.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_forecast_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Computes standard evaluation metrics for forecast accuracy.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)

    # Safe MAPE calculation (avoiding division by zero)
    mask = actual > 0
    if mask.any():
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": mape,
    }


def compute_forecast_error(
    actual_energy_kwh: float,
    predicted_energy_kwh: float,
) -> Dict[str, float]:
    """
    Calculates canonical forecast error metrics:
    - forecast_error_kwh = actual_energy_kwh - predicted_energy_kwh
    - forecast_error_pct = (abs(forecast_error_kwh) / actual_energy_kwh) * 100
    """
    err_kwh = float(actual_energy_kwh - predicted_energy_kwh)
    err_pct = (
        float((abs(err_kwh) / actual_energy_kwh) * 100.0)
        if actual_energy_kwh > 0
        else 0.0
    )
    return {
        "forecast_error_kwh": err_kwh,
        "forecast_error_pct": err_pct,
    }
