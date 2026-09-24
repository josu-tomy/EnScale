"""
Evaluation metrics and historical baseline models for energy forecasting in EnScale.

Calculates:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CV(RMSE) (Coefficient of Variation of RMSE in %)
- sMAPE (Symmetric Mean Absolute Percentage Error in %)
- R² (Coefficient of Determination)
- improvement_vs_baseline_pct
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class SameHourHistoricalBaseline:
    """
    Non-ML historical benchmark baseline model.
    Predicts energy consumption based on same-hour and day-of-week historical averages from training data.
    """

    def __init__(self):
        self.lookup: Dict[tuple, float] = {}
        self.overall_mean: float = 0.0

    def fit(self, df: pd.DataFrame, target_col: str = "energy_kwh") -> "SameHourHistoricalBaseline":
        if "hour" not in df.columns or "day_of_week" not in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hours = ts.dt.hour
            dows = ts.dt.dayofweek
        else:
            hours = df["hour"]
            dows = df["day_of_week"]

        temp_df = pd.DataFrame({
            "hour": hours,
            "day_of_week": dows,
            "target": df[target_col],
        })

        grouped = temp_df.groupby(["hour", "day_of_week"])["target"].mean()
        self.lookup = grouped.to_dict()
        self.overall_mean = float(df[target_col].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "hour" not in df.columns or "day_of_week" not in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hours = ts.dt.hour
            dows = ts.dt.dayofweek
        else:
            hours = df["hour"]
            dows = df["day_of_week"]

        preds = []
        for h, d in zip(hours, dows):
            val = self.lookup.get((h, d), self.overall_mean)
            preds.append(val)

        return np.array(preds, dtype=float)


def calculate_forecast_metrics(
    actual: Union[pd.Series, np.ndarray],
    predicted: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Computes standard required metrics:
    - MAE
    - RMSE
    - CV(RMSE)
    - sMAPE
    - R²
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    mean_actual = float(np.mean(y_true))
    cv_rmse = float((rmse / mean_actual * 100.0) if mean_actual != 0 else 0.0)

    # Symmetric Mean Absolute Percentage Error (sMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid_mask = denominator > 0
    if valid_mask.any():
        smape = float(np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]) / denominator[valid_mask]) * 100.0)
    else:
        smape = 0.0

    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "CV(RMSE)": round(cv_rmse, 2),
        "sMAPE": round(smape, 2),
        "R2": round(r2, 4),
    }


def compare_models(
    baseline_metrics: Dict[str, float],
    ml_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compares Baseline vs ML and calculates improvement_vs_baseline_pct.
    """
    base_mae = baseline_metrics["MAE"]
    ml_mae = ml_metrics["MAE"]

    improvement_pct = (
        round(float((base_mae - ml_mae) / base_mae * 100.0), 2)
        if base_mae > 0
        else 0.0
    )

    return {
        "baseline_metrics": baseline_metrics,
        "ml_metrics": ml_metrics,
        "improvement_vs_baseline_pct": improvement_pct,
        "ml_outperformed_baseline": bool(ml_mae < base_mae),
    }


def compute_forecast_error(
    actual_energy_kwh: float,
    predicted_energy_kwh: float,
) -> Dict[str, float]:
    """
    Canonical forecast error computation:
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
