"""
Anomaly and operational energy waste detection service for EnScale.

Enforces:
1. Transparent residual-based method:
   - forecast_error_kwh = actual_energy_kwh - predicted_energy_kwh
   - forecast_error_pct = (abs(forecast_error_kwh) / actual_energy_kwh) * 100
2. Rolling residual distribution + robust thresholding using Median and Median Absolute Deviation (MAD).
3. Output schema includes:
   timestamp, actual_energy_kwh, predicted_energy_kwh, forecast_error_kwh,
   forecast_error_pct, anomaly_flag, severity.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from config.constants import DEFAULT_TARIFF_INR_PER_KWH
from domain.models import AnomalyResult


def calculate_forecast_errors(
    actual: Union[pd.Series, np.ndarray, list],
    predicted: Union[pd.Series, np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes forecast_error_kwh and forecast_error_pct with safe numerical handling.
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)

    error_kwh = actual_arr - pred_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual_arr > 0, (np.abs(error_kwh) / actual_arr) * 100.0, 0.0)
        error_pct = np.nan_to_num(error_pct, nan=0.0, posinf=0.0, neginf=0.0)

    return error_kwh, error_pct


def detect_anomalies(
    actual_energy_kwh: Union[pd.Series, np.ndarray, list],
    predicted_energy_kwh: Union[pd.Series, np.ndarray, list],
    timestamp: Union[pd.Series, np.ndarray, list],
    window_size: int = 24,
    min_periods: int = 6,
    threshold_sigma: float = 2.5,
    min_waste_kwh: float = 4.0,
    tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH,
) -> pd.DataFrame:
    """
    Transparent residual-based anomaly detection using rolling residual distribution
    and robust Median Absolute Deviation (MAD) scale estimation.

    Parameters:
        actual_energy_kwh: Actual measured interval energy (kWh)
        predicted_energy_kwh: Expected or predicted interval energy (kWh)
        timestamp: Interval timestamps
        window_size: Number of periods for rolling window (default 24 hours)
        min_periods: Minimum observations required in rolling window
        threshold_sigma: Robust z-score threshold to flag anomaly
        min_waste_kwh: Minimum excess kWh required to avoid false positives on low loads
        tariff_inr_per_kwh: Electricity tariff for computing potential waste costs

    Returns:
        DataFrame with required fields:
        timestamp, actual_energy_kwh, predicted_energy_kwh, forecast_error_kwh,
        forecast_error_pct, anomaly_flag, severity.
    """
    n = len(actual_energy_kwh)
    actual_series = pd.Series(actual_energy_kwh, dtype=float)
    pred_series = pd.Series(predicted_energy_kwh, dtype=float)
    ts_series = pd.Series(timestamp)

    # 1. Calculate forecast errors
    error_kwh, error_pct = calculate_forecast_errors(actual_series, pred_series)
    err_series = pd.Series(error_kwh)

    # 2. Rolling residual distribution (Robust statistics)
    rolling_median = err_series.rolling(window=window_size, min_periods=min_periods).median()
    # Backward fill initial window points using global median
    global_median = float(err_series.median())
    rolling_median = rolling_median.bfill().fillna(global_median)

    # Rolling MAD (Median Absolute Deviation)
    rolling_abs_diff = (err_series - rolling_median).abs()
    rolling_mad = rolling_abs_diff.rolling(window=window_size, min_periods=min_periods).median()
    global_mad = float(rolling_abs_diff.median())
    if global_mad <= 0 or np.isnan(global_mad):
        global_mad = float(err_series.std()) if len(err_series) > 1 else 1.0
        if global_mad <= 0 or np.isnan(global_mad):
            global_mad = 1.0

    rolling_mad = rolling_mad.bfill().fillna(global_mad)
    # Normal consistency scale factor: sigma ~= 1.4826 * MAD
    rolling_scale = np.maximum(rolling_mad * 1.4826, 1.5)

    # Robust residual z-score (only positive deviations indicate energy waste)
    robust_z = (err_series - rolling_median) / rolling_scale

    # 3. Anomaly flagging and severity classification
    anomaly_flags = (err_series > min_waste_kwh) & (robust_z >= threshold_sigma)

    severities = []
    waste_kwh_list = []
    waste_cost_list = []

    for is_anom, err_val, z_val in zip(anomaly_flags, err_series, robust_z):
        if is_anom:
            waste_kwh_list.append(round(float(err_val), 2))
            waste_cost_list.append(round(float(err_val * tariff_inr_per_kwh), 2))
            if z_val >= threshold_sigma * 1.8 or err_val >= 35.0:
                severities.append("high")
            elif z_val >= threshold_sigma * 1.3 or err_val >= 18.0:
                severities.append("medium")
            else:
                severities.append("low")
        else:
            waste_kwh_list.append(0.0)
            waste_cost_list.append(0.0)
            severities.append("normal")

    out_df = pd.DataFrame({
        "timestamp": ts_series,
        "actual_energy_kwh": actual_series.round(2),
        "predicted_energy_kwh": pred_series.round(2),
        "forecast_error_kwh": np.round(error_kwh, 2),
        "forecast_error_pct": np.round(error_pct, 2),
        "anomaly_flag": anomaly_flags.astype(bool),
        "severity": severities,
        "waste_kwh": waste_kwh_list,
        "potential_waste_cost_inr": waste_cost_list,
        "robust_z_score": robust_z.round(2),
    })

    return out_df


class AnomalyService:
    """Service for detecting energy anomalies and quantifying operational waste."""

    def __init__(
        self,
        default_tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH,
        window_size: int = 24,
        threshold_sigma: float = 2.5,
    ):
        self.tariff_inr_per_kwh = default_tariff_inr_per_kwh
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

    def detect_anomalies(
        self,
        actual_energy_kwh: Union[pd.Series, np.ndarray, list],
        predicted_energy_kwh: Union[pd.Series, np.ndarray, list],
        timestamp: Union[pd.Series, np.ndarray, list],
    ) -> pd.DataFrame:
        """
        Executes rolling residual distribution anomaly detection.
        """
        return detect_anomalies(
            actual_energy_kwh=actual_energy_kwh,
            predicted_energy_kwh=predicted_energy_kwh,
            timestamp=timestamp,
            window_size=self.window_size,
            threshold_sigma=self.threshold_sigma,
            tariff_inr_per_kwh=self.tariff_inr_per_kwh,
        )

    def detect_waste(
        self,
        actual_series: pd.Series,
        expected_series: pd.Series,
        timestamps: Optional[pd.Series] = None,
        threshold_std: float = 2.0,
    ) -> List[AnomalyResult]:
        """
        Backward-compatible method returning list of AnomalyResult domain objects.
        """
        ts = timestamps if timestamps is not None else pd.Series(range(len(actual_series)))
        df = detect_anomalies(
            actual_energy_kwh=actual_series,
            predicted_energy_kwh=expected_series,
            timestamp=ts,
            threshold_sigma=threshold_std,
            tariff_inr_per_kwh=self.tariff_inr_per_kwh,
        )

        results: List[AnomalyResult] = []
        for _, row in df.iterrows():
            results.append(
                AnomalyResult(
                    timestamp=row["timestamp"],
                    actual_energy_kwh=float(row["actual_energy_kwh"]),
                    predicted_energy_kwh=float(row["predicted_energy_kwh"]),
                    is_anomaly=bool(row["anomaly_flag"]),
                    forecast_error_kwh=float(row["forecast_error_kwh"]),
                    forecast_error_pct=float(row["forecast_error_pct"]),
                    anomaly_flag=bool(row["anomaly_flag"]),
                    anomaly_score=float(row.get("robust_z_score", 0.0)),
                    waste_kwh=float(row["waste_kwh"]),
                    potential_waste_cost_inr=float(row["potential_waste_cost_inr"]),
                    severity=str(row["severity"]),
                    description="Excess energy detected" if row["anomaly_flag"] else "Normal operation",
                )
            )
        return results
