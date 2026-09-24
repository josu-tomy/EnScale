"""
Anomaly and energy waste detection service for EnScale.

Compares actual vs expected energy consumption to isolate operational waste.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from domain.models import AnomalyResult
from config.constants import DEFAULT_TARIFF_INR_PER_KWH


class AnomalyService:
    """Service skeleton for detecting anomalies and quantifying energy waste."""

    def __init__(self, default_tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH):
        self.tariff_inr_per_kwh = default_tariff_inr_per_kwh

    def detect_waste(
        self,
        actual_series: pd.Series,
        expected_series: pd.Series,
        timestamps: Optional[pd.Series] = None,
        threshold_std: float = 2.0,
    ) -> List[AnomalyResult]:
        """
        Detects positive deviations (actual > expected) representing operational waste.
        """
        results: List[AnomalyResult] = []
        residuals = actual_series - expected_series
        std_residual = residuals.std() if len(residuals) > 1 else 1.0
        if std_residual == 0.0 or pd.isna(std_residual):
            std_residual = 1.0

        for idx in range(len(actual_series)):
            actual = float(actual_series.iloc[idx])
            expected = float(expected_series.iloc[idx])
            diff = actual - expected
            score = diff / std_residual
            is_anomaly = bool(diff > 0 and score > threshold_std)
            waste_kwh = max(0.0, diff) if is_anomaly else 0.0
            waste_cost = waste_kwh * self.tariff_inr_per_kwh
            ts = timestamps.iloc[idx] if timestamps is not None else idx

            severity = "low"
            if score > threshold_std * 2:
                severity = "high"
            elif score > threshold_std * 1.5:
                severity = "medium"

            results.append(
                AnomalyResult(
                    timestamp=ts,
                    actual_energy_kwh=actual,
                    predicted_energy_kwh=expected,
                    is_anomaly=is_anomaly,
                    anomaly_score=float(score),
                    waste_kwh=waste_kwh,
                    potential_waste_cost_inr=waste_cost,
                    severity=severity,
                    description="Excess energy above baseline threshold" if is_anomaly else "Normal operation",
                )
            )
        return results
