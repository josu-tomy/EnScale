"""
Forecast service interface for EnScale.

Adheres strictly to canonical forecast fields:
- predicted_energy_kwh
- actual_energy_kwh
- forecast_error_kwh
- forecast_error_pct
"""

from typing import List, Optional, Dict, Any
import pandas as pd

from domain.models import ForecastResult


class ForecastService:
    """Service skeleton for ML-driven energy consumption forecasting."""

    def __init__(self, model_artifact_path: Optional[str] = None):
        self.model_artifact_path = model_artifact_path
        self.model = None

    def predict(self, feature_df: pd.DataFrame) -> List[ForecastResult]:
        """
        Generates energy forecasts from preprocessed feature DataFrame.
        Returns list of ForecastResult adhering to canonical fields.
        """
        results: List[ForecastResult] = []
        for _, row in feature_df.iterrows():
            ts = row.get("timestamp")
            predicted_kwh = float(row.get("energy_kwh", 0.0))
            actual_kwh = float(row.get("actual_energy_kwh")) if "actual_energy_kwh" in row else None

            err_kwh = (actual_kwh - predicted_kwh) if actual_kwh is not None else None
            err_pct = (abs(err_kwh) / actual_kwh * 100.0) if (actual_kwh is not None and actual_kwh > 0) else None

            results.append(
                ForecastResult(
                    timestamp=ts,
                    predicted_energy_kwh=predicted_kwh,
                    actual_energy_kwh=actual_kwh,
                    forecast_error_kwh=err_kwh,
                    forecast_error_pct=err_pct,
                )
            )
        return results
