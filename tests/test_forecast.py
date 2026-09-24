"""
Tests for ML forecasting, preprocessing, and forecast error canonical fields.
"""

import pandas as pd
import numpy as np

from domain.models import ForecastResult
from ml.preprocessing import prepare_time_features
from ml.evaluation import compute_forecast_error, calculate_forecast_metrics
from ml.predict import EnergyPredictor
from services.forecast_service import ForecastService


def test_prepare_time_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=24, freq="h"),
        "energy_kwh": [50.0] * 24,
    })
    processed = prepare_time_features(df)
    assert "hour" in processed.columns
    assert "dayofweek" in processed.columns
    assert "is_weekend" in processed.columns
    assert "hour_sin" in processed.columns
    assert "hour_cos" in processed.columns


def test_canonical_forecast_error():
    """
    Verifies canonical forecast error calculation:
    forecast_error_kwh = actual_energy_kwh - predicted_energy_kwh
    forecast_error_pct = (abs(forecast_error_kwh) / actual_energy_kwh) * 100
    """
    actual = 100.0
    predicted = 90.0
    res = compute_forecast_error(actual, predicted)
    assert res["forecast_error_kwh"] == 10.0
    assert res["forecast_error_pct"] == 10.0


def test_forecast_service_predict():
    service = ForecastService()
    df = pd.DataFrame({
        "timestamp": ["2026-08-01 10:00:00"],
        "energy_kwh": [65.0],
        "actual_energy_kwh": [70.0],
    })
    results = service.predict(df)
    assert len(results) == 1
    item = results[0]
    assert isinstance(item, ForecastResult)
    assert item.predicted_energy_kwh == 65.0
    assert item.actual_energy_kwh == 70.0
    assert item.forecast_error_kwh == 5.0


def test_energy_predictor_inference():
    predictor = EnergyPredictor()
    assert predictor.is_loaded() is True
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01 09:00:00", periods=5, freq="h"),
        "energy_kwh": [50.0, 55.0, 60.0, 58.0, 52.0],
    })
    preds = predictor.predict(test_df)
    assert len(preds) == 5
    assert all(isinstance(p, (float, np.floating)) for p in preds)
