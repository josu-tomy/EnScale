"""
Tests for anomaly detection and operational waste quantification.
"""

import pandas as pd
import numpy as np
from domain.models import AnomalyResult
from services.anomaly_service import (
    AnomalyService,
    detect_anomalies,
    calculate_forecast_errors,
)


def test_calculate_forecast_errors():
    actual = [100.0, 50.0, 0.0]
    predicted = [80.0, 60.0, 10.0]
    err_kwh, err_pct = calculate_forecast_errors(actual, predicted)

    assert err_kwh[0] == 20.0
    assert err_pct[0] == 20.0
    assert err_kwh[1] == -10.0
    assert err_pct[1] == 20.0
    assert err_pct[2] == 0.0  # safe division on actual=0


def test_detect_anomalies_output_schema():
    """
    Verifies that anomaly detection produces all required canonical and output fields:
    timestamp, actual_energy_kwh, predicted_energy_kwh, forecast_error_kwh,
    forecast_error_pct, anomaly_flag, severity.
    """
    timestamps = pd.date_range("2026-08-01", periods=30, freq="h")
    predicted = [50.0] * 30
    actual = [50.0] * 30
    # Inject a distinct spike at index 25
    actual[25] = 120.0

    df = detect_anomalies(
        actual_energy_kwh=actual,
        predicted_energy_kwh=predicted,
        timestamp=timestamps,
        window_size=24,
        threshold_sigma=2.5,
    )

    required_cols = [
        "timestamp",
        "actual_energy_kwh",
        "predicted_energy_kwh",
        "forecast_error_kwh",
        "forecast_error_pct",
        "anomaly_flag",
        "severity",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert df.loc[25, "anomaly_flag"] == True
    assert df.loc[25, "severity"] in ["medium", "high"]
    assert df.loc[10, "anomaly_flag"] == False
    assert df.loc[10, "severity"] == "normal"


def test_anomaly_result_model():
    res = AnomalyResult(
        timestamp="2026-08-01 14:00:00",
        actual_energy_kwh=120.0,
        predicted_energy_kwh=80.0,
        is_anomaly=True,
        anomaly_score=3.5,
        waste_kwh=40.0,
        potential_waste_cost_inr=340.0,
        severity="high",
        description="Significant excess energy load",
    )
    d = res.to_dict()
    assert d["actual_energy_kwh"] == 120.0
    assert d["waste_kwh"] == 40.0
    assert d["is_anomaly"] is True
    assert d["anomaly_flag"] is True


def test_anomaly_service_detect_waste_backward_compatibility():
    service = AnomalyService(default_tariff_inr_per_kwh=10.0)
    actual = pd.Series([50.0, 52.0, 49.0, 51.0, 150.0])
    expected = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0])

    anomalies = service.detect_waste(actual, expected, threshold_std=1.5)
    assert len(anomalies) == 5

    spike_res = anomalies[4]
    assert spike_res.is_anomaly is True
    assert spike_res.anomaly_flag is True
    assert spike_res.waste_kwh == 100.0
    assert spike_res.potential_waste_cost_inr == 1000.0
