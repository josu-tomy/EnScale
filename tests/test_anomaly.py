"""
Tests for anomaly detection and operational waste quantification.
"""

import pandas as pd
from domain.models import AnomalyResult
from services.anomaly_service import AnomalyService


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


def test_anomaly_detection_logic():
    service = AnomalyService(default_tariff_inr_per_kwh=10.0)
    actual = pd.Series([50.0, 52.0, 49.0, 51.0, 150.0])
    expected = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0])

    anomalies = service.detect_waste(actual, expected, threshold_std=1.5)
    assert len(anomalies) == 5

    spike_res = anomalies[4]
    assert spike_res.is_anomaly is True
    assert spike_res.waste_kwh == 100.0
    assert spike_res.potential_waste_cost_inr == 1000.0
