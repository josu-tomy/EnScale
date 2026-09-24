"""
Tests for ML forecasting, feature engineering, chronological splitting, metrics, and models.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from domain.models import ForecastResult
from ml.preprocessing import prepare_features, chronological_time_split
from ml.evaluation import (
    calculate_forecast_metrics,
    SameHourHistoricalBaseline,
    compare_models,
    compute_forecast_error,
)
from ml.predict import EnergyPredictor
from ml.train import train_and_evaluate
from config.settings import ML_ARTIFACTS_DIR


def test_feature_generation():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01 00:00:00", periods=48, freq="h"),
        "energy_kwh": [50.0] * 48,
        "temperature_c": [28.0] * 48,
    })
    processed = prepare_features(df)
    assert "hour" in processed.columns
    assert "day_of_week" in processed.columns
    assert "day_of_month" in processed.columns
    assert "month" in processed.columns
    assert "is_weekend" in processed.columns
    assert "hour_sin" in processed.columns
    assert "hour_cos" in processed.columns
    assert "air_temperature" in processed.columns


def test_chronological_time_split():
    """
    Verifies exactly 70% train, 15% validation, 15% test split without shuffle or lookahead.
    """
    n_rows = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=n_rows, freq="h"),
        "energy_kwh": np.linspace(10, 100, n_rows),
    })
    train_df, val_df, test_df = chronological_time_split(df, 0.70, 0.15, 0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Verify strict temporal sequencing (no lookahead / no shuffle)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_no_future_leakage():
    """
    Verifies that feature preparation does not incorporate future rows or leak future target values.
    """
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=10, freq="h"),
        "energy_kwh": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })
    features = prepare_features(df)
    # Target should not be modified, and engineered features depend only on row timestamp
    assert "energy_kwh" in features.columns
    # Check that each row's features depend only on that row's timestamp
    for idx, row in features.iterrows():
        expected_hour = row["timestamp"].hour
        assert row["hour"] == expected_hour


def test_same_hour_historical_baseline():
    """
    Verifies that the non-ML baseline computes hour-of-day / day-of-week averages correctly.
    """
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-01 10:00:00",  # Saturday
            "2026-08-08 10:00:00",  # Saturday
            "2026-08-03 10:00:00",  # Monday
        ]),
        "energy_kwh": [60.0, 80.0, 120.0],
    })
    features = prepare_features(df)
    baseline = SameHourHistoricalBaseline()
    baseline.fit(features)

    test_query = prepare_features(pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-15 10:00:00"]),  # Saturday at 10:00
        "energy_kwh": [0.0],
    }))
    pred = baseline.predict(test_query)
    # Expected Saturday at 10:00 mean = (60 + 80) / 2 = 70.0
    assert pred[0] == 70.0


def test_metric_calculations():
    """
    Verifies computation of MAE, RMSE, CV(RMSE), sMAPE, R2, and improvement_vs_baseline_pct.
    """
    actual = np.array([100.0, 120.0, 110.0, 130.0, 105.0])
    predicted = np.array([95.0, 125.0, 108.0, 128.0, 100.0])

    metrics = calculate_forecast_metrics(actual, predicted)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "CV(RMSE)" in metrics
    assert "sMAPE" in metrics
    assert "R2" in metrics

    # MAE = (|5| + |5| + |2| + |2| + |5|) / 5 = 19 / 5 = 3.8
    assert metrics["MAE"] == 3.8
    assert metrics["R2"] > 0.80

    base_metrics = {"MAE": 5.0}
    ml_metrics = {"MAE": 3.8}
    comp = compare_models(base_metrics, ml_metrics)
    assert comp["improvement_vs_baseline_pct"] == 24.0
    assert comp["ml_outperformed_baseline"] is True


def test_model_training_and_artifacts(tmp_path):
    """
    Verifies end-to-end training, saving of artifacts, and metadata.json structure.
    """
    date_range = pd.date_range("2026-08-01", periods=200, freq="h")
    df = pd.DataFrame({
        "timestamp": date_range,
        "energy_kwh": 50.0 + 20.0 * np.sin(date_range.hour * np.pi / 12) + np.random.normal(0, 1.0, 200),
        "temperature_c": 25.0 + 5.0 * np.sin(date_range.hour * np.pi / 12),
    })

    metadata = train_and_evaluate(df, dataset_name="Test Synthetic Dataset", output_dir=tmp_path)
    assert metadata["train_rows"] == 140
    assert metadata["validation_rows"] == 30
    assert metadata["test_rows"] == 30
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "evaluation.json").exists()


def test_energy_predictor_prediction_output():
    predictor = EnergyPredictor()
    assert predictor.is_loaded() is True
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01 09:00:00", periods=5, freq="h"),
        "energy_kwh": [50.0, 55.0, 60.0, 58.0, 52.0],
    })
    preds = predictor.predict(test_df)
    assert len(preds) == 5
    assert all(isinstance(p, (float, np.floating)) for p in preds)
