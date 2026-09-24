"""
Model training and evaluation orchestration module for EnScale.

Supports:
- HistGradientBoostingRegressor (preferred) and RandomForestRegressor
- Chronological 70% / 15% / 15% time split
- Comparison against SameHourHistoricalBaseline
- Verification of acceptance rule (ML validation MAE < Baseline validation MAE)
- Serialization of model.joblib, metadata.json, and evaluation.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from config.settings import ML_ARTIFACTS_DIR
from ml.preprocessing import prepare_features, chronological_time_split
from ml.evaluation import (
    calculate_forecast_metrics,
    SameHourHistoricalBaseline,
    compare_models,
)

FEATURE_CANDIDATES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "air_temperature",
    "dew_temperature",
    "building_area",
    "occupancy",
    "floor_count",
]


def train_and_evaluate(
    df: pd.DataFrame,
    dataset_name: str = "Demo Commercial Office Energy Profile",
    dataset_manifest: Optional[Dict[str, Any]] = None,
    target_col: str = "energy_kwh",
    model_type: str = "HistGradientBoostingRegressor",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes full ML training pipeline:
    1. Preprocesses features (without future leakage)
    2. Chronologically splits into 70% train, 15% validation, 15% test
    3. Fits SameHourHistoricalBaseline benchmark
    4. Fits ML Regressor on training set
    5. Validates on validation set (Acceptance Rule: ML MAE < Baseline MAE)
    6. Evaluates final model on untouched test set
    7. Saves artifacts: model.joblib, metadata.json, evaluation.json
    """
    processed = prepare_features(df)
    train_df, val_df, test_df = chronological_time_split(processed, 0.70, 0.15, 0.15)

    # Determine available feature columns
    feature_cols = [c for c in FEATURE_CANDIDATES if c in processed.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 1. Non-ML Baseline Model
    baseline = SameHourHistoricalBaseline()
    baseline.fit(train_df, target_col=target_col)
    base_val_preds = baseline.predict(val_df)
    base_val_metrics = calculate_forecast_metrics(y_val, base_val_preds)

    # 2. Scikit-learn ML Model
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
        )

    model.fit(X_train, y_train)

    ml_val_preds = model.predict(X_val)
    ml_val_metrics = calculate_forecast_metrics(y_val, ml_val_preds)

    val_comparison = compare_models(base_val_metrics, ml_val_metrics)

    # Acceptance rule check: ML must outperform baseline on validation MAE
    if not val_comparison["ml_outperformed_baseline"]:
        # Fallback tuning if needed
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        ml_val_preds = model.predict(X_val)
        ml_val_metrics = calculate_forecast_metrics(y_val, ml_val_preds)
        val_comparison = compare_models(base_val_metrics, ml_val_metrics)

    # 3. Final Test Evaluation (touched only once)
    base_test_preds = baseline.predict(test_df)
    base_test_metrics = calculate_forecast_metrics(y_test, base_test_preds)

    ml_test_preds = model.predict(X_test)
    ml_test_metrics = calculate_forecast_metrics(y_test, ml_test_preds)
    test_comparison = compare_models(base_test_metrics, ml_test_metrics)

    # Save artifacts
    target_dir = Path(output_dir) if output_dir is not None else ML_ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "model.joblib"
    joblib.dump({
        "model": model,
        "feature_names": feature_cols,
        "model_type": model_type,
    }, model_path)

    # Also keep energy_forecast_model.joblib for Stage 1 compatibility
    joblib.dump({
        "model": model,
        "feature_names": feature_cols,
        "model_type": model_type,
    }, target_dir / "energy_forecast_model.joblib")

    metadata = {
        "model_type": model_type,
        "feature_columns": feature_cols,
        "dataset_name": dataset_name,
        "dataset_manifest": dataset_manifest or {},
        "training_date": datetime.now().isoformat(),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "metrics": {
            "validation": val_comparison,
            "test": test_comparison,
        },
    }

    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    eval_summary = {
        "dataset_name": dataset_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "validation_comparison": val_comparison,
        "test_comparison": test_comparison,
        "acceptance_criteria_met": bool(val_comparison["ml_outperformed_baseline"]),
    }

    evaluation_path = target_dir / "evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    return metadata


# Backward compatibility wrapper for Stage 1
def train_baseline_model(
    df: pd.DataFrame,
    model_name: str = "energy_forecast_model.joblib",
    model_type: str = "ridge",
) -> Path:
    res = train_and_evaluate(df, dataset_name="EnScale Baseline")
    return ML_ARTIFACTS_DIR / model_name
