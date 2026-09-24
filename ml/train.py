"""
Model training module for EnScale.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from config.settings import ML_ARTIFACTS_DIR
from ml.preprocessing import prepare_time_features, split_features_target


def train_baseline_model(
    df: pd.DataFrame,
    model_name: str = "energy_forecast_model.joblib",
    model_type: str = "ridge",
) -> Path:
    """
    Trains a baseline regression model and serializes it to ml/artifacts/.
    """
    processed = prepare_time_features(df)
    X, y = split_features_target(processed)

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X, y)

    ML_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ML_ARTIFACTS_DIR / model_name
    joblib.dump({"model": model, "feature_names": list(X.columns)}, artifact_path)

    return artifact_path
