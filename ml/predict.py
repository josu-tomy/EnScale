"""
Inference and prediction module for EnScale.
"""

from pathlib import Path
from typing import Optional, Union, List
import joblib
import pandas as pd
import numpy as np

from config.settings import ML_ARTIFACTS_DIR
from ml.preprocessing import prepare_features


class EnergyPredictor:
    """Loads a trained model artifact and performs inference."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path:
            self.model_path = Path(model_path)
        else:
            preferred = ML_ARTIFACTS_DIR / "model.joblib"
            legacy = ML_ARTIFACTS_DIR / "energy_forecast_model.joblib"
            self.model_path = preferred if preferred.exists() else legacy

        self.model = None
        self.feature_names: List[str] = []
        self.model_type: str = "Unknown"
        self._load()

    def _load(self) -> None:
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.model_type = data.get("model_type", "ScikitLearnRegressor")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Runs inference on input dataframe.
        Gracefully handles missing optional features with neutral defaults.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Model artifact not loaded from {self.model_path}")

        features = prepare_features(df)
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0

        X = features[self.feature_names]
        return self.model.predict(X)
