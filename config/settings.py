"""
Application settings and filesystem path configurations for EnScale.
"""

from pathlib import Path
from typing import Dict, Any
import json

from config.constants import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    DEFAULT_TARIFF_INR_PER_KWH,
    DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH,
)

# Project Root Directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DEMO_DATA_DIR: Path = DATA_DIR / "demo"
REFERENCE_DATA_DIR: Path = DATA_DIR / "reference"
USER_DATA_DIR: Path = DATA_DIR / "user"

# Machine Learning Artifacts Directory
ML_ARTIFACTS_DIR: Path = BASE_DIR / "ml" / "artifacts"


def load_emission_factor_config(region: str = "India_National_Grid") -> Dict[str, Any]:
    """
    Loads grid emission factor and its authoritative metadata from data/reference/emission_factors.json.
    Avoids hardcoding emission factors across multiple files.
    """
    json_path = REFERENCE_DATA_DIR / "emission_factors.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            factors = data.get("factors", {})
            if region in factors:
                return factors[region]
            default_key = data.get("default_region", "India_National_Grid")
            if default_key in factors:
                return factors[default_key]
        except Exception:
            pass

    return {
        "factor_kg_co2_per_kwh": DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH,
        "unit": "kg CO2 / kWh",
        "source": "CEA CO2 Baseline Database (Default Reference)",
        "description": "National default grid emission factor",
    }


class Settings:
    """Central settings for local execution."""

    app_name: str = APP_NAME
    app_tagline: str = APP_TAGLINE
    app_version: str = APP_VERSION

    default_tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH

    @property
    def default_emission_factor_kg_per_kwh(self) -> float:
        meta = load_emission_factor_config()
        return float(meta.get("factor_kg_co2_per_kwh", DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH))

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    demo_data_dir: Path = DEMO_DATA_DIR
    reference_data_dir: Path = REFERENCE_DATA_DIR
    user_data_dir: Path = USER_DATA_DIR
    ml_artifacts_dir: Path = ML_ARTIFACTS_DIR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_tagline": self.app_tagline,
            "app_version": self.app_version,
            "default_tariff_inr_per_kwh": self.default_tariff_inr_per_kwh,
            "default_emission_factor_kg_per_kwh": self.default_emission_factor_kg_per_kwh,
            "base_dir": str(self.base_dir),
            "data_dir": str(self.data_dir),
        }


settings = Settings()
