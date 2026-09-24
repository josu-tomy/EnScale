"""
Data ingestion service for EnScale.

Supports:
- Historical CSV ingestion (load_energy_csv) with comprehensive structural validation
- Equipment inventory parsing (MODE 2)
- Demo scenario generation and loading (MODE 3)
- Canonical CSV template provisioning
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import io
import pandas as pd
import numpy as np

from config.constants import (
    REQUIRED_ENERGY_CSV_COLUMNS,
    OPTIONAL_ENERGY_CSV_COLUMNS,
)
from domain.enums import DataMode
from domain.models import BuildingProfile, Equipment
from domain.schemas import ValidationResult
from services.validation_service import (
    validate_energy_dataframe,
    validate_building_profile,
    validate_equipment,
    PROHIBITED_ALIASES,
)


def load_energy_csv(file: Any) -> pd.DataFrame:
    """
    Ingests and validates a user historical energy consumption CSV.

    Performs:
    - Column checking (required: timestamp, energy_kwh; optional: temperature_c, occupancy, equipment_load_kw)
    - Prohibited alias detection (e.g. area, energy, power, cost)
    - Timestamp parsing into datetime objects
    - Chronological sorting
    - Duplicate timestamp detection
    - Missing-value inspection
    - Negative-energy validation
    - Numeric validity verification
    - Sampling-frequency inspection

    Does NOT silently discard bad data.
    Attaches full structured validation details to `df.attrs['validation_info']`.

    Raises:
        ValueError: If required columns are missing, unparseable, or completely invalid.
    """
    if isinstance(file, (str, Path)):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    validation_info: Dict[str, Any] = {
        "is_valid": True,
        "row_count": len(df),
        "columns_found": list(df.columns),
        "required_columns_present": True,
        "missing_columns": [],
        "prohibited_aliases": [],
        "missing_values_per_col": {},
        "duplicate_timestamps_count": 0,
        "is_chronologically_sorted": True,
        "negative_energy_count": 0,
        "non_numeric_errors": [],
        "sampling_frequency": "unknown",
        "date_range_start": None,
        "date_range_end": None,
        "errors": [],
        "warnings": [],
    }

    # 1. Column contract validation
    for alias, canonical in PROHIBITED_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            msg = f"Prohibited alias column '{alias}' found. Use canonical column '{canonical}'."
            validation_info["prohibited_aliases"].append(alias)
            validation_info["errors"].append(msg)
            validation_info["is_valid"] = False

    for col in REQUIRED_ENERGY_CSV_COLUMNS:
        if col not in df.columns:
            validation_info["missing_columns"].append(col)
            validation_info["errors"].append(f"Missing required energy column: '{col}'")
            validation_info["is_valid"] = False
            validation_info["required_columns_present"] = False

    if not validation_info["required_columns_present"]:
        # Attach report and raise
        df.attrs["validation_info"] = validation_info
        raise ValueError(f"Historical CSV validation failed: {'; '.join(validation_info['errors'])}")

    # 2. Missing-values inspection (without discarding)
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        validation_info["missing_values_per_col"][col] = null_count
        if null_count > 0:
            validation_info["warnings"].append(f"Column '{col}' contains {null_count} missing (NaN) values.")

    # 3. Timestamp parsing
    try:
        raw_ts = df["timestamp"].copy()
        parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        unparseable_count = int(parsed_ts.isna().sum() - raw_ts.isna().sum())
        if unparseable_count > 0:
            validation_info["errors"].append(f"{unparseable_count} timestamps could not be parsed as datetime format.")
            validation_info["is_valid"] = False
        df["timestamp"] = parsed_ts
    except Exception as e:
        validation_info["errors"].append(f"Failed to parse timestamp column: {e}")
        validation_info["is_valid"] = False

    # 4. Chronological order & sorting
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        is_sorted = df["timestamp"].is_monotonic_increasing
        validation_info["is_chronologically_sorted"] = bool(is_sorted)
        if not is_sorted:
            validation_info["warnings"].append("Timestamps were not in chronological order. Sorted automatically.")
            df = df.sort_values("timestamp").reset_index(drop=True)

        if not df.empty and df["timestamp"].notna().any():
            validation_info["date_range_start"] = str(df["timestamp"].dropna().iloc[0])
            validation_info["date_range_end"] = str(df["timestamp"].dropna().iloc[-1])

    # 5. Duplicate timestamp detection
    dup_mask = df["timestamp"].duplicated()
    dup_count = int(dup_mask.sum())
    validation_info["duplicate_timestamps_count"] = dup_count
    if dup_count > 0:
        validation_info["warnings"].append(
            f"Detected {dup_count} duplicate timestamp records. Records are preserved without silent loss."
        )

    # 6. Numeric validation and negative energy validation
    if not pd.api.types.is_numeric_dtype(df["energy_kwh"]):
        try:
            converted = pd.to_numeric(df["energy_kwh"], errors="coerce")
            nan_diff = int(converted.isna().sum() - df["energy_kwh"].isna().sum())
            if nan_diff > 0:
                err_msg = f"energy_kwh contains {nan_diff} non-numeric values."
                validation_info["non_numeric_errors"].append(err_msg)
                validation_info["errors"].append(err_msg)
                validation_info["is_valid"] = False
            df["energy_kwh"] = converted
        except Exception:
            err_msg = "energy_kwh is not numeric."
            validation_info["non_numeric_errors"].append(err_msg)
            validation_info["errors"].append(err_msg)
            validation_info["is_valid"] = False

    # Check for negative energy
    if pd.api.types.is_numeric_dtype(df["energy_kwh"]):
        neg_count = int((df["energy_kwh"] < 0).sum())
        validation_info["negative_energy_count"] = neg_count
        if neg_count > 0:
            validation_info["errors"].append(f"Found {neg_count} negative energy consumption records in energy_kwh.")
            validation_info["is_valid"] = False

    # Validate optional numeric columns
    for opt_col in OPTIONAL_ENERGY_CSV_COLUMNS:
        if opt_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[opt_col]):
                try:
                    df[opt_col] = pd.to_numeric(df[opt_col], errors="coerce")
                except Exception:
                    validation_info["warnings"].append(f"Optional column '{opt_col}' has non-numeric values.")

    # 7. Sampling frequency inspection
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]) and len(df) > 1:
        time_diffs = df["timestamp"].diff().dropna()
        if len(time_diffs) > 0:
            median_seconds = time_diffs.median().total_seconds()
            if median_seconds == 900:
                validation_info["sampling_frequency"] = "15-minute"
            elif median_seconds == 1800:
                validation_info["sampling_frequency"] = "30-minute"
            elif median_seconds == 3600:
                validation_info["sampling_frequency"] = "Hourly"
            elif median_seconds == 86400:
                validation_info["sampling_frequency"] = "Daily"
            else:
                validation_info["sampling_frequency"] = f"{median_seconds:.0f}s (irregular/custom)"

    # Attach validation report
    df.attrs["validation_info"] = validation_info

    # If fatal errors occurred (e.g. prohibited aliases or unparseable columns)
    if not validation_info["is_valid"] and len(validation_info["errors"]) > 0:
        raise ValueError(f"Historical CSV validation failed: {'; '.join(validation_info['errors'])}")

    return df


def get_energy_csv_template() -> str:
    """Returns downloadable canonical CSV template header and example rows."""
    template_path = Path(__file__).resolve().parent.parent / "data" / "reference" / "energy_template.csv"
    if template_path.exists():
        return template_path.read_text()
    return (
        "timestamp,energy_kwh,temperature_c,occupancy,equipment_load_kw\n"
        "2026-09-01 00:00:00,45.2,26.5,5,42.0\n"
        "2026-09-01 01:00:00,41.8,25.8,2,39.5\n"
    )


def load_demo_energy_data() -> pd.DataFrame:
    """Loads the pre-generated simulated office energy demo dataset."""
    demo_file = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_office_energy.csv"
    if not demo_file.exists():
        raise FileNotFoundError(f"Demo dataset not found at {demo_file}")
    return load_energy_csv(demo_file)


class IngestionService:
    """Service interface for ingesting building and energy data across all three modes."""

    def __init__(self, data_mode: DataMode = DataMode.MODE_3_DEMO):
        self.data_mode = data_mode

    def ingest_historical_csv(self, filepath_or_buffer: Any) -> pd.DataFrame:
        """
        MODE 1: Ingests user historical energy consumption CSV using load_energy_csv.
        """
        return load_energy_csv(filepath_or_buffer)

    def ingest_equipment_inventory(self, items: List[Dict[str, Any]]) -> List[Equipment]:
        """
        MODE 2: Ingests equipment inventory for equipment-based baseline calculations.
        """
        equipment_list: List[Equipment] = []
        for item in items:
            eq = Equipment.from_dict(item) if isinstance(item, dict) else item
            val_res = validate_equipment(eq)
            if not val_res.is_valid:
                raise ValueError(f"Equipment validation failed for {eq.equipment_id}: {'; '.join(val_res.errors)}")
            equipment_list.append(eq)
        return equipment_list

    def load_demo_scenario(self) -> Dict[str, Any]:
        """
        MODE 3: Loads reference demo scenario for testing and evaluation.
        """
        demo_building = BuildingProfile(
            building_id="BLD-DEMO-001",
            building_type="Commercial Office",
            location_state="Maharashtra",
            climate_zone="Composite",
            net_built_up_area_m2=2500.0,
            operating_start=9,
            operating_end=18,
            operating_days_per_month=22,
            occupancy=150.0,
            monthly_budget_inr=150000.0,
            tariff_inr_per_kwh=8.50,
        )
        return {
            "mode": DataMode.MODE_3_DEMO,
            "building_profile": demo_building,
            "description": "Standard 2500 m2 commercial office demo scenario in Mumbai (Composite climate).",
        }
