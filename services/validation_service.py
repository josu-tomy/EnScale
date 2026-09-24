"""
Validation service for EnScale.

Enforces canonical variable contracts and schema validations for:
- BuildingProfile
- Equipment
- Energy DataFrame
- Incentive records
"""

from typing import Union, Dict, Any, List
import pandas as pd

from config.constants import (
    CANONICAL_BUILDING_FIELDS,
    CANONICAL_EQUIPMENT_FIELDS,
    REQUIRED_ENERGY_CSV_COLUMNS,
    OPTIONAL_ENERGY_CSV_COLUMNS,
    VALIDATION_LIMITS,
)
from domain.models import BuildingProfile, Equipment, IncentiveMatch
from domain.schemas import ValidationResult, INCENTIVE_RECORD_REQUIRED_FIELDS

# Prohibited aliases to proactively catch contract violations
PROHIBITED_ALIASES: Dict[str, str] = {
    "area": "net_built_up_area_m2",
    "sqft": "net_built_up_area_m2",
    "floor_area": "net_built_up_area_m2",
    "energy": "energy_kwh",
    "power": "rated_power_kw / equipment_load_kw",
    "cost": "baseline_cost_inr / monthly_budget_inr",
}


def validate_building_profile(profile: Union[BuildingProfile, Dict[str, Any]]) -> ValidationResult:
    """
    Validates a building profile against the canonical building contract.
    """
    res = ValidationResult()
    data = profile.to_dict() if isinstance(profile, BuildingProfile) else profile

    # Check for alias violations
    for alias, canonical in PROHIBITED_ALIASES.items():
        if alias in data and canonical not in data:
            res.add_error(f"Prohibited alias '{alias}' found. Use canonical field '{canonical}'.")

    # Check required canonical fields
    for field_name in CANONICAL_BUILDING_FIELDS:
        if field_name not in data or data[field_name] is None:
            res.add_error(f"Missing required building field: '{field_name}'")

    if not res.is_valid:
        return res

    # Numeric and range checks
    try:
        area = float(data["net_built_up_area_m2"])
        if area <= 0:
            res.add_error("net_built_up_area_m2 must be greater than 0")
        elif area > VALIDATION_LIMITS["max_net_built_up_area_m2"]:
            res.add_warning(f"net_built_up_area_m2 ({area}) is unusually large")
    except (ValueError, TypeError):
        res.add_error("net_built_up_area_m2 must be numeric")

    try:
        start = int(data["operating_start"])
        end = int(data["operating_end"])
        if not (VALIDATION_LIMITS["min_operating_start"] <= start <= VALIDATION_LIMITS["max_operating_start"]):
            res.add_error("operating_start must be between 0 and 23")
        if not (VALIDATION_LIMITS["min_operating_end"] <= end <= VALIDATION_LIMITS["max_operating_end"]):
            res.add_error("operating_end must be between 0 and 24")
        if start >= end:
            res.add_error("operating_start must be strictly earlier than operating_end")
    except (ValueError, TypeError):
        res.add_error("operating_start and operating_end must be integer hours")

    try:
        days = int(data["operating_days_per_month"])
        if not (VALIDATION_LIMITS["min_operating_days_per_month"] <= days <= VALIDATION_LIMITS["max_operating_days_per_month"]):
            res.add_error("operating_days_per_month must be between 1 and 31")
    except (ValueError, TypeError):
        res.add_error("operating_days_per_month must be an integer")

    try:
        occupancy = float(data["occupancy"])
        if occupancy < 0:
            res.add_error("occupancy cannot be negative")
    except (ValueError, TypeError):
        res.add_error("occupancy must be numeric")

    try:
        budget = float(data["monthly_budget_inr"])
        if budget < 0:
            res.add_error("monthly_budget_inr cannot be negative")
    except (ValueError, TypeError):
        res.add_error("monthly_budget_inr must be numeric")

    try:
        tariff = float(data["tariff_inr_per_kwh"])
        if tariff <= 0:
            res.add_error("tariff_inr_per_kwh must be greater than 0")
        elif not (VALIDATION_LIMITS["min_tariff_inr_per_kwh"] <= tariff <= VALIDATION_LIMITS["max_tariff_inr_per_kwh"]):
            res.add_warning(f"tariff_inr_per_kwh ({tariff}) is outside typical range (1-50 INR/kWh)")
    except (ValueError, TypeError):
        res.add_error("tariff_inr_per_kwh must be numeric")

    return res


def validate_equipment(equipment: Union[Equipment, Dict[str, Any]]) -> ValidationResult:
    """
    Validates an equipment definition against the canonical equipment contract.
    """
    res = ValidationResult()
    data = equipment.to_dict() if isinstance(equipment, Equipment) else equipment

    # Check for alias violations
    for alias, canonical in PROHIBITED_ALIASES.items():
        if alias in data and canonical not in data:
            res.add_error(f"Prohibited alias '{alias}' found. Use canonical field '{canonical}'.")

    # Check required canonical fields
    for field_name in CANONICAL_EQUIPMENT_FIELDS:
        if field_name not in data or data[field_name] is None:
            res.add_error(f"Missing required equipment field: '{field_name}'")

    if not res.is_valid:
        return res

    # Numeric and range checks
    try:
        power = float(data["rated_power_kw"])
        if power <= 0:
            res.add_error("rated_power_kw must be greater than 0")
    except (ValueError, TypeError):
        res.add_error("rated_power_kw must be numeric")

    try:
        qty = int(data["quantity"])
        if qty < 1:
            res.add_error("quantity must be at least 1")
    except (ValueError, TypeError):
        res.add_error("quantity must be an integer")

    try:
        hpd = float(data["hours_per_day"])
        if not (VALIDATION_LIMITS["min_hours_per_day"] <= hpd <= VALIDATION_LIMITS["max_hours_per_day"]):
            res.add_error("hours_per_day must be between 0 and 24")
    except (ValueError, TypeError):
        res.add_error("hours_per_day must be numeric")

    try:
        op_days = int(data["operating_days"])
        if not (VALIDATION_LIMITS["min_operating_days"] <= op_days <= VALIDATION_LIMITS["max_operating_days"]):
            res.add_error("operating_days must be between 1 and 31")
    except (ValueError, TypeError):
        res.add_error("operating_days must be an integer")

    try:
        util = float(data["utilization_factor"])
        if not (VALIDATION_LIMITS["min_utilization_factor"] <= util <= VALIDATION_LIMITS["max_utilization_factor"]):
            res.add_error("utilization_factor must be between 0.0 and 1.0")
    except (ValueError, TypeError):
        res.add_error("utilization_factor must be numeric")

    try:
        min_h = float(data["minimum_hours"])
        max_h = float(data["maximum_hours"])
        if min_h < 0 or max_h > 24:
            res.add_error("minimum_hours and maximum_hours must be within [0, 24]")
        if min_h > max_h:
            res.add_error("minimum_hours cannot exceed maximum_hours")
    except (ValueError, TypeError):
        res.add_error("minimum_hours and maximum_hours must be numeric")

    return res


def validate_energy_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Validates a historical energy consumption DataFrame.
    Required columns: timestamp, energy_kwh.
    Optional: temperature_c, occupancy, equipment_load_kw.
    """
    res = ValidationResult()

    if df is None:
        res.add_error("Energy DataFrame cannot be None")
        return res

    if not isinstance(df, pd.DataFrame):
        res.add_error("Input must be a pandas DataFrame")
        return res

    if df.empty:
        res.add_error("Energy DataFrame is empty")
        return res

    # Check for alias violations in columns
    for alias, canonical in PROHIBITED_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            res.add_error(f"Prohibited alias column '{alias}' found. Use canonical column '{canonical}'.")

    # Check required columns
    for col in REQUIRED_ENERGY_CSV_COLUMNS:
        if col not in df.columns:
            res.add_error(f"Missing required energy column: '{col}'")

    if not res.is_valid:
        return res

    # Validate energy_kwh numeric and non-negative
    if not pd.api.types.is_numeric_dtype(df["energy_kwh"]):
        try:
            converted = pd.to_numeric(df["energy_kwh"], errors="coerce")
            if converted.isna().any():
                res.add_error("energy_kwh contains non-numeric or unparseable values")
        except Exception:
            res.add_error("energy_kwh must be numeric")
    else:
        if (df["energy_kwh"] < 0).any():
            res.add_error("energy_kwh cannot contain negative values")

    # Validate timestamp parseability
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            pd.to_datetime(df["timestamp"])
        except Exception as e:
            res.add_error(f"timestamp column cannot be parsed as datetime: {e}")

    # Validate optional columns if present
    if "temperature_c" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["temperature_c"]):
            res.add_error("temperature_c must be numeric")
        else:
            if ((df["temperature_c"] < VALIDATION_LIMITS["min_temperature_c"]) |
                (df["temperature_c"] > VALIDATION_LIMITS["max_temperature_c"])).any():
                res.add_warning("temperature_c has values outside typical ambient range (-20 to 60 °C)")

    if "occupancy" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["occupancy"]):
            res.add_error("occupancy must be numeric")
        elif (df["occupancy"] < 0).any():
            res.add_error("occupancy cannot contain negative values")

    if "equipment_load_kw" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["equipment_load_kw"]):
            res.add_error("equipment_load_kw must be numeric")
        elif (df["equipment_load_kw"] < 0).any():
            res.add_error("equipment_load_kw cannot contain negative values")

    return res


def validate_incentive_record(record: Union[IncentiveMatch, Dict[str, Any]]) -> ValidationResult:
    """
    Validates an incentive record against schema requirements.
    """
    res = ValidationResult()
    data = record.to_dict() if isinstance(record, IncentiveMatch) else record

    for field_name in INCENTIVE_RECORD_REQUIRED_FIELDS:
        if field_name not in data or data[field_name] is None:
            res.add_error(f"Missing required incentive field: '{field_name}'")

    if not res.is_valid:
        return res

    try:
        rebate = float(data["estimated_rebate_inr"])
        if rebate < 0:
            res.add_error("estimated_rebate_inr cannot be negative")
    except (ValueError, TypeError):
        res.add_error("estimated_rebate_inr must be numeric")

    return res
