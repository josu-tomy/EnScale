"""
Global constants and canonical definitions for EnScale.

All field names and standard values defined here are global contracts.
Do not introduce aliases or alternative spellings.
"""

from typing import Final, List, Tuple

# Application Identity
APP_NAME: Final[str] = "EnScale"
APP_TAGLINE: Final[str] = (
    "EnScale is an ML-assisted energy decision platform for commercial and small industrial buildings."
)
APP_VERSION: Final[str] = "0.1.0"

# Canonical Building Fields (Global Contract)
CANONICAL_BUILDING_FIELDS: Final[Tuple[str, ...]] = (
    "building_id",
    "building_type",
    "location_state",
    "climate_zone",
    "net_built_up_area_m2",
    "operating_start",
    "operating_end",
    "operating_days_per_month",
    "occupancy",
    "monthly_budget_inr",
    "tariff_inr_per_kwh",
)

# Canonical Equipment Fields (Global Contract)
CANONICAL_EQUIPMENT_FIELDS: Final[Tuple[str, ...]] = (
    "equipment_id",
    "equipment_type",
    "equipment_name",
    "rated_power_kw",
    "quantity",
    "hours_per_day",
    "operating_days",
    "utilization_factor",
    "minimum_hours",
    "maximum_hours",
    "is_flexible",
)

# Canonical Energy Fields (Global Contract)
CANONICAL_ENERGY_FIELDS: Final[Tuple[str, ...]] = (
    "timestamp",
    "energy_kwh",
    "temperature_c",
    "occupancy",
    "equipment_load_kw",
)

REQUIRED_ENERGY_CSV_COLUMNS: Final[Tuple[str, ...]] = (
    "timestamp",
    "energy_kwh",
)

OPTIONAL_ENERGY_CSV_COLUMNS: Final[Tuple[str, ...]] = (
    "temperature_c",
    "occupancy",
    "equipment_load_kw",
)

# Canonical Forecast Fields (Global Contract)
CANONICAL_FORECAST_FIELDS: Final[Tuple[str, ...]] = (
    "predicted_energy_kwh",
    "actual_energy_kwh",
    "forecast_error_kwh",
    "forecast_error_pct",
)

# Canonical Optimization Fields (Global Contract)
CANONICAL_OPTIMIZATION_FIELDS: Final[Tuple[str, ...]] = (
    "baseline_energy_kwh",
    "optimized_energy_kwh",
    "energy_savings_kwh",
    "baseline_cost_inr",
    "optimized_cost_inr",
    "cost_savings_inr",
    "savings_pct",
)

# Canonical EPI Fields (Global Contract)
CANONICAL_EPI_FIELDS: Final[Tuple[str, ...]] = (
    "annual_energy_kwh",
    "epi_kwh_m2_year",
)

# Tariffs & Emissions Defaults (India commercial / industrial reference)
DEFAULT_TARIFF_INR_PER_KWH: Final[float] = 8.50
DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH: Final[float] = 0.82  # kg CO2 / kWh (CEA standard baseline)

# Supported Building Types
SUPPORTED_BUILDING_TYPES: Final[Tuple[str, ...]] = (
    "Commercial Office",
    "Retail / Mall",
    "Warehouse / Logistics",
    "Light Industrial / Workshop",
    "Healthcare / Hospital",
    "Hospitality / Hotel",
    "Educational Institution",
)

# Supported Equipment Types
SUPPORTED_EQUIPMENT_TYPES: Final[Tuple[str, ...]] = (
    "HVAC",
    "Lighting",
    "Motors & Pumps",
    "Air Compressors",
    "Refrigeration",
    "IT Equipment",
    "Process Machinery",
    "Other",
)

# Supported Climate Zones (ECBC India Classification)
SUPPORTED_CLIMATE_ZONES: Final[Tuple[str, ...]] = (
    "Composite",
    "Hot and Dry",
    "Warm and Humid",
    "Moderate",
    "Cold",
)

# Data Validation Limits
VALIDATION_LIMITS: Final[dict] = {
    "min_net_built_up_area_m2": 10.0,
    "max_net_built_up_area_m2": 1_000_000.0,
    "min_tariff_inr_per_kwh": 1.0,
    "max_tariff_inr_per_kwh": 50.0,
    "min_operating_start": 0,
    "max_operating_start": 23,
    "min_operating_end": 0,
    "max_operating_end": 24,
    "min_operating_days_per_month": 1,
    "max_operating_days_per_month": 31,
    "min_occupancy": 0.0,
    "max_occupancy": 100_000.0,
    "min_rated_power_kw": 0.01,
    "max_rated_power_kw": 50_000.0,
    "min_quantity": 1,
    "max_quantity": 10_000,
    "min_hours_per_day": 0.0,
    "max_hours_per_day": 24.0,
    "min_operating_days": 1,
    "max_operating_days": 31,
    "min_utilization_factor": 0.0,
    "max_utilization_factor": 1.0,
    "min_temperature_c": -20.0,
    "max_temperature_c": 60.0,
}
