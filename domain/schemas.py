"""
Schema definitions, type specifications, and validation result containers for EnScale.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Final, Tuple

from config.constants import (
    REQUIRED_ENERGY_CSV_COLUMNS,
    OPTIONAL_ENERGY_CSV_COLUMNS,
    CANONICAL_BUILDING_FIELDS,
    CANONICAL_EQUIPMENT_FIELDS,
    CANONICAL_FORECAST_FIELDS,
    CANONICAL_OPTIMIZATION_FIELDS,
    CANONICAL_EPI_FIELDS,
)


@dataclass
class ValidationResult:
    """Standard container for data validation outcomes."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


# Schema Column Specifications
ENERGY_DF_REQUIRED_COLUMNS: Final[Tuple[str, ...]] = REQUIRED_ENERGY_CSV_COLUMNS
ENERGY_DF_OPTIONAL_COLUMNS: Final[Tuple[str, ...]] = OPTIONAL_ENERGY_CSV_COLUMNS

# Type mappings for validation
BUILDING_FIELD_TYPES: Final[Dict[str, type]] = {
    "building_id": str,
    "building_type": str,
    "location_state": str,
    "climate_zone": str,
    "net_built_up_area_m2": (int, float),
    "operating_start": int,
    "operating_end": int,
    "operating_days_per_month": int,
    "occupancy": (int, float),
    "monthly_budget_inr": (int, float),
    "tariff_inr_per_kwh": (int, float),
}

EQUIPMENT_FIELD_TYPES: Final[Dict[str, type]] = {
    "equipment_id": str,
    "equipment_type": str,
    "equipment_name": str,
    "rated_power_kw": (int, float),
    "quantity": int,
    "hours_per_day": (int, float),
    "operating_days": int,
    "utilization_factor": (int, float),
    "minimum_hours": (int, float),
    "maximum_hours": (int, float),
    "is_flexible": bool,
}

INCENTIVE_RECORD_REQUIRED_FIELDS: Final[Tuple[str, ...]] = (
    "incentive_id",
    "program_name",
    "target_equipment_type",
    "eligibility_status",
    "estimated_rebate_inr",
)
