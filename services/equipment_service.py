"""Equipment inventory editing and building-window constraint validation."""
from typing import List, Tuple, Union

from config.constants import CANONICAL_EQUIPMENT_FIELDS
from domain.models import BuildingProfile, Equipment
from domain.schemas import ValidationResult
from services.validation_service import validate_equipment


def available_operating_window(building_profile: Union[BuildingProfile, dict]) -> float:
    if isinstance(building_profile, dict):
        return float(building_profile["operating_end"] - building_profile["operating_start"])
    return float(building_profile.operating_end - building_profile.operating_start)


def validate_equipment_for_building(equipment: Union[Equipment, dict], building_profile: Union[BuildingProfile, dict]) -> ValidationResult:
    result = validate_equipment(equipment)
    data = equipment.to_dict() if isinstance(equipment, Equipment) else equipment
    window = available_operating_window(building_profile)
    if window <= 0:
        result.add_error("Building operating window must be greater than zero hours")
        return result
    try:
        for field_name in ("hours_per_day", "minimum_hours", "maximum_hours"):
            value = float(data[field_name])
            if value > window:
                result.add_error(f"{field_name} ({value:g}h) exceeds available building operating window ({window:g}h)")
    except (KeyError, TypeError, ValueError):
        pass  # Base field validation reports missing or malformed values.
    return result


def create_equipment(equipment_id: str, operating_window_hours: float) -> Equipment:
    """Create a canonical editable item whose initial schedule fits the current window."""
    window = float(operating_window_hours)
    if not 0 < window <= 24:
        raise ValueError("A valid positive operating window of at most 24 hours is required")
    return Equipment(equipment_id=equipment_id, equipment_type="Other", equipment_name="New equipment",
                     rated_power_kw=1.0, quantity=1, hours_per_day=window,
                     operating_days=22, utilization_factor=1.0,
                     minimum_hours=min(1.0, window), maximum_hours=window,
                     is_flexible=False)


def add_equipment(equipment_list: List[Equipment], equipment: Equipment) -> None:
    equipment_list.append(equipment)


def remove_equipment(equipment_list: List[Equipment], equipment_id: str) -> bool:
    for index, equipment in enumerate(equipment_list):
        if equipment.equipment_id == equipment_id:
            del equipment_list[index]
            return True
    return False


def equipment_canonical_fields() -> Tuple[str, ...]:
    """Return the public canonical equipment contract, excluding legacy runtime flags."""
    return CANONICAL_EQUIPMENT_FIELDS
