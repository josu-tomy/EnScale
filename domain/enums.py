"""
Domain enumerations for EnScale.
"""

from enum import Enum


class DataMode(str, Enum):
    """The three official operating modes for EnScale."""
    MODE_1_HISTORICAL = "historical"
    MODE_2_EQUIPMENT_BASELINE = "equipment_baseline"
    MODE_3_DEMO = "demo"


class BuildingType(str, Enum):
    """Supported commercial and light-industrial building types."""
    COMMERCIAL_OFFICE = "Commercial Office"
    RETAIL_MALL = "Retail / Mall"
    WAREHOUSE_LOGISTICS = "Warehouse / Logistics"
    LIGHT_INDUSTRIAL_WORKSHOP = "Light Industrial / Workshop"
    HEALTHCARE_HOSPITAL = "Healthcare / Hospital"
    HOSPITALITY_HOTEL = "Hospitality / Hotel"
    EDUCATIONAL_INSTITUTION = "Educational Institution"


class ClimateZone(str, Enum):
    """Indian ECBC Climate Zones."""
    COMPOSITE = "Composite"
    HOT_DRY = "Hot and Dry"
    WARM_HUMID = "Warm and Humid"
    MODERATE = "Moderate"
    COLD = "Cold"


class EquipmentType(str, Enum):
    """Supported equipment categories."""
    HVAC = "HVAC"
    LIGHTING = "Lighting"
    MOTORS_PUMPS = "Motors & Pumps"
    AIR_COMPRESSORS = "Air Compressors"
    REFRIGERATION = "Refrigeration"
    IT_EQUIPMENT = "IT Equipment"
    PROCESS_MACHINERY = "Process Machinery"
    OTHER = "Other"


class AnomalySeverity(str, Enum):
    """Severity levels for detected energy anomalies/waste."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EligibilityStatus(str, Enum):
    """Incentive program qualification status."""
    ELIGIBLE = "eligible"
    CONDITIONALLY_ELIGIBLE = "conditionally_eligible"
    NOT_ELIGIBLE = "not_eligible"
    UNDER_REVIEW = "under_review"
