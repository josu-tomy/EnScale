"""
Canonical typed domain models for EnScale.

All field names here adhere strictly to the Canonical Variable Contract.
No aliases (area, sqft, floor_area, energy, power, cost) are permitted.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class BuildingProfile:
    """
    Profile representing a commercial or light-industrial building.
    Field names adhere strictly to the canonical building contract.
    """
    building_id: str
    building_type: str
    location_state: str
    climate_zone: str
    net_built_up_area_m2: float
    operating_start: int
    operating_end: int
    operating_days_per_month: int
    occupancy: float
    monthly_budget_inr: float
    tariff_inr_per_kwh: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildingProfile":
        return cls(
            building_id=str(data["building_id"]),
            building_type=str(data["building_type"]),
            location_state=str(data["location_state"]),
            climate_zone=str(data["climate_zone"]),
            net_built_up_area_m2=float(data["net_built_up_area_m2"]),
            operating_start=int(data["operating_start"]),
            operating_end=int(data["operating_end"]),
            operating_days_per_month=int(data["operating_days_per_month"]),
            occupancy=float(data["occupancy"]),
            monthly_budget_inr=float(data["monthly_budget_inr"]),
            tariff_inr_per_kwh=float(data["tariff_inr_per_kwh"]),
        )


@dataclass
class Equipment:
    """
    Equipment entity for inventory and equipment-based baseline calculations.
    Field names adhere strictly to the canonical equipment contract.
    """
    equipment_id: str
    equipment_type: str
    equipment_name: str
    rated_power_kw: float
    quantity: int
    hours_per_day: float
    operating_days: int
    utilization_factor: float
    minimum_hours: float
    maximum_hours: float
    is_flexible: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Equipment":
        return cls(
            equipment_id=str(data["equipment_id"]),
            equipment_type=str(data["equipment_type"]),
            equipment_name=str(data["equipment_name"]),
            rated_power_kw=float(data["rated_power_kw"]),
            quantity=int(data["quantity"]),
            hours_per_day=float(data["hours_per_day"]),
            operating_days=int(data["operating_days"]),
            utilization_factor=float(data["utilization_factor"]),
            minimum_hours=float(data["minimum_hours"]),
            maximum_hours=float(data["maximum_hours"]),
            is_flexible=bool(data["is_flexible"]),
        )


@dataclass
class EnergyRecord:
    """
    Single energy consumption record at a point in time.
    Required: timestamp, energy_kwh.
    Optional: temperature_c, occupancy, equipment_load_kw.
    """
    timestamp: Any
    energy_kwh: float
    temperature_c: Optional[float] = None
    occupancy: Optional[float] = None
    equipment_load_kw: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnergyRecord":
        return cls(
            timestamp=data["timestamp"],
            energy_kwh=float(data["energy_kwh"]),
            temperature_c=float(data["temperature_c"]) if data.get("temperature_c") is not None else None,
            occupancy=float(data["occupancy"]) if data.get("occupancy") is not None else None,
            equipment_load_kw=float(data["equipment_load_kw"]) if data.get("equipment_load_kw") is not None else None,
        )


@dataclass
class ForecastResult:
    """
    ML Energy Forecast output record.
    Adheres to canonical forecast fields.
    """
    timestamp: Any
    predicted_energy_kwh: float
    actual_energy_kwh: Optional[float] = None
    forecast_error_kwh: Optional[float] = None
    forecast_error_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyResult:
    """
    Energy anomaly and waste detection result record.
    """
    timestamp: Any
    actual_energy_kwh: float
    predicted_energy_kwh: float
    is_anomaly: bool = False
    forecast_error_kwh: float = 0.0
    forecast_error_pct: float = 0.0
    anomaly_flag: bool = False
    anomaly_score: float = 0.0
    waste_kwh: float = 0.0
    potential_waste_cost_inr: float = 0.0
    severity: str = "normal"
    description: str = ""

    def __post_init__(self):
        if self.is_anomaly and not self.anomaly_flag:
            self.anomaly_flag = True
        elif self.anomaly_flag and not self.is_anomaly:
            self.is_anomaly = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    """
    Constrained equipment scheduling and energy optimization result.
    Adheres strictly to canonical optimization fields.
    Note: These reflect MODELED savings, never guaranteed savings.
    """
    baseline_energy_kwh: float
    optimized_energy_kwh: float
    energy_savings_kwh: float
    baseline_cost_inr: float
    optimized_cost_inr: float
    cost_savings_inr: float
    savings_pct: float
    schedule_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    savings_statement: str = "Projected savings under the modeled operating constraints"
    is_feasible: bool = True
    status_message: str = "Feasible schedule found"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncentiveMatch:
    """
    Policy or utility incentive matched against the building or equipment profile.
    """
    incentive_id: str
    program_name: str
    target_equipment_type: str
    eligibility_status: str
    estimated_rebate_inr: float
    policy_reference: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactResult:
    """
    High-level ₹ / kWh / CO2 impact summary including EPI/MEPI normalization.
    """
    baseline_energy_kwh: float
    optimized_energy_kwh: float
    energy_savings_kwh: float
    cost_savings_inr: float
    co2_reduction_kg: float
    annual_energy_kwh: float
    epi_kwh_m2_year: Optional[float] = None
    monthly_energy_savings: float = 0.0
    annual_energy_savings: float = 0.0
    monthly_cost_savings: float = 0.0
    annual_cost_savings: float = 0.0
    co2_avoided_kg: float = 0.0
    co2_factor_used: float = 0.82
    co2_factor_statement: str = ""
    mepi_kwh_m2_year: Optional[float] = None
    measured_epi_status: str = ""
    annualization_assumptions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.monthly_energy_savings == 0.0 and self.energy_savings_kwh != 0.0:
            self.monthly_energy_savings = self.energy_savings_kwh
        if self.monthly_cost_savings == 0.0 and self.cost_savings_inr != 0.0:
            self.monthly_cost_savings = self.cost_savings_inr
        if self.co2_avoided_kg == 0.0 and self.co2_reduction_kg != 0.0:
            self.co2_avoided_kg = self.co2_reduction_kg
        if self.mepi_kwh_m2_year is None and self.epi_kwh_m2_year is not None:
            self.mepi_kwh_m2_year = self.epi_kwh_m2_year

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
