"""
Energy baseline service for EnScale.

Enforces:
1. Equipment Energy Formula:
   estimated_energy_kwh = rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor
   (Building area alone is NEVER used to calculate energy consumption)
2. EPI Contract:
   epi_kwh_m2_year = annual_energy_kwh / net_built_up_area_m2
   (EPI is a normalized metric; area does not determine electricity consumption by itself)
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd

from domain.models import Equipment, BuildingProfile


def calculate_equipment_energy_kwh(
    rated_power_kw: float,
    quantity: int,
    hours_per_day: float,
    operating_days: int,
    utilization_factor: float,
) -> float:
    """
    Canonical Equipment Energy Formula:
    estimated_energy_kwh = rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor
    """
    return float(rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor)


def calculate_equipment_energy(equipment_list: List[Union[Equipment, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Computes equipment baseline according to canonical formula:
    estimated_energy_kwh = rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor

    Returns:
    - per_equipment_energy: list of individual equipment calculations with full calculation trace
    - total_energy: total aggregated energy consumption across all equipment (kWh)
    - assumptions: documented mathematical model and traceability details
    """
    per_equipment: List[Dict[str, Any]] = []
    total_energy: float = 0.0

    for item in equipment_list:
        if isinstance(item, Equipment):
            eq_id = item.equipment_id
            eq_name = item.equipment_name
            eq_type = item.equipment_type
            rated_power = item.rated_power_kw
            qty = item.quantity
            hpd = item.hours_per_day
            op_days = item.operating_days
            util = item.utilization_factor
        else:
            eq_id = str(item.get("equipment_id", "UNKNOWN"))
            eq_name = str(item.get("equipment_name", "Unknown Equipment"))
            eq_type = str(item.get("equipment_type", "Other"))
            rated_power = float(item["rated_power_kw"])
            qty = int(item["quantity"])
            hpd = float(item["hours_per_day"])
            op_days = int(item["operating_days"])
            util = float(item["utilization_factor"])

        kwh = calculate_equipment_energy_kwh(
            rated_power_kw=rated_power,
            quantity=qty,
            hours_per_day=hpd,
            operating_days=op_days,
            utilization_factor=util,
        )
        total_energy += kwh

        trace = (
            f"{rated_power} kW * {qty} unit(s) * {hpd} hrs/day * "
            f"{op_days} days * {util} util = {kwh:.2f} kWh"
        )

        per_equipment.append({
            "equipment_id": eq_id,
            "equipment_name": eq_name,
            "equipment_type": eq_type,
            "rated_power_kw": rated_power,
            "quantity": qty,
            "hours_per_day": hpd,
            "operating_days": op_days,
            "utilization_factor": util,
            "estimated_energy_kwh": round(kwh, 2),
            "calculation_trace": trace,
        })

    assumptions = {
        "formula": (
            "estimated_energy_kwh = rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor"
        ),
        "traceability": "Every kWh is explicitly mapped back to individual equipment operational parameters.",
        "area_independence": (
            "Energy consumption is derived strictly from rated mechanical/electrical equipment loads, "
            "not floor area."
        ),
        "equipment_count": len(equipment_list),
    }

    return {
        "per_equipment_energy": per_equipment,
        "total_energy": round(total_energy, 2),
        "assumptions": assumptions,
    }


def calculate_epi(annual_energy_kwh: float, net_built_up_area_m2: float) -> float:
    """
    Canonical EPI calculation (kWh/m2/year).
    EPI = annual_energy_kwh / net_built_up_area_m2

    Area does not determine electricity consumption by itself. EPI is a normalized metric.
    """
    if net_built_up_area_m2 <= 0:
        raise ValueError("net_built_up_area_m2 must be greater than 0 to calculate EPI")
    return float(annual_energy_kwh / net_built_up_area_m2)


class BaselineService:
    """Service for computing energy baselines via historical data or equipment inventory."""

    def compute_historical_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes baseline metrics from historical energy records.
        """
        total_kwh = float(df["energy_kwh"].sum())
        mean_kwh = float(df["energy_kwh"].mean())
        peak_kwh = float(df["energy_kwh"].max())
        return {
            "baseline_energy_kwh": total_kwh,
            "mean_energy_kwh": mean_kwh,
            "peak_energy_kwh": peak_kwh,
            "record_count": len(df),
        }

    def compute_equipment_baseline(self, equipment_list: List[Equipment]) -> Dict[str, Any]:
        """
        Computes baseline metrics aggregated across equipment inventory using calculate_equipment_energy.
        """
        calc = calculate_equipment_energy(equipment_list)
        return {
            "baseline_energy_kwh": calc["total_energy"],
            "equipment_breakdown": calc["per_equipment_energy"],
            "assumptions": calc["assumptions"],
        }
