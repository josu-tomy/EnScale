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

from typing import List, Dict, Any, Optional
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
        Computes baseline metrics aggregated across equipment inventory.
        Uses canonical equipment energy formula.
        """
        total_baseline_kwh = 0.0
        equipment_breakdown = []

        for eq in equipment_list:
            eq_energy_kwh = calculate_equipment_energy_kwh(
                rated_power_kw=eq.rated_power_kw,
                quantity=eq.quantity,
                hours_per_day=eq.hours_per_day,
                operating_days=eq.operating_days,
                utilization_factor=eq.utilization_factor,
            )
            total_baseline_kwh += eq_energy_kwh
            equipment_breakdown.append({
                "equipment_id": eq.equipment_id,
                "equipment_name": eq.equipment_name,
                "equipment_type": eq.equipment_type,
                "estimated_energy_kwh": eq_energy_kwh,
            })

        return {
            "baseline_energy_kwh": total_baseline_kwh,
            "equipment_breakdown": equipment_breakdown,
        }
