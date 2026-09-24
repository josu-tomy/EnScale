"""
Constrained energy optimization service for EnScale.

Enforces:
1. Savings Contract:
   energy_savings_kwh = baseline_energy_kwh - optimized_energy_kwh
   cost_savings_inr = baseline_cost_inr - optimized_cost_inr
   savings_pct = (energy_savings_kwh / baseline_energy_kwh) * 100

IMPORTANT:
These are MODELED savings based on equipment scheduling, peak shaving, and setpoint adjustments.
They are NEVER guaranteed savings.
"""

from typing import List, Dict, Any
from domain.models import Equipment, OptimizationResult


def compute_modeled_savings(
    baseline_energy_kwh: float,
    optimized_energy_kwh: float,
    tariff_inr_per_kwh: float,
) -> OptimizationResult:
    """
    Computes modeled savings according to the Canonical Savings Contract.
    """
    energy_savings_kwh = float(baseline_energy_kwh - optimized_energy_kwh)
    baseline_cost_inr = float(baseline_energy_kwh * tariff_inr_per_kwh)
    optimized_cost_inr = float(optimized_energy_kwh * tariff_inr_per_kwh)
    cost_savings_inr = float(baseline_cost_inr - optimized_cost_inr)

    savings_pct = (
        float((energy_savings_kwh / baseline_energy_kwh) * 100.0)
        if baseline_energy_kwh > 0
        else 0.0
    )

    return OptimizationResult(
        baseline_energy_kwh=float(baseline_energy_kwh),
        optimized_energy_kwh=float(optimized_energy_kwh),
        energy_savings_kwh=energy_savings_kwh,
        baseline_cost_inr=baseline_cost_inr,
        optimized_cost_inr=optimized_cost_inr,
        cost_savings_inr=cost_savings_inr,
        savings_pct=savings_pct,
    )


class OptimizationService:
    """Service skeleton for constrained equipment scheduling and energy optimization."""

    def optimize_schedule(
        self,
        equipment_list: List[Equipment],
        tariff_inr_per_kwh: float,
        target_reduction_pct: float = 10.0,
    ) -> OptimizationResult:
        """
        Generates optimized equipment schedules respecting constraints (min_hours, max_hours, is_flexible).
        Returns OptimizationResult containing modeled savings.
        """
        baseline_energy_kwh = 0.0
        optimized_energy_kwh = 0.0
        recommendations = []

        for eq in equipment_list:
            eq_base_kwh = eq.rated_power_kw * eq.quantity * eq.hours_per_day * eq.operating_days * eq.utilization_factor
            baseline_energy_kwh += eq_base_kwh

            if eq.is_flexible:
                opt_hours = max(eq.minimum_hours, eq.hours_per_day * (1.0 - target_reduction_pct / 100.0))
                eq_opt_kwh = eq.rated_power_kw * eq.quantity * opt_hours * eq.operating_days * eq.utilization_factor
                recommendations.append({
                    "equipment_id": eq.equipment_id,
                    "action": f"Adjust operating hours from {eq.hours_per_day}h to {opt_hours:.1f}h per day",
                    "modeled_reduction_kwh": eq_base_kwh - eq_opt_kwh,
                })
            else:
                eq_opt_kwh = eq_base_kwh
            optimized_energy_kwh += eq_opt_kwh

        result = compute_modeled_savings(
            baseline_energy_kwh=baseline_energy_kwh,
            optimized_energy_kwh=optimized_energy_kwh,
            tariff_inr_per_kwh=tariff_inr_per_kwh,
        )
        result.schedule_recommendations = recommendations
        return result
