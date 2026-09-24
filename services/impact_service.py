"""
Impact calculation service for EnScale.

Computes financial, energy, emissions (CO2), and EPI normalization impacts:
- ₹ (INR) savings
- kWh energy savings
- kg CO2 reduction
- EPI (kWh/m2/year)
"""

from typing import Dict, Any
from domain.models import ImpactResult, OptimizationResult
from services.baseline_service import calculate_epi
from config.constants import DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH


class ImpactService:
    """Service skeleton for holistic impact quantification."""

    def __init__(self, emission_factor_kg_per_kwh: float = DEFAULT_GRID_EMISSION_FACTOR_KG_PER_KWH):
        self.emission_factor = emission_factor_kg_per_kwh

    def compute_impact(
        self,
        optimization_result: OptimizationResult,
        net_built_up_area_m2: float,
        annualization_factor: float = 12.0,
    ) -> ImpactResult:
        """
        Computes ₹ / kWh / CO2 impact and normalized EPI.
        """
        co2_reduction_kg = optimization_result.energy_savings_kwh * self.emission_factor
        annual_energy_kwh = optimization_result.optimized_energy_kwh * annualization_factor
        epi_val = calculate_epi(annual_energy_kwh, net_built_up_area_m2)

        return ImpactResult(
            baseline_energy_kwh=optimization_result.baseline_energy_kwh,
            optimized_energy_kwh=optimization_result.optimized_energy_kwh,
            energy_savings_kwh=optimization_result.energy_savings_kwh,
            cost_savings_inr=optimization_result.cost_savings_inr,
            co2_reduction_kg=co2_reduction_kg,
            annual_energy_kwh=annual_energy_kwh,
            epi_kwh_m2_year=epi_val,
        )
