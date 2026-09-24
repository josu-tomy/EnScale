"""
Impact calculation service for EnScale.

Computes comprehensive financial, energy, emissions (CO2), and EPI/MEPI normalization impacts:
- Monthly & annual modeled energy savings (kWh)
- Monthly & annual modeled cost savings (INR / ₹)
- CO2 emissions avoided (kg CO2)
- Modeled Energy Performance Index (MEPI: kWh/m²/year)
- Measured Energy Performance Index (EPI: kWh/m²/year)
- Explicit annualization assumptions and emission factor provenance
"""

from typing import Dict, Any, Optional
from domain.models import ImpactResult, OptimizationResult
from services.baseline_service import calculate_epi
from config.settings import load_emission_factor_config


def calculate_impact(
    optimization_result: OptimizationResult,
    net_built_up_area_m2: float,
    measured_annual_energy_kwh: Optional[float] = None,
    months_per_year: float = 12.0,
    region: str = "India_National_Grid",
    custom_emission_factor: Optional[float] = None,
    emission_factor_kg_per_kwh: Optional[float] = None,
) -> ImpactResult:
    """
    Computes holistic environmental, financial, and normalized EPI/MEPI impacts.

    Parameters:
        optimization_result: OptimizationResult containing baseline and optimized values
        net_built_up_area_m2: Net built-up floor area of the facility (m²)
        measured_annual_energy_kwh: Actual measured 12-month interval energy if available
        months_per_year: Operational annualization scaling factor (default 12.0)
        region: Geographic region key matching data/reference/emission_factors.json
        custom_emission_factor: Optional override for grid emission factor (kg CO2/kWh)
        emission_factor_kg_per_kwh: Backward-compatible alias for custom_emission_factor
    """
    if net_built_up_area_m2 <= 0:
        raise ValueError("net_built_up_area_m2 must be greater than 0 to compute EPI/MEPI.")

    # 1. Load emissions factor configuration
    effective_custom_factor = custom_emission_factor if custom_emission_factor is not None else emission_factor_kg_per_kwh
    factor_meta = load_emission_factor_config(region=region)
    if effective_custom_factor is not None:
        emission_factor = float(effective_custom_factor)
        source_desc = "User-specified custom emission factor"
    else:
        emission_factor = float(factor_meta.get("factor_kg_co2_per_kwh", 0.82))
        source_desc = str(factor_meta.get("source", "Central Electricity Authority (CEA)"))

    co2_factor_statement = f"CO₂ factor used: {emission_factor:.2f} kg CO₂/kWh (Source: {source_desc})"

    # 2. Modeled savings (monthly and annual)
    monthly_energy_savings = float(optimization_result.energy_savings_kwh)
    annual_energy_savings = monthly_energy_savings * months_per_year

    monthly_cost_savings = float(optimization_result.cost_savings_inr)
    annual_cost_savings = monthly_cost_savings * months_per_year

    # 3. Emissions reduction (CO2 avoided)
    co2_avoided_annual = annual_energy_savings * emission_factor
    co2_avoided_monthly = monthly_energy_savings * emission_factor

    # 4. Modeled Annual Energy and MEPI (Modeled Energy Performance Index)
    modeled_annual_energy_kwh = float(optimization_result.optimized_energy_kwh) * months_per_year
    mepi_val = calculate_epi(modeled_annual_energy_kwh, net_built_up_area_m2)

    # 5. Measured Annual Energy and EPI (Energy Performance Index)
    if measured_annual_energy_kwh is not None and measured_annual_energy_kwh > 0:
        measured_epi_val: Optional[float] = calculate_epi(measured_annual_energy_kwh, net_built_up_area_m2)
        measured_epi_status = f"{measured_epi_val:.2f} kWh/m²/year"
    else:
        measured_epi_val = None
        measured_epi_status = "Insufficient data to calculate measured EPI."

    # 6. Explicit assumptions documentation
    assumptions = {
        "annualization_scaling": f"{months_per_year:.1f} operational months per year",
        "emission_factor_kg_co2_per_kwh": emission_factor,
        "emission_factor_source": source_desc,
        "co2_factor_statement": co2_factor_statement,
        "mepi_definition": "MEPI (Modeled Energy Performance Index) = modeled_annual_energy_kwh / net_built_up_area_m2",
        "epi_definition": "EPI (Measured Energy Performance Index) = measured_annual_energy_kwh / net_built_up_area_m2",
        "area_normalization_rule": (
            "Electricity consumption is never inferred from area alone. Area is used solely as a "
            "normalizing denominator for benchmarking efficiency (kWh/m²/year)."
        ),
    }

    return ImpactResult(
        baseline_energy_kwh=round(optimization_result.baseline_energy_kwh, 2),
        optimized_energy_kwh=round(optimization_result.optimized_energy_kwh, 2),
        energy_savings_kwh=round(monthly_energy_savings, 2),
        cost_savings_inr=round(monthly_cost_savings, 2),
        co2_reduction_kg=round(co2_avoided_monthly, 2),
        annual_energy_kwh=round(modeled_annual_energy_kwh, 2),
        epi_kwh_m2_year=round(measured_epi_val, 2) if measured_epi_val is not None else round(mepi_val, 2),
        monthly_energy_savings=round(monthly_energy_savings, 2),
        annual_energy_savings=round(annual_energy_savings, 2),
        monthly_cost_savings=round(monthly_cost_savings, 2),
        annual_cost_savings=round(annual_cost_savings, 2),
        co2_avoided_kg=round(co2_avoided_annual, 2),
        co2_factor_used=emission_factor,
        co2_factor_statement=co2_factor_statement,
        mepi_kwh_m2_year=round(mepi_val, 2),
        measured_epi_status=measured_epi_status,
        annualization_assumptions=assumptions,
    )


class ImpactService:
    """Service class encapsulating energy, financial, CO2, and EPI/MEPI impact calculations."""

    def __init__(
        self,
        region: str = "India_National_Grid",
        custom_emission_factor: Optional[float] = None,
        emission_factor_kg_per_kwh: Optional[float] = None,
    ):
        self.region = region
        self.custom_emission_factor = (
            emission_factor_kg_per_kwh if emission_factor_kg_per_kwh is not None else custom_emission_factor
        )

    def compute_impact(
        self,
        optimization_result: OptimizationResult,
        net_built_up_area_m2: float,
        annualization_factor: float = 12.0,
        measured_annual_energy_kwh: Optional[float] = None,
    ) -> ImpactResult:
        """
        Computes ₹ / kWh / CO2 impact and normalized EPI/MEPI.
        """
        return calculate_impact(
            optimization_result=optimization_result,
            net_built_up_area_m2=net_built_up_area_m2,
            measured_annual_energy_kwh=measured_annual_energy_kwh,
            months_per_year=annualization_factor,
            region=self.region,
            custom_emission_factor=self.custom_emission_factor,
        )
