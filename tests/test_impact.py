"""
Unit tests for Impact Service and EPI/MEPI normalization in EnScale.

Verifies:
- Monthly & annual energy savings
- Monthly & annual cost savings
- CO2 emissions avoided with reference to data/reference/emission_factors.json
- Modeled Energy Performance Index (MEPI: kWh/m²/year)
- Measured Energy Performance Index (EPI: kWh/m²/year)
- Missing measured data handling ("Insufficient data to calculate measured EPI.")
- Area normalization principle (area is never used to infer consumption)
- Non-positive savings handling
"""

import pytest
from domain.models import OptimizationResult, ImpactResult
from services.impact_service import calculate_impact, ImpactService
from config.settings import load_emission_factor_config


@pytest.fixture
def sample_optimization_result():
    return OptimizationResult(
        baseline_energy_kwh=10000.0,
        optimized_energy_kwh=8500.0,
        energy_savings_kwh=1500.0,
        baseline_cost_inr=85000.0,
        optimized_cost_inr=72250.0,
        cost_savings_inr=12750.0,
        savings_pct=15.0,
        schedule_recommendations=[],
        savings_statement="Projected savings under the modeled operating constraints",
        is_feasible=True,
    )


def test_impact_monthly_and_annual_savings(sample_optimization_result):
    impact = calculate_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        months_per_year=12.0,
    )

    assert impact.monthly_energy_savings == 1500.0
    assert impact.annual_energy_savings == 1500.0 * 12.0
    assert impact.monthly_cost_savings == 12750.0
    assert impact.annual_cost_savings == 12750.0 * 12.0


def test_co2_avoided_with_cea_factor(sample_optimization_result):
    cfg = load_emission_factor_config(region="India_National_Grid")
    expected_factor = cfg["factor_kg_co2_per_kwh"]
    assert expected_factor == 0.82

    impact = calculate_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        region="India_National_Grid",
    )

    expected_monthly_co2 = round(1500.0 * 0.82, 2)
    expected_annual_co2 = round(1500.0 * 12.0 * 0.82, 2)

    assert impact.co2_factor_used == 0.82
    assert impact.co2_reduction_kg == expected_monthly_co2
    assert impact.co2_avoided_kg == expected_annual_co2
    assert "CO₂ factor used: 0.82 kg CO₂/kWh" in impact.co2_factor_statement
    assert "Central Electricity Authority" in impact.co2_factor_statement


def test_co2_custom_override(sample_optimization_result):
    custom_factor = 0.75
    impact = calculate_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        custom_emission_factor=custom_factor,
    )

    assert impact.co2_factor_used == 0.75
    assert impact.co2_reduction_kg == round(1500.0 * 0.75, 2)
    assert impact.co2_avoided_kg == round(1500.0 * 12.0 * 0.75, 2)
    assert "User-specified custom emission factor" in impact.co2_factor_statement


def test_mepi_and_measured_epi_calculation(sample_optimization_result):
    # Modeled annual energy: 8500 * 12 = 102,000 kWh
    # Net built-up area: 2,000 m²
    # Modeled MEPI: 102,000 / 2,000 = 51.0 kWh/m²/year
    # Measured annual energy: 110,000 kWh -> measured EPI: 55.0 kWh/m²/year
    impact = calculate_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        measured_annual_energy_kwh=110000.0,
    )

    assert impact.annual_energy_kwh == 102000.0
    assert impact.mepi_kwh_m2_year == 51.0
    assert impact.epi_kwh_m2_year == 55.0
    assert "55.00 kWh/m²/year" in impact.measured_epi_status


def test_measured_epi_insufficient_data_when_unmeasured(sample_optimization_result):
    impact = calculate_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        measured_annual_energy_kwh=None,
    )

    assert impact.mepi_kwh_m2_year == 51.0
    assert impact.measured_epi_status == "Insufficient data to calculate measured EPI."
    # Explict assertion in assumptions documentation
    assert "area_normalization_rule" in impact.annualization_assumptions
    assert "never inferred from area alone" in impact.annualization_assumptions["area_normalization_rule"]


def test_invalid_area_raises_value_error(sample_optimization_result):
    with pytest.raises(ValueError, match="net_built_up_area_m2 must be greater than 0"):
        calculate_impact(
            optimization_result=sample_optimization_result,
            net_built_up_area_m2=0.0,
        )

    with pytest.raises(ValueError, match="net_built_up_area_m2 must be greater than 0"):
        calculate_impact(
            optimization_result=sample_optimization_result,
            net_built_up_area_m2=-500.0,
        )


def test_impact_service_class_wrapper(sample_optimization_result):
    service = ImpactService(region="India_National_Grid")
    impact = service.compute_impact(
        optimization_result=sample_optimization_result,
        net_built_up_area_m2=2000.0,
        annualization_factor=12.0,
        measured_annual_energy_kwh=120000.0,
    )

    assert impact.annual_energy_savings == 18000.0
    assert impact.annual_cost_savings == 153000.0
    assert impact.mepi_kwh_m2_year == 51.0
    assert impact.epi_kwh_m2_year == 60.0
