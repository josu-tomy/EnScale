"""
Tests for constrained optimization and canonical modeled savings contract.
"""

from domain.models import Equipment, OptimizationResult
from services.optimization_service import compute_modeled_savings, OptimizationService


def test_savings_contract_formula():
    """
    Verifies canonical modeled savings formulas:
    energy_savings_kwh = baseline_energy_kwh - optimized_energy_kwh
    cost_savings_inr = baseline_cost_inr - optimized_cost_inr
    """
    baseline_energy_kwh = 1000.0
    optimized_energy_kwh = 850.0
    tariff_inr_per_kwh = 8.0

    res = compute_modeled_savings(
        baseline_energy_kwh=baseline_energy_kwh,
        optimized_energy_kwh=optimized_energy_kwh,
        tariff_inr_per_kwh=tariff_inr_per_kwh,
    )

    assert isinstance(res, OptimizationResult)
    assert res.energy_savings_kwh == 150.0
    assert res.baseline_cost_inr == 8000.0
    assert res.optimized_cost_inr == 6800.0
    assert res.cost_savings_inr == 1200.0
    assert res.savings_pct == 15.0


def test_optimizer_respects_constraints():
    service = OptimizationService()
    eq = Equipment(
        equipment_id="EQ-FLEX-01",
        equipment_type="HVAC",
        equipment_name="Variable Chiller",
        rated_power_kw=20.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=20,
        utilization_factor=1.0,
        minimum_hours=8.0,
        maximum_hours=12.0,
        is_flexible=True,
    )
    res = service.optimize_schedule([eq], tariff_inr_per_kwh=10.0, target_reduction_pct=30.0)

    assert res.baseline_energy_kwh == 4000.0
    assert res.optimized_energy_kwh == 3200.0
    assert res.energy_savings_kwh == 800.0
    assert res.cost_savings_inr == 8000.0
